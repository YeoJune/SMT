"""
SSM (Mamba) Memory Module
Stateful compression for long-range context with TBPTT support
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class SSMMemory(nn.Module):
    """
    Stateful SSM-based memory using Mamba
    
    Maintains internal state across forward calls for efficient
    long-context processing with TBPTT (Truncated Backpropagation Through Time).
    
    Usage:
        # Initialize
        ssm = SSMMemory(d_model=768, n_layers=24)
        
        # Training with TBPTT
        ssm.init_cache(batch_size=4, device='cuda')
        for chunk in chunks:
            for step in chunk:
                if step % stride == 0:
                    output = ssm(pooled_vector)  # Uses and updates cache
            ssm.detach_cache()  # Truncate gradients between chunks
        
        # Inference
        ssm.init_cache(batch_size=1, device='cuda')
        for step in sequence:
            output = ssm(vector)  # Maintains state across all steps
    """
    
    def __init__(
        self, 
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ):
        """
        Args:
            d_model: Model dimension
            n_layers: Number of Mamba layers
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand_factor: Expansion factor for Mamba
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        
        # Import Mamba
        try:
            from mamba_ssm import Mamba as MambaCUDA
            from mamba_ssm.ops.triton.layer_norm import RMSNorm
        except ImportError:
            raise ImportError(
                "mamba-ssm required. Install with:\n"
                "  pip install mamba-ssm causal-conv1d"
            )
        
        # Create Mamba layers with norms
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(n_layers):
            self.norms.append(RMSNorm(d_model))
            self.layers.append(
                MambaCUDA(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand_factor,
                    layer_idx=i,  # Required for state management
                )
            )
        
        self.norm_f = RMSNorm(d_model)
        
        # State cache for TBPTT
        self._conv_states: Optional[List[torch.Tensor]] = None
        self._ssm_states: Optional[List[torch.Tensor]] = None
        
        print(f"✅ SSMMemory: {n_layers} layers, {d_model}d")
    
    def init_cache(self, batch_size: int, device: torch.device):
        """
        Initialize state cache for recurrent processing.
        
        Call this at the start of each sequence (training or inference).
        
        Args:
            batch_size: Batch size
            device: Device to allocate tensors
        """
        self._conv_states = []
        self._ssm_states = []
        
        d_inner = self.d_model * self.expand_factor
        
        for layer in self.layers:
            # Convolution state: last d_conv inputs
            conv_state = torch.zeros(
                batch_size,
                d_inner,
                self.d_conv,
                device=device,
                dtype=layer.conv1d.weight.dtype,
            )
            
            # SSM state: accumulated hidden state
            ssm_state = torch.zeros(
                batch_size,
                d_inner,
                self.d_state,
                device=device,
                dtype=layer.dt_proj.weight.dtype,
            )
            
            self._conv_states.append(conv_state)
            self._ssm_states.append(ssm_state)
    
    def reset_cache(self):
        """Reset cache to zeros (keeps allocated tensors)."""
        if self._conv_states is not None:
            for state in self._conv_states:
                state.zero_()
            for state in self._ssm_states:
                state.zero_()
    
    def detach_cache(self):
        """
        Detach cache from computation graph.
        
        Call this at chunk boundaries for TBPTT to truncate gradients.
        """
        if self._conv_states is not None:
            self._conv_states = [s.detach() for s in self._conv_states]
            self._ssm_states = [s.detach() for s in self._ssm_states]
    
    def clear_cache(self):
        """Clear cache entirely (free memory)."""
        self._conv_states = None
        self._ssm_states = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through SSM with state.
        
        Args:
            x: (batch, d_model) - Single input vector
        
        Returns:
            output: (batch, d_model) - Processed output
        
        Note:
            - If cache is initialized, uses recurrent mode (step-by-step)
            - If cache is None, falls back to parallel mode (less efficient)
        """
        # Ensure 2D input
        if x.dim() == 3:
            assert x.size(1) == 1, "For cached mode, use single timestep"
            x = x.squeeze(1)
        
        assert x.dim() == 2, f"Expected (B, D), got {x.shape}"
        
        # Check if cache is available
        if self._conv_states is None:
            return self._forward_parallel(x)
        else:
            return self._forward_recurrent(x)
    
    def _forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recurrent forward with state (efficient).
        
        Uses Mamba's step() method to update states in-place.
        """
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Residual connection
            residual = x
            
            # Pre-norm
            x_norm = norm(x)
            
            # Mamba step (updates states in-place)
            x_out, _, _ = layer.step(
                x_norm,
                self._conv_states[i],
                self._ssm_states[i],
            )
            
            # Residual
            x = residual + x_out
        
        # Final norm
        x = self.norm_f(x)
        
        return x
    
    def _forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel forward without state (fallback).
        
        Less efficient but works without cache initialization.
        """
        # Add sequence dimension
        x = x.unsqueeze(1)  # (B, 1, D)
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x
            x_norm = norm(x)
            x_out = layer(x_norm)
            x = residual + x_out
        
        x = self.norm_f(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (B, D)
        
        return x
    
    def get_cache_state(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get current cache state (for checkpointing).
        
        Returns:
            (conv_states, ssm_states): Lists of state tensors
        """
        if self._conv_states is None:
            raise RuntimeError("Cache not initialized")
        
        return (
            [s.clone() for s in self._conv_states],
            [s.clone() for s in self._ssm_states],
        )
    
    def set_cache_state(
        self,
        conv_states: List[torch.Tensor],
        ssm_states: List[torch.Tensor],
        detach: bool = False,
    ):
        """
        Set cache state (for checkpointing).
        
        Args:
            conv_states: List of convolution states
            ssm_states: List of SSM states
            detach: If True, detach from computation graph
        """
        self._conv_states = conv_states
        self._ssm_states = ssm_states
        
        if detach:
            self.detach_cache()


class SimpleSSM(nn.Module):
    """
    Simplified SSM for testing without mamba-ssm.
    
    Uses stacked linear layers with GELU activation.
    Not as powerful as Mamba but useful for debugging.
    """
    
    def __init__(self, d_model: int = 768, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            ])
        
        self.layers = nn.Sequential(*layers)
        
        print(f"✅ SimpleSSM: {n_layers} layers (fallback)")
    
    def init_cache(self, batch_size: int, device: torch.device):
        """No-op for compatibility."""
        pass
    
    def reset_cache(self):
        """No-op for compatibility."""
        pass
    
    def detach_cache(self):
        """No-op for compatibility."""
        pass
    
    def clear_cache(self):
        """No-op for compatibility."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
        Returns:
            output: (batch, d_model)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.layers(x)


if __name__ == "__main__":
    print("="*80)
    print("Testing SSMMemory")
    print("="*80)
    
    batch_size = 2
    d_model = 768
    n_steps = 100
    
    try:
        # Test with Mamba
        print("\n[Test 1: Recurrent Mode]")
        ssm = SSMMemory(d_model=d_model, n_layers=24)
        
        # Initialize cache
        ssm.init_cache(batch_size=batch_size, device='cuda')
        
        # Process sequence
        x = torch.randn(batch_size, d_model).cuda()
        outputs = []
        
        for step in range(n_steps):
            out = ssm(x)
            outputs.append(out)
            
            if step % 10 == 0:
                print(f"  Step {step}: output shape = {out.shape}")
        
        print(f"✅ Processed {n_steps} steps in recurrent mode")
        
        # Test detach
        print("\n[Test 2: Detach Cache]")
        ssm.detach_cache()
        out = ssm(x)
        print(f"✅ Forward after detach works: {out.shape}")
        
        # Test parallel fallback
        print("\n[Test 3: Parallel Mode (no cache)]")
        ssm.clear_cache()
        out = ssm(x)
        print(f"✅ Parallel mode works: {out.shape}")
        
        # Parameter count
        n_params = sum(p.numel() for p in ssm.parameters())
        print(f"\n📊 Parameters: {n_params/1e6:.1f}M")
        
    except ImportError as e:
        print(f"\n⚠️  {e}")
        print("Testing SimpleSSM instead...\n")
        
        ssm = SimpleSSM(d_model=d_model, n_layers=4)
        x = torch.randn(batch_size, d_model)
        out = ssm(x)
        print(f"✅ SimpleSSM output: {out.shape}")
    
    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)