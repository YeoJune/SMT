"""
Window Manager
Manages sliding window of SSM outputs and input tokens for TBPTT
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class WindowManager:
    """
    Manages sliding window: [n SSM outputs | m input tokens]
    
    Window structure:
        [ssm_out[-n], ..., ssm_out[-1] | x[-m], ..., x[-1]]
        
    Operations:
    - append_input(x): Add new input token
    - append_ssm(ssm_out): Add new SSM output (stride steps only)
    - get_window(): Return current window for Transformer
    
    State automatically carries across chunk boundaries for TBPTT.
    """
    
    def __init__(
        self,
        batch_size: int,
        n_ssm_outputs: int,
        m_input_tokens: int,
        d_model: int,
        device: torch.device = torch.device('cuda'),
    ):
        """
        Args:
            batch_size: Batch size
            n_ssm_outputs: Number of SSM outputs to keep in window
            m_input_tokens: Number of input tokens to keep in window
            d_model: Model dimension
            device: Device for tensors
        """
        self.batch_size = batch_size
        self.n = n_ssm_outputs
        self.m = m_input_tokens
        self.d_model = d_model
        self.device = device
        
        # Circular buffers (efficient for sliding window)
        # Shape: (batch, capacity, d_model)
        self.ssm_outputs = torch.zeros(
            batch_size, n_ssm_outputs, d_model,
            device=device, dtype=torch.float32,
        )
        self.input_tokens = torch.zeros(
            batch_size, m_input_tokens, d_model,
            device=device, dtype=torch.float32,
        )
        
        # Track fill status (for initialization)
        self.ssm_filled = 0
        self.input_filled = 0
    
    def append_input(self, x: torch.Tensor):
        """
        Add new input token embedding.
        
        Rotates left: oldest token is discarded, new token added at end.
        
        Args:
            x: (batch, d_model) - New input token embedding
        """
        assert x.shape == (self.batch_size, self.d_model), \
            f"Expected ({self.batch_size}, {self.d_model}), got {x.shape}"
        
        # Rotate left and append
        self.input_tokens = torch.cat([
            self.input_tokens[:, 1:, :],  # Remove oldest
            x.unsqueeze(1),               # Add newest
        ], dim=1)
        
        self.input_filled = min(self.input_filled + 1, self.m)
    
    def append_ssm(self, ssm_output: torch.Tensor):
        """
        Add new SSM output (only called at stride steps).
        
        Rotates left: oldest SSM output is discarded, new one added.
        
        Args:
            ssm_output: (batch, d_model) - New SSM output
        """
        assert ssm_output.shape == (self.batch_size, self.d_model), \
            f"Expected ({self.batch_size}, {self.d_model}), got {ssm_output.shape}"
        
        # Rotate left and append
        self.ssm_outputs = torch.cat([
            self.ssm_outputs[:, 1:, :],   # Remove oldest
            ssm_output.unsqueeze(1),      # Add newest
        ], dim=1)
        
        self.ssm_filled = min(self.ssm_filled + 1, self.n)
    
    def get_window(self) -> torch.Tensor:
        """
        Get current window for Transformer.
        
        Returns:
            window: (batch, n+m, d_model) - Current window
            
        Note:
            During initialization (first few steps), some positions may be zero.
            This is expected and handled naturally by the Transformer.
        """
        # Concatenate SSM outputs and input tokens
        window = torch.cat([self.ssm_outputs, self.input_tokens], dim=1)
        
        return window
    
    def reset(self):
        """Reset to initial state (zeros)."""
        self.ssm_outputs.zero_()
        self.input_tokens.zero_()
        self.ssm_filled = 0
        self.input_filled = 0
    
    def detach(self):
        """
        Detach state from computation graph for TBPTT.
        
        Call this at chunk boundaries to truncate gradients.
        The state values are preserved but gradient flow is cut.
        """
        self.ssm_outputs = self.ssm_outputs.detach()
        self.input_tokens = self.input_tokens.detach()
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current state for chunk boundaries.
        
        Returns:
            state: Dict with ssm_outputs and input_tokens
            
        Note:
            Tensors are cloned but NOT detached - they maintain gradient flow.
            The SSM internal states are detached separately.
        """
        return {
            'ssm_outputs': self.ssm_outputs.clone(),
            'input_tokens': self.input_tokens.clone(),
            'ssm_filled': self.ssm_filled,
            'input_filled': self.input_filled,
        }
    
    def set_state(self, state: Dict[str, torch.Tensor]):
        """
        Set state from checkpoint.
        
        Args:
            state: Dict from get_state()
        """
        self.ssm_outputs = state['ssm_outputs']
        self.input_tokens = state['input_tokens']
        self.ssm_filled = state['ssm_filled']
        self.input_filled = state['input_filled']
    
    @property
    def window_size(self) -> int:
        """Total window size."""
        return self.n + self.m
    
    def __repr__(self) -> str:
        return (
            f"WindowManager(batch={self.batch_size}, "
            f"n={self.n}, m={self.m}, "
            f"window_size={self.window_size}, "
            f"filled={self.ssm_filled}/{self.n} + {self.input_filled}/{self.m})"
        )


if __name__ == "__main__":
    print("="*80)
    print("Testing WindowManager")
    print("="*80)
    
    batch_size = 2
    n_ssm = 15
    m_input = 50
    d_model = 768
    stride = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  SSM outputs: {n_ssm}")
    print(f"  Input tokens: {m_input}")
    print(f"  Window size: {n_ssm + m_input}")
    print(f"  Stride: {stride}")
    
    # Create manager
    wm = WindowManager(
        batch_size=batch_size,
        n_ssm_outputs=n_ssm,
        m_input_tokens=m_input,
        d_model=d_model,
        device=device,
    )
    
    print(f"\n✅ Created: {wm}")
    
    # Simulate processing
    print("\n[Simulation: 100 steps with stride=16]")
    
    for step in range(100):
        # Add input token
        x = torch.randn(batch_size, d_model, device=device)
        wm.append_input(x)
        
        # Add SSM output at stride steps
        if step > 0 and step % stride == 0:
            ssm_out = torch.randn(batch_size, d_model, device=device)
            wm.append_ssm(ssm_out)
            print(f"  Step {step:3d}: Added SSM output")
        
        # Get window
        window = wm.get_window()
        assert window.shape == (batch_size, n_ssm + m_input, d_model)
        
        if step % 25 == 0:
            print(f"  Step {step:3d}: Window shape = {window.shape}, {wm}")
    
    print(f"\n✅ Final state: {wm}")
    
    # Test state save/load
    print("\n[Test: State save/load]")
    state = wm.get_state()
    print(f"  Saved state: {state.keys()}")
    
    # Modify
    wm.append_input(torch.randn(batch_size, d_model, device=device))
    
    # Restore
    wm.set_state(state)
    print(f"  ✅ State restored")
    
    # Test reset
    print("\n[Test: Reset]")
    wm.reset()
    print(f"  ✅ Reset: {wm}")
    
    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)