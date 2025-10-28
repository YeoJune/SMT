"""
SSM (Mamba) Wrapper
Wrapper around pre-trained Mamba for compressed memory
"""

import torch
import torch.nn as nn


class SSMMemory(nn.Module):
    """
    SSM-based memory using pre-trained Mamba
    
    Takes pooled representation and updates internal state
    Outputs a new compressed memory vector
    """
    
    def __init__(self, d_model=768, n_layers=24, d_state=16,
                 d_conv=4, expand_factor=2, pretrained_name="state-spaces/mamba-130m-hf"):
        """
        Args:
            d_model: Model dimension
            n_layers: Number of Mamba layers
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand_factor: Expansion factor in Mamba
            pretrained_name: HuggingFace model name
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        try:
            from mamba_ssm import Mamba as MambaCUDA
            from mamba_ssm.ops.triton.layer_norm import RMSNorm
            self.use_cuda = True
        except ImportError:
            print("⚠️  mamba-ssm not available. Install with:")
            print("   pip install mamba-ssm causal-conv1d")
            raise ImportError("mamba-ssm required")
        
        print(f"Loading pre-trained SSM from {pretrained_name}...")
        
        # Create Mamba layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layer_norms.append(RMSNorm(d_model))
            self.layers.append(
                MambaCUDA(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand_factor
                )
            )
        
        # Final norm
        self.norm_f = RMSNorm(d_model)
        
        print(f"✅ Created {n_layers}-layer SSM")
    
    def forward(self, x):
        """
        Process input through SSM
        
        Args:
            x: (batch, d_model) or (batch, 1, d_model) - Pooled representation
            
        Returns:
            output: (batch, d_model) - Compressed memory
        """
        # Ensure 3D: (batch, seq_len=1, d_model)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Process through all layers
        for i, layer in enumerate(self.layers):
            residual = x
            x_norm = self.layer_norms[i](x)
            x_out = layer(x_norm)
            x = residual + x_out
        
        # Final norm
        x = self.norm_f(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch, d_model)
        
        return x
    
    def load_pretrained_weights(self, pretrained_name="state-spaces/mamba-130m-hf"):
        """
        Load pre-trained weights from HuggingFace
        
        Args:
            pretrained_name: Model name on HuggingFace
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        print(f"Loading weights from {pretrained_name}...")
        hf_model = AutoModelForCausalLM.from_pretrained(pretrained_name)
        hf_state_dict = hf_model.state_dict()
        
        # Load layer norms
        for i in range(self.n_layers):
            norm_key = f"backbone.layers.{i}.norm.weight"
            if norm_key in hf_state_dict:
                self.layer_norms[i].weight.data.copy_(
                    hf_state_dict[norm_key]
                )
        
        # Load Mamba layers
        for i in range(self.n_layers):
            prefix = f"backbone.layers.{i}.mixer."
            layer_dict = {}
            
            for key, value in hf_state_dict.items():
                if key.startswith(prefix):
                    local_key = key.replace(prefix, "")
                    layer_dict[local_key] = value
            
            if len(layer_dict) > 0:
                self.layers[i].load_state_dict(layer_dict, strict=False)
        
        # Load final norm
        if "backbone.norm_f.weight" in hf_state_dict:
            self.norm_f.weight.data.copy_(
                hf_state_dict["backbone.norm_f.weight"]
            )
        
        print(f"✅ Loaded pre-trained weights successfully")


class SimpleSSM(nn.Module):
    """
    Simplified SSM for testing (without mamba-ssm)
    Just a stack of linear layers with GELU
    """
    
    def __init__(self, d_model=768, n_layers=4, dropout=0.1):
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
                nn.Dropout(dropout)
            ])
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch, d_model) or (batch, 1, d_model)
        Returns:
            output: (batch, d_model)
        """
        if x.dim() == 2:
            return self.layers(x)
        elif x.dim() == 3:
            B, S, D = x.shape
            x = x.view(-1, D)
            output = self.layers(x)
            return output.view(B, S, D).squeeze(1) if S == 1 else output.view(B, S, D)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")


if __name__ == "__main__":
    # Test SSM
    print("Testing SSMMemory...")
    
    batch_size = 2
    d_model = 768
    
    try:
        ssm = SSMMemory(
            d_model=d_model,
            n_layers=24,
            d_state=16,
            pretrained_name="state-spaces/mamba-130m-hf"
        )
        
        # Test forward
        x = torch.randn(batch_size, d_model).cuda()
        output = ssm(x)
        
        print(f"✅ Input:  {x.shape}")
        print(f"✅ Output: {output.shape}")
        
        # Count parameters
        n_params = sum(p.numel() for p in ssm.parameters())
        print(f"✅ Parameters: {n_params/1e6:.1f}M")
        
        # Test with pretrained weights
        # ssm.load_pretrained_weights()
        
    except (ImportError, RuntimeError) as e:
        print(f"⚠️  Could not test with mamba-ssm: {e}")
        print("   Testing with SimpleSSM instead...")
        
        ssm = SimpleSSM(d_model=d_model, n_layers=4)
        x = torch.randn(batch_size, d_model)
        output = ssm(x)
        
        print(f"✅ Input:  {x.shape}")
        print(f"✅ Output: {output.shape}")
