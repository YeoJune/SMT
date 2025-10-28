"""
Windowed Transformer Decoder
Wrapper around pre-trained GPT-2 for processing small windows
"""

import torch
import torch.nn as nn


class WindowedTransformer(nn.Module):
    """
    Windowed Transformer using pre-trained GPT-2
    
    Processes ONLY the window (65 tokens), not full context
    No positional encoding (uses relative positions within window)
    """
    
    def __init__(self, d_model=768, n_layers=12, n_heads=12, d_ff=3072,
                 dropout=0.1, pretrained_name="gpt2"):
        """
        Args:
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            pretrained_name: HuggingFace model name (or None for random init)
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        if pretrained_name:
            try:
                from transformers import GPT2Model, GPT2Config
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers")
            
            # Load pre-trained GPT-2
            print(f"Loading pre-trained transformer from {pretrained_name}...")
            self.transformer = GPT2Model.from_pretrained(pretrained_name)
            
            # Freeze embedding (we'll use external embedding)
            self.transformer.wte.weight.requires_grad = False
            
            print(f"✅ Loaded {n_layers}-layer transformer (pretrained)")
        else:
            # Random initialization
            print(f"Initializing random {n_layers}-layer transformer...")
            from transformers import GPT2Model, GPT2Config
            
            config = GPT2Config(
                vocab_size=1,  # Not used (we use inputs_embeds)
                n_positions=1024,
                n_embd=d_model,
                n_layer=n_layers,
                n_head=n_heads,
                n_inner=d_ff,
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
            )
            self.transformer = GPT2Model(config)
            print(f"✅ Initialized {n_layers}-layer transformer (random)")
        
    def forward(self, x, attention_mask=None):
        """
        Forward pass through transformer
        
        Args:
            x: (batch, window_size, d_model) - Window embeddings
            attention_mask: Optional attention mask
            
        Returns:
            output: (batch, window_size, d_model) - Processed embeddings
        """
        B, S, D = x.shape
        
        # GPT-2 expects input_ids, but we provide embeddings directly
        # Use inputs_embeds parameter
        outputs = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=False,  # Don't use KV cache (we're not generating)
            output_hidden_states=False,
            return_dict=True
        )
        
        # Get last hidden state
        output = outputs.last_hidden_state  # (batch, window_size, d_model)
        
        return output
    
    def get_params_by_group(self):
        """
        Get parameters grouped by layer depth
        Useful for layer-wise learning rate decay
        
        Returns:
            List of (name, param) tuples
        """
        return list(self.transformer.named_parameters())


class SimpleCausalTransformer(nn.Module):
    """
    Lightweight causal transformer if you don't want to use GPT-2
    """
    
    def __init__(self, d_model=768, n_layers=12, n_heads=12, 
                 d_ff=3072, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        from torch.nn import TransformerDecoder, TransformerDecoderLayer
        
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm like GPT-2
        )
        
        self.transformer = TransformerDecoder(
            decoder_layer,
            num_layers=n_layers
        )
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: Optional mask
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, S, D = x.shape
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        # TransformerDecoder needs tgt and memory
        # For decoder-only, we use x as both
        output = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=causal_mask
        )
        
        return output


if __name__ == "__main__":
    # Test transformer
    print("Testing WindowedTransformer...")
    
    batch_size = 2
    window_size = 65
    d_model = 768
    
    # Test with GPT-2
    try:
        transformer = WindowedTransformer(
            d_model=d_model,
            n_layers=12,
            n_heads=12,
            pretrained_name="gpt2"
        )
        
        x = torch.randn(batch_size, window_size, d_model)
        output = transformer(x)
        
        print(f"✅ Input:  {x.shape}")
        print(f"✅ Output: {output.shape}")
        
        # Count parameters
        n_params = sum(p.numel() for p in transformer.parameters())
        n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        print(f"✅ Parameters: {n_params/1e6:.1f}M ({n_trainable/1e6:.1f}M trainable)")
        
    except ImportError:
        print("⚠️  transformers not installed, skipping GPT-2 test")
    
    # Test simple version
    print("\nTesting SimpleCausalTransformer...")
    simple_transformer = SimpleCausalTransformer(
        d_model=d_model,
        n_layers=6,
        n_heads=12
    )
    
    x = torch.randn(batch_size, window_size, d_model)
    output = simple_transformer(x)
    
    print(f"✅ Input:  {x.shape}")
    print(f"✅ Output: {output.shape}")
