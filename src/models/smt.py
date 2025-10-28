"""
SMT (Stride Memory Transformer) Model
Main architecture combining window, pooling, and SSM
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .components.attention_pooling import AttentionPooling
from .components.transformer import WindowedTransformer
from .components.ssm import SSMMemory
from .window_manager import WindowManager, BatchedWindowManager


class StrideMemoryTransformer(nn.Module):
    """
    SMT (Stride Memory Transformer)
    
    Architecture:
    1. Input → Embedding
    2. Window = [n SSM outputs | m recent tokens]
    3. Transformer processes window → logits
    4. If step % stride == 0:
         - Attention Pooling compresses window
         - SSM updates with pooled vector
         - Add SSM output to window
    5. Always rotate input tokens
    
    Efficiency: Only 65 tokens in attention, 6.25% SSM updates
    """
    
    def __init__(self, config):
        """
        Args:
            config: SMTConfig object (wraps YAML config dict)
        """
        super().__init__()
        
        self.config = config
        
        # Extract config values
        model_cfg = config.model_config
        window_cfg = config.window_config
        transformer_cfg = config.transformer_config
        ssm_cfg = config.ssm_config
        pooling_cfg = config.pooling_config
        
        self.n = window_cfg['n_memory_tokens']
        self.m = window_cfg['n_input_tokens']
        self.stride = window_cfg['stride']
        self.d_model = model_cfg['d_model']
        vocab_size = model_cfg['vocab_size']
        pad_token_id = model_cfg['pad_token_id']
        
        # Embedding (shared)
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.pad_token_id = pad_token_id
        
        # Transformer (GPT-2 based)
        self.transformer = WindowedTransformer(
            d_model=self.d_model,
            n_layers=transformer_cfg['num_layers'],
            n_heads=transformer_cfg['num_attention_heads'],
            d_ff=transformer_cfg['intermediate_size'],
            dropout=transformer_cfg['hidden_dropout_prob'],
            pretrained_name=transformer_cfg.get('pretrained_model')
        )
        
        # Attention Pooling
        self.attention_pooling = AttentionPooling(
            d_model=self.d_model,
            d_k=pooling_cfg['query_dim'],
            dropout=model_cfg['dropout']
        )
        
        # SSM (Mamba based)
        self.ssm = SSMMemory(
            d_model=self.d_model,
            n_layers=ssm_cfg['num_layers'],
            d_state=ssm_cfg['state_size'],
            d_conv=ssm_cfg['conv_kernel'],
            expand_factor=ssm_cfg['expand_factor'],
            pretrained_name=ssm_cfg.get('pretrained_model')
        )
        
        # LM Head
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        
        # Tie weights with embedding
        self.lm_head.weight = self.embedding.weight
        
        # Gradient checkpointing (optional)
        use_checkpointing = model_cfg.get('use_gradient_checkpointing', False)
        if use_checkpointing:
            if hasattr(self.transformer.transformer, 'gradient_checkpointing_enable'):
                self.transformer.transformer.gradient_checkpointing_enable()
                print("✅ Enabled gradient checkpointing for Transformer")
            else:
                print("⚠️  Gradient checkpointing not available for this transformer")

        print(f"✅ Created Stride Memory Transformer")
        config.summary()
    
    def forward(self, input_ids, return_all_logits=False):
        """
        Forward pass (training mode - processes entire sequence)
        
        Args:
            input_ids: (batch, seq_len) - Input token IDs
            return_all_logits: If True, return logits for all positions
            
        Returns:
            logits: (batch, seq_len, vocab_size) - Predictions
            aux_outputs: Dict with auxiliary information
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # Embed all tokens
        embeddings = self.embedding(input_ids)  # (B, S, D)
        
        # Initialize window manager (batched)
        window_mgr = BatchedWindowManager(
            batch_size=B,
            n_ssm_outputs=self.n,
            m_input_tokens=self.m,
            d_model=self.d_model,
            device=device
        )
        
        # Store outputs
        all_logits = []
        write_steps = []
        attention_weights_history = []
        
        # Process each position
        for t in range(S):
            # Get current token embedding
            x_t = embeddings[:, t, :]  # (B, D)
            
            # Add to window
            window_mgr.append_input(x_t)
            
            # Get current window
            window = window_mgr.get_window()  # (B, n+m, D)
            
            # Transformer processes window
            transformer_out = self.transformer(window)  # (B, n+m, D)
            
            # Get logits from last position (current token)
            logits_t = self.lm_head(transformer_out[:, -1, :])  # (B, vocab)
            all_logits.append(logits_t)
            
            # Stride-based write
            if t % self.stride == 0 and t > 0:
                # Attention pooling
                pooled, attn_weights = self.attention_pooling(window)  # (B, D)
                
                # SSM update
                ssm_output = self.ssm(pooled)  # (B, D)
                
                # Add SSM output to window
                window_mgr.append_ssm(ssm_output)
                
                # Track statistics
                write_steps.append(t)
                attention_weights_history.append(attn_weights.detach())
        
        # Stack logits
        logits = torch.stack(all_logits, dim=1)  # (B, S, vocab)
        
        # Auxiliary outputs
        aux_outputs = {
            'write_steps': write_steps,
            'n_writes': len(write_steps),
            'write_frequency': len(write_steps) / S if S > 0 else 0.0,
        }
        
        if len(attention_weights_history) > 0:
            aux_outputs['attention_weights'] = torch.stack(attention_weights_history, dim=0)
        
        return logits, aux_outputs
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, 
                 top_k=None, top_p=None):
        """
        Auto-regressive generation
        
        Args:
            input_ids: (batch, prefix_len) - Prefix tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            generated_ids: (batch, prefix_len + max_new_tokens)
        """
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize with prefix
        generated = input_ids.clone()
        
        # Initialize window manager
        window_mgr = BatchedWindowManager(
            batch_size=B,
            n_ssm_outputs=self.n,
            m_input_tokens=self.m,
            d_model=self.d_model,
            device=device
        )
        
        # Process prefix
        with torch.no_grad():
            embeddings = self.embedding(input_ids)
            for t in range(input_ids.shape[1]):
                x_t = embeddings[:, t, :]
                window_mgr.append_input(x_t)
                
                if t % self.stride == 0 and t > 0:
                    window = window_mgr.get_window()
                    pooled, _ = self.attention_pooling(window)
                    ssm_output = self.ssm(pooled)
                    window_mgr.append_ssm(ssm_output)
        
        # Generate new tokens
        for _ in range(max_new_tokens):
            # Get window
            window = window_mgr.get_window()
            
            # Transformer forward
            with torch.no_grad():
                transformer_out = self.transformer(window)
                logits = self.lm_head(transformer_out[:, -1, :])  # (B, vocab)
            
            # Sample next token
            if temperature > 0:
                logits = logits / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    for i in range(B):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        logits[i, indices_to_remove] = -float('Inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update window
            next_embedding = self.embedding(next_token).squeeze(1)
            window_mgr.append_input(next_embedding)
            
            # Stride-based write
            current_step = generated.shape[1] - 1
            if current_step % self.stride == 0:
                window = window_mgr.get_window()
                pooled, _ = self.attention_pooling(window)
                ssm_output = self.ssm(pooled)
                window_mgr.append_ssm(ssm_output)
        
        return generated
    
    def count_parameters(self):
        """Count trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Break down by component
        breakdown = {
            'embedding': sum(p.numel() for p in self.embedding.parameters()),
            'transformer': sum(p.numel() for p in self.transformer.parameters()),
            'attention_pooling': sum(p.numel() for p in self.attention_pooling.parameters()),
            'ssm': sum(p.numel() for p in self.ssm.parameters()),
            'lm_head': 0  # Tied with embedding
        }
        
        return {
            'total': total,
            'trainable': trainable,
            'breakdown': breakdown
        }


if __name__ == "__main__":
    # Test model
    print("Testing Stride Memory Transformer...")
    
    from config.model_config import SMTConfig
    
    # Create config
    config = SMTConfig(
        n_ssm_outputs=15,
        m_input_tokens=50,
        stride=16,
        transformer_n_layers=12,
        ssm_n_layers=24,
        device='cpu'
    )
    
    try:
        # Create model
        model = StrideMemoryTransformer(config)
        
        # Test forward
        batch_size = 2
        seq_len = 100
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        print(f"\nForward pass...")
        logits, aux = model(input_ids)
        
        print(f"✅ Input:  {input_ids.shape}")
        print(f"✅ Output: {logits.shape}")
        print(f"✅ Writes: {aux['n_writes']} / {seq_len} steps ({aux['write_frequency']:.2%})")
        
        # Count parameters
        params = model.count_parameters()
        print(f"\n📊 Parameters:")
        print(f"  Total:      {params['total']/1e6:.1f}M")
        print(f"  Trainable:  {params['trainable']/1e6:.1f}M")
        print(f"\n  Breakdown:")
        for name, count in params['breakdown'].items():
            print(f"    {name:20s}: {count/1e6:6.1f}M")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
