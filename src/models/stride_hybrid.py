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


class StrideHybridModel(nn.Module):
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
            config: StrideHybridConfig object
        """
        super().__init__()
        
        self.config = config
        self.n = config.n_ssm_outputs
        self.m = config.m_input_tokens
        self.stride = config.stride
        self.d_model = config.d_model
        
        # Embedding (shared)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer (GPT-2 based)
        self.transformer = WindowedTransformer(
            d_model=config.d_model,
            n_layers=config.transformer_n_layers,
            n_heads=config.transformer_n_heads,
            dropout=config.dropout,
            pretrained_name=config.transformer_model
        )
        
        # Attention Pooling
        self.attention_pooling = AttentionPooling(
            d_model=config.d_model,
            d_k=config.d_k,
            dropout=config.dropout
        )
        
        # SSM (Mamba based)
        self.ssm = SSMMemory(
            d_model=config.d_model,
            n_layers=config.ssm_n_layers,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand_factor=config.ssm_expand_factor,
            pretrained_name=config.ssm_model
        )
        
        # LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights with embedding
        self.lm_head.weight = self.embedding.weight
        
        print(f"✅ Created Stride-based Hybrid Model")
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
    print("Testing StrideHybridModel...")
    
    from config.model_config import StrideHybridConfig
    
    # Create config
    config = StrideHybridConfig(
        n_ssm_outputs=15,
        m_input_tokens=50,
        stride=16,
        transformer_n_layers=12,
        ssm_n_layers=24,
        device='cpu'
    )
    
    try:
        # Create model
        model = StrideHybridModel(config)
        
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
