"""
Attention Pooling Layer
Compresses window into single vector using learned Query-Key-Value attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to compress window into single vector
    
    Process:
    1. Query = W_q @ mean(window)     - Single query from average
    2. Keys = W_k @ window            - Key for each position
    3. Attention = softmax(Q^T K / √d)
    4. Output = Attention @ window    - Weighted sum
    
    This learns what to remember from the current window.
    """
    
    def __init__(self, d_model, d_k=None, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_k: Key dimension (default: same as d_model)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.scale = self.d_k ** -0.5
        
        # Learnable projections
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, window):
        """
        Args:
            window: (batch, window_size, d_model) - Current window
            
        Returns:
            pooled: (batch, d_model) - Single compressed vector
            attention_weights: (batch, window_size) - Attention distribution
        """
        B, W, D = window.shape
        
        # Query: average of window projected
        window_mean = window.mean(dim=1)  # (B, D)
        q = self.W_q(window_mean)  # (B, d_k)
        
        # Keys: each position projected
        K = self.W_k(window)  # (B, W, d_k)
        
        # Attention scores
        # q: (B, d_k) -> (B, 1, d_k)
        # K: (B, W, d_k) -> (B, d_k, W)
        attn_scores = torch.bmm(
            q.unsqueeze(1),  # (B, 1, d_k)
            K.transpose(1, 2)  # (B, d_k, W)
        ) * self.scale  # (B, 1, W)
        
        attn_scores = attn_scores.squeeze(1)  # (B, W)
        
        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, W)
        
        # Apply dropout to attention weights during training
        attn_weights_dropped = self.dropout(attn_weights)
        
        # Weighted sum of original window (not keys!)
        # window: (B, W, D)
        # attn_weights_dropped: (B, W) -> (B, W, 1)
        pooled = torch.bmm(
            attn_weights_dropped.unsqueeze(1),  # (B, 1, W)
            window  # (B, W, D)
        ).squeeze(1)  # (B, D)
        
        # Return original attn_weights (before dropout) for analysis
        return pooled, attn_weights
    
    def extra_repr(self):
        return f'd_model={self.d_model}, d_k={self.d_k}'


class AttentionPoolingAnalyzer:
    """Utility to analyze attention patterns"""
    
    @staticmethod
    def analyze_attention(attn_weights, window_size, n_ssm=15):
        """
        Analyze where attention focuses
        
        Args:
            attn_weights: (batch, window_size) or (window_size,)
            window_size: Total window size
            n_ssm: Number of SSM outputs in window
            
        Returns:
            dict with statistics
        """
        if attn_weights.dim() == 2:
            # Average over batch
            attn_weights = attn_weights.mean(dim=0)
        
        attn_weights = attn_weights.cpu().numpy()
        
        # Split into SSM outputs and input tokens
        ssm_attn = attn_weights[:n_ssm]
        input_attn = attn_weights[n_ssm:]
        
        stats = {
            'ssm_total': float(ssm_attn.sum()),
            'input_total': float(input_attn.sum()),
            'ssm_max': float(ssm_attn.max()),
            'input_max': float(input_attn.max()),
            'entropy': float(-((attn_weights + 1e-10) * 
                             np.log(attn_weights + 1e-10)).sum()),
        }
        
        return stats
    
    @staticmethod
    def visualize_attention(attn_weights, save_path=None):
        """
        Visualize attention distribution
        
        Args:
            attn_weights: (batch, window_size) or (window_size,)
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("⚠️  matplotlib not installed, skipping visualization")
            return
        
        if attn_weights.dim() == 2:
            attn_weights = attn_weights.mean(dim=0)
        
        attn_weights = attn_weights.cpu().numpy()
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(attn_weights)), attn_weights)
        plt.xlabel('Window Position')
        plt.ylabel('Attention Weight')
        plt.title('Attention Pooling Distribution')
        plt.axvline(x=15-0.5, color='r', linestyle='--', 
                   label='SSM | Input boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved attention visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    # Test attention pooling
    print("Testing AttentionPooling...")
    
    batch_size = 2
    window_size = 65
    d_model = 768
    
    pooling = AttentionPooling(d_model=d_model)
    
    # Random window
    window = torch.randn(batch_size, window_size, d_model)
    
    # Forward
    pooled, attn_weights = pooling(window)
    
    print(f"✅ Input:  {window.shape}")
    print(f"✅ Output: {pooled.shape}")
    print(f"✅ Attention: {attn_weights.shape}")
    print(f"✅ Attention sum: {attn_weights.sum(dim=1)}")
    
    # Analyze
    import numpy as np
    stats = AttentionPoolingAnalyzer.analyze_attention(
        attn_weights[0], window_size, n_ssm=15
    )
    print(f"\n📊 Attention Statistics:")
    for k, v in stats.items():
        print(f"  - {k}: {v:.4f}")
