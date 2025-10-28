"""
Model configuration for SMT (Stride Memory Transformer)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SMTConfig:
    """Configuration for SMT (Stride Memory Transformer) model"""
    
    # Window configuration
    n_ssm_outputs: int = 15  # Number of SSM outputs in window
    m_input_tokens: int = 50  # Number of recent input tokens in window
    
    @property
    def window_size(self) -> int:
        return self.n_ssm_outputs + self.m_input_tokens
    
    # Write configuration
    stride: int = 16  # Update SSM every N steps (m // 3)
    
    @property
    def write_frequency(self) -> float:
        return 1.0 / self.stride
    
    # Model dimensions
    d_model: int = 768
    vocab_size: int = 50280  # GPT-2 tokenizer
    pad_token_id: int = 50256  # GPT-2 EOS token used as pad
    
    # Transformer (GPT-2 based)
    transformer_n_layers: int = 12
    transformer_n_heads: int = 12
    transformer_model: str = "gpt2"  # Pretrained model name
    
    # SSM (Mamba based)
    ssm_n_layers: int = 24
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand_factor: int = 2
    ssm_model: str = "state-spaces/mamba-130m-hf"
    
    # Attention Pooling
    d_k: int = 768  # Attention key dimension
    
    # Training
    dropout: float = 0.1
    max_position_embeddings: int = 1024
    
    # Device
    device: str = "cuda"
    use_cuda: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.stride <= self.m_input_tokens, \
            f"Stride ({self.stride}) should be <= m_input_tokens ({self.m_input_tokens})"
        assert self.d_model % self.transformer_n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.transformer_n_heads})"
        
        # Recommend: stride = m // 3 for 3x coverage
        recommended_stride = self.m_input_tokens // 3
        if self.stride != recommended_stride:
            print(f"⚠️  Warning: stride={self.stride}, recommended={recommended_stride} "
                  f"(m/3 for 3x coverage)")
    
    def summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("SMT (Stride Memory Transformer) Configuration")
        print("=" * 60)
        print(f"\n📊 Window Configuration:")
        print(f"  - SSM outputs (n):      {self.n_ssm_outputs}")
        print(f"  - Input tokens (m):     {self.m_input_tokens}")
        print(f"  - Total window size:    {self.window_size}")
        
        print(f"\n⏱️  Write Configuration:")
        print(f"  - Stride:               {self.stride}")
        print(f"  - Write frequency:      {self.write_frequency:.2%}")
        print(f"  - Coverage:             {self.m_input_tokens / self.stride:.1f}x")
        
        print(f"\n🔧 Model Architecture:")
        print(f"  - Transformer layers:   {self.transformer_n_layers}")
        print(f"  - Transformer heads:    {self.transformer_n_heads}")
        print(f"  - SSM layers:           {self.ssm_n_layers}")
        print(f"  - Model dimension:      {self.d_model}")
        
        print(f"\n💾 Pre-trained Models:")
        print(f"  - Transformer:          {self.transformer_model}")
        print(f"  - SSM:                  {self.ssm_model}")
        
        print(f"\n⚡ Efficiency:")
        window_flops = self.window_size ** 2 * self.d_model
        print(f"  - Window attention:     {window_flops / 1e6:.1f}M FLOPs")
        print(f"  - vs Full (L=4096):     {(4096**2 * self.d_model) / window_flops:.0f}x more")
        print("=" * 80 + "\n")


@dataclass  
class TrainingConfig:
    """Training configuration"""
    
    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    max_length: int = 1024
    
    # Optimization
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    learning_rate_pooling: float = 1e-4  # Higher LR for new attention pooling
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Schedule
    num_epochs: int = 20
    warmup_steps: int = 1000
    eval_every: int = 500
    save_every: int = 1000
    
    # Logging
    log_dir: str = "outputs/logs"
    checkpoint_dir: str = "outputs/checkpoints"
    use_wandb: bool = False
    wandb_project: str = "smt"
    
    # Compute
    num_workers: int = 4
    pin_memory: bool = True
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def summary(self):
        """Print training configuration"""
        print("=" * 80)
        print("Training Configuration")
        print("=" * 80)
        print(f"\n📚 Dataset:")
        print(f"  - Name:                 {self.dataset_name}")
        print(f"  - Config:               {self.dataset_config}")
        print(f"  - Max length:           {self.max_length}")
        
        print(f"\n🎯 Optimization:")
        print(f"  - Batch size:           {self.batch_size}")
        print(f"  - Gradient accum:       {self.gradient_accumulation_steps}")
        print(f"  - Effective batch:      {self.effective_batch_size}")
        print(f"  - Learning rate:        {self.learning_rate:.2e}")
        print(f"  - LR (pooling):         {self.learning_rate_pooling:.2e}")
        
        print(f"\n📈 Schedule:")
        print(f"  - Epochs:               {self.num_epochs}")
        print(f"  - Warmup steps:         {self.warmup_steps}")
        print(f"  - Eval every:           {self.eval_every} steps")
        print("=" * 80 + "\n")
