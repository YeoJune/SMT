"""
Model configuration for SMT (Stride Memory Transformer)
Simplified to work directly with YAML configs
"""

from typing import Dict, Any, Optional


class SMTConfig:
    """
    Configuration wrapper for SMT model
    Accepts standard YAML config dict and provides validation
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize from YAML config dictionary
        
        Args:
            config_dict: Full config dictionary from YAML
        """
        self.config = config_dict
        self.model_config = config_dict['model']
        self.window_config = self.model_config['window']
        self.transformer_config = self.model_config['transformer']
        self.ssm_config = self.model_config['ssm']
        self.pooling_config = self.model_config['attention_pooling']
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration"""
        stride = self.window_config['stride']
        n_input = self.window_config['n_input_tokens']
        d_model = self.model_config['d_model']
        n_heads = self.transformer_config['num_attention_heads']
        
        # Check stride
        if stride > n_input:
            raise ValueError(
                f"stride ({stride}) must be <= n_input_tokens ({n_input})"
            )
        
        # Check attention heads
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_attention_heads ({n_heads})"
            )
        
        # Warn about stride recommendation
        recommended_stride = n_input // 3
        if stride != recommended_stride:
            print(f"⚠️  Warning: stride={stride}, recommended={recommended_stride} "
                  f"(n_input/3 for 3x coverage)")
    
    @property
    def window_size(self) -> int:
        """Total window size (memory + input tokens)"""
        return (self.window_config['n_memory_tokens'] + 
                self.window_config['n_input_tokens'])
    
    @property
    def write_frequency(self) -> float:
        """Write frequency (1/stride)"""
        return 1.0 / self.window_config['stride']
    
    def get(self, *keys, default=None):
        """Get nested config value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def summary(self):
        """Print configuration summary"""
        print("=" * 80)
        print("SMT (Stride Memory Transformer) Configuration")
        print("=" * 80)
        
        print(f"\n📊 Window Configuration:")
        print(f"  - Memory tokens (n):    {self.window_config['n_memory_tokens']}")
        print(f"  - Input tokens (m):     {self.window_config['n_input_tokens']}")
        print(f"  - Total window size:    {self.window_size}")
        print(f"  - Stride:               {self.window_config['stride']}")
        print(f"  - Write frequency:      {self.write_frequency:.2%}")
        print(f"  - Coverage:             {self.window_config['n_input_tokens'] / self.window_config['stride']:.1f}x")
        
        print(f"\n🔧 Model Architecture:")
        print(f"  - Model dimension:      {self.model_config['d_model']}")
        print(f"  - Vocab size:           {self.model_config['vocab_size']}")
        print(f"  - Dropout:              {self.model_config['dropout']}")
        
        print(f"\n🤖 Transformer:")
        print(f"  - Layers:               {self.transformer_config['num_layers']}")
        print(f"  - Attention heads:      {self.transformer_config['num_attention_heads']}")
        print(f"  - FFN dimension:        {self.transformer_config['intermediate_size']}")
        print(f"  - Pretrained:           {self.transformer_config.get('pretrained_model', 'None')}")
        
        print(f"\n🧠 SSM (Mamba):")
        print(f"  - Layers:               {self.ssm_config['num_layers']}")
        print(f"  - State size:           {self.ssm_config['state_size']}")
        print(f"  - Conv kernel:          {self.ssm_config['conv_kernel']}")
        print(f"  - Expand factor:        {self.ssm_config['expand_factor']}")
        print(f"  - Pretrained:           {self.ssm_config.get('pretrained_model', 'None')}")
        
        print(f"\n⚡ Efficiency:")
        window_flops = self.window_size ** 2 * self.model_config['d_model']
        full_flops = 4096 ** 2 * self.model_config['d_model']
        print(f"  - Window attention:     {window_flops / 1e6:.1f}M FLOPs")
        print(f"  - vs Full (L=4096):     {full_flops / window_flops:.0f}x more")
        print("=" * 80 + "\n")


def load_config(config_path: str) -> SMTConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        SMTConfig object
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return SMTConfig(config_dict)
