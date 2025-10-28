"""
Config validation test - Verify YAML config is properly applied
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.model_config import load_config

print("=" * 80)
print("CONFIG VALIDATION TEST")
print("=" * 80)

# Load config
config_path = "configs/wikitext2_experiment.yaml"
print(f"\n📄 Loading: {config_path}\n")

try:
    config = load_config(config_path)
    
    print("✅ Config loaded successfully!\n")
    
    # Print summary
    config.summary()
    
    # Verify key values
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    checks = [
        ("Window size", config.window_size, 65),
        ("Memory tokens", config.window_config['n_memory_tokens'], 15),
        ("Input tokens", config.window_config['n_input_tokens'], 50),
        ("Stride", config.window_config['stride'], 16),
        ("d_model", config.model_config['d_model'], 768),
        ("Transformer layers", config.transformer_config['num_layers'], 12),
        ("Transformer heads", config.transformer_config['num_attention_heads'], 12),
        ("FFN dimension", config.transformer_config['intermediate_size'], 3072),
        ("SSM layers", config.ssm_config['num_layers'], 12),
        ("SSM state size", config.ssm_config['state_size'], 16),
    ]
    
    all_passed = True
    for name, actual, expected in checks:
        status = "✅" if actual == expected else "❌"
        print(f"{status} {name:20s}: {actual:6} (expected: {expected})")
        if actual != expected:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Config is properly applied!")
    else:
        print("⚠️  SOME CHECKS FAILED - Please review the config")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Error loading config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

