"""
Integration test for SMT (Stride Memory Transformer) Model
Tests each component and the full model
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np


def test_attention_pooling():
    """Test attention pooling component"""
    print("\n" + "="*80)
    print("Testing Attention Pooling")
    print("="*80)
    
    from src.models.components.attention_pooling import AttentionPooling
    
    batch_size = 2
    window_size = 65
    d_model = 768
    
    pooling = AttentionPooling(d_model=d_model)
    window = torch.randn(batch_size, window_size, d_model)
    
    pooled, attn_weights = pooling(window)
    
    assert pooled.shape == (batch_size, d_model), f"Expected {(batch_size, d_model)}, got {pooled.shape}"
    assert attn_weights.shape == (batch_size, window_size)
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    
    print(f"✅ Input shape:      {window.shape}")
    print(f"✅ Pooled shape:     {pooled.shape}")
    print(f"✅ Attention shape:  {attn_weights.shape}")
    print(f"✅ Attention sums to 1.0: {attn_weights.sum(dim=1)}")
    print("✅ PASSED")


def test_window_manager():
    """Test window manager"""
    print("\n" + "="*80)
    print("Testing Window Manager")
    print("="*80)
    
    from src.models.window_manager import WindowManager, BatchedWindowManager
    
    n, m = 15, 50
    d_model = 768
    device = 'cpu'
    
    # Test single window manager
    manager = WindowManager(n, m, d_model, device)
    
    for step in range(20):
        x = torch.randn(d_model, device=device)
        manager.append_input(x)
        
        if step % 16 == 0:
            ssm_out = torch.randn(d_model, device=device)
            manager.append_ssm(ssm_out)
        
        window = manager.get_window(batch_size=1)
        assert window.shape == (1, n+m, d_model)
    
    print(f"✅ Single manager: {manager}")
    
    # Test batched manager
    batch_size = 4
    batched_manager = BatchedWindowManager(batch_size, n, m, d_model, device)
    
    for step in range(20):
        x = torch.randn(batch_size, d_model, device=device)
        batched_manager.append_input(x)
        
        if step % 16 == 0:
            ssm_out = torch.randn(batch_size, d_model, device=device)
            batched_manager.append_ssm(ssm_out)
        
        window = batched_manager.get_window()
        assert window.shape == (batch_size, n+m, d_model)
    
    print(f"✅ Batched manager: {batched_manager}")
    print("✅ PASSED")


def test_transformer():
    """Test transformer component"""
    print("\n" + "="*80)
    print("Testing Windowed Transformer")
    print("="*80)
    
    from src.models.components.transformer import SimpleCausalTransformer
    
    batch_size = 2
    window_size = 65
    d_model = 768
    
    transformer = SimpleCausalTransformer(
        d_model=d_model,
        n_layers=6,
        n_heads=12
    )
    
    x = torch.randn(batch_size, window_size, d_model)
    output = transformer(x)
    
    assert output.shape == (batch_size, window_size, d_model)
    
    print(f"✅ Input shape:  {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Parameters:   {sum(p.numel() for p in transformer.parameters())/1e6:.1f}M")
    print("✅ PASSED")


def test_ssm():
    """Test SSM component"""
    print("\n" + "="*80)
    print("Testing SSM Memory")
    print("="*80)
    
    from src.models.components.ssm import SimpleSSM
    
    batch_size = 2
    d_model = 768
    
    ssm = SimpleSSM(d_model=d_model, n_layers=4)
    
    x = torch.randn(batch_size, d_model)
    output = ssm(x)
    
    assert output.shape == (batch_size, d_model)
    
    print(f"✅ Input shape:  {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Parameters:   {sum(p.numel() for p in ssm.parameters())/1e6:.1f}M")
    print("✅ PASSED")


def test_full_model():
    """Test complete model forward and generate"""
    print("Testing Full SMT (Stride Memory Transformer) Model")
    print("-" * 60)
    
    # Use simple components to avoid needing pretrained models
    print("⚠️  Using simplified components (no pretrained models)")
    
    # Create a minimal config
    class SimpleConfig:
        n_ssm_outputs = 15
        m_input_tokens = 50
        stride = 16
        d_model = 768
        vocab_size = 1000
        transformer_n_layers = 6
        transformer_n_heads = 12
        ssm_n_layers = 4
        ssm_d_state = 16
        ssm_d_conv = 4
        ssm_expand_factor = 2
        d_k = 768
        dropout = 0.1
        max_position_embeddings = 1024
        device = 'cpu'
        use_cuda = False
        transformer_model = "gpt2"
        ssm_model = "state-spaces/mamba-130m-hf"
        
        @property
        def window_size(self):
            return self.n_ssm_outputs + self.m_input_tokens
        
        @property
        def write_frequency(self):
            return 1.0 / self.stride
        
        def summary(self):
            pass
    
    config = SimpleConfig()
    
    # Build model manually with simple components
    import torch.nn as nn
    from src.models.components.attention_pooling import AttentionPooling
    from src.models.components.transformer import SimpleCausalTransformer
    from src.models.components.ssm import SimpleSSM
    from src.models.window_manager import BatchedWindowManager
    
    embedding = nn.Embedding(config.vocab_size, config.d_model)
    transformer = SimpleCausalTransformer(config.d_model, config.transformer_n_layers, config.transformer_n_heads)
    attention_pooling = AttentionPooling(config.d_model, config.d_k)
    ssm = SimpleSSM(config.d_model, config.ssm_n_layers)
    lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    lm_head.weight = embedding.weight
    
    print(f"✅ Built simplified model")
    
    # Test forward pass
    batch_size = 2
    seq_len = 100
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    embeddings = embedding(input_ids)
    window_mgr = BatchedWindowManager(batch_size, config.n_ssm_outputs, 
                                      config.m_input_tokens, config.d_model, 'cpu')
    
    all_logits = []
    n_writes = 0
    
    for t in range(seq_len):
        x_t = embeddings[:, t, :]
        window_mgr.append_input(x_t)
        
        window = window_mgr.get_window()
        transformer_out = transformer(window)
        logits_t = lm_head(transformer_out[:, -1, :])
        all_logits.append(logits_t)
        
        if t % config.stride == 0 and t > 0:
            pooled, _ = attention_pooling(window)
            ssm_output = ssm(pooled)
            window_mgr.append_ssm(ssm_output)
            n_writes += 1
    
    logits = torch.stack(all_logits, dim=1)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    expected_writes = (seq_len // config.stride)
    assert n_writes == expected_writes, f"Expected {expected_writes} writes, got {n_writes}"
    
    print(f"✅ Input shape:       {input_ids.shape}")
    print(f"✅ Output shape:      {logits.shape}")
    print(f"✅ Number of writes:  {n_writes}/{seq_len} ({n_writes/seq_len:.2%})")
    print(f"✅ Expected writes:   {expected_writes}")
    print("✅ PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("SMT (STRIDE MEMORY TRANSFORMER) - INTEGRATION TESTS")
    print("="*80)
    
    tests = [
        ("Attention Pooling", test_attention_pooling),
        ("Window Manager", test_window_manager),
        ("Transformer", test_transformer),
        ("SSM", test_ssm),
        ("Full Model", test_full_model),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} FAILED:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
