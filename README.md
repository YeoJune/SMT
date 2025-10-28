# SMT (Stride Memory Transformer)

Implementation of a novel hybrid architecture combining:

- **Fixed local Transformer window** (65 tokens) for powerful short-term processing
- **State Space Model (Mamba)** for compressed long-term memory
- **Stride-based update** with Attention Pooling for efficiency

## Architecture Overview

```
Input Token → Embedding
    ↓
Window = [n SSM outputs | m recent tokens]
    ↓
Transformer Decoder (GPT-2, 12 layers)
    ↓
Output Logits
    ↓
if step % stride == 0:
    Attention Pooling → SSM update
```

## Key Features

- **Fixed Window**: Only 65 tokens in Transformer attention (O(65²) instead of O(L²))
- **Stride-based Write**: Updates SSM every 16 steps (6.25% frequency)
- **Attention Pooling**: Learned importance weighting for memory compression
- **Pre-trained Models**: Uses GPT-2 (117M) + Mamba (130M) as base

## Installation

```bash
pip install torch transformers mamba-ssm causal-conv1d
```

## Quick Start

```python
from src.models.stride_hybrid import StrideHybridModel

model = StrideHybridModel(
    n_ssm_outputs=15,
    m_input_tokens=50,
    stride=16
)

# Training
logits = model(input_ids)
```

## Project Structure

```
SMT/
├── config/              # Configuration files
├── src/
│   ├── models/         # Model implementations
│   │   ├── components/ # Transformer, SSM, Pooling
│   │   ├── stride_hybrid.py
│   │   └── window_manager.py
│   ├── data/           # Dataset loaders
│   ├── training/       # Training loops
│   └── utils/          # Utilities
├── experiments/        # Experiment scripts
└── tests/             # Unit tests
```

## References

- Transformer-XL: https://arxiv.org/abs/1901.02860
- Mamba: https://arxiv.org/abs/2312.00752
- GPT-2: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
