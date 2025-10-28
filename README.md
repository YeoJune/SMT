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
pip install torch transformers mamba-ssm causal-conv1d datasets matplotlib
```

## Quick Start

### Running an Experiment

```bash
# Run WikiText-2 experiment
python scripts/run_experiment.py --config configs/wikitext2_experiment.yaml
```

The experiment will:

1. Create a timestamped output directory: `experiments/results/wikitext2_smt/YYYYMMDD_HHMMSS/`
2. Save configuration, metrics, and checkpoints
3. Track training loss and validation perplexity
4. Save best model based on validation loss

### Analyzing Results

```bash
# Visualize single experiment
python scripts/plot_results.py --result_dir experiments/results/wikitext2_smt/20241028_123456

# Compare multiple experiments
python scripts/compare_experiments.py experiments/results/wikitext2_smt
```

### Output Structure

Each experiment creates:

```
experiments/results/wikitext2_smt/YYYYMMDD_HHMMSS/
├── config.yaml              # Saved configuration
├── results.json             # Final results summary
├── metrics_history.json     # Training metrics over time
├── best_model.pt           # Best checkpoint (lowest val loss)
├── checkpoint_stepXXXX.pt  # Regular checkpoints
└── training_curves.png     # Training plots (after plot_results.py)
```

### Results Format

`results.json` contains:

- Experiment info (name, timestamp, training time)
- Model configuration (parameters, architecture)
- Dataset info (samples, sequence length)
- Performance metrics (best/final loss and perplexity)

Example:

```json
{
  "results": {
    "best_val_loss": 4.2134,
    "best_val_ppl": 67.89,
    "final_val_loss": 4.205,
    "final_val_ppl": 67.32
  }
}
```

## Quick Start (Programmatic)

```python
from src.models.smt import StrideMemoryTransformer
from config.model_config import SMTConfig

config = SMTConfig(
    n_ssm_outputs=15,
    m_input_tokens=50,
    stride=16
)

model = StrideMemoryTransformer(config)

# Training
logits, aux = model(input_ids)
print(f"Writes: {aux['n_writes']}")
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
