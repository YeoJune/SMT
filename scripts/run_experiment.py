r"""
SMT Experiment Runner - WikiText-2 Language Modeling

Usage (PowerShell):
python .\scripts\run_experiment.py --config configs\wikitext2_experiment.yaml
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import yaml
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.model_config import SMTConfig
from src.models.smt import StrideMemoryTransformer

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("⚠️  Missing dependencies. Install with:")
    print("   pip install datasets transformers")
    sys.exit(1)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    """Load YAML configuration"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_wikitext2_data(tokenizer, max_length=512, split="train"):
    """Load and tokenize WikiText-2 dataset"""
    print(f"\n📚 Loading WikiText-2 ({split})...")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # Filter empty lines
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    print(f"✅ Loaded {len(tokenized)} examples")
    return tokenized


def create_model(cfg):
    """Create SMT model from config"""
    print("\n🔧 Creating SMT model...")
    
    model_cfg = cfg["model"]
    
    # Create config
    config = SMTConfig(
        n_ssm_outputs=model_cfg["n_ssm"],
        m_input_tokens=model_cfg["m_input"],
        stride=model_cfg["stride"],
        d_model=model_cfg["d_model"],
        vocab_size=model_cfg["vocab_size"],
        transformer_n_layers=model_cfg["n_layers"],
        transformer_n_heads=model_cfg["n_heads"],
        transformer_model="gpt2",  # Use GPT-2 base
        ssm_n_layers=model_cfg["n_layers"],
        ssm_d_state=model_cfg["d_state"],
        ssm_d_conv=model_cfg["d_conv"],
        ssm_expand_factor=model_cfg["expand"],
        dropout=model_cfg["dropout"],
        device="cpu"  # Will move to device later
    )
    
    # Create model
    model = StrideMemoryTransformer(config)
    
    # Count parameters
    params = model.count_parameters()
    print(f"\n📊 Model size: {params['total']/1e6:.1f}M parameters")
    
    return model


def train_step(model, batch, device, optimizer, grad_clip):
    """Single training step"""
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Forward pass
    logits, aux = model(input_ids)
    
    # Shift for language modeling
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    
    # Calculate loss (only on non-padded tokens)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none"
    )
    loss = (loss * shift_mask.view(-1)).sum() / shift_mask.sum()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    
    return loss.item(), aux


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=None):
    """Evaluate on validation set"""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Shift for language modeling
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Calculate loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none"
        )
        
        batch_tokens = shift_mask.sum().item()
        total_loss += (loss * shift_mask.view(-1)).sum().item()
        total_tokens += batch_tokens
        
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    print("=" * 80)
    print("SMT (STRIDE MEMORY TRANSFORMER) - EXPERIMENT RUNNER")
    print("=" * 80)
    
    # Load config
    cfg = load_config(args.config)
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    
    # Set seed
    set_seed(exp_cfg["seed"])
    print(f"\n🌱 Seed: {exp_cfg['seed']}")
    
    # Setup device with compatibility check
    if torch.cuda.is_available():
        # Check CUDA compute capability
        try:
            device = torch.device("cuda")
            # Try a simple operation to verify CUDA works
            _ = torch.zeros(1).to(device)
            print(f"🖥️  Device: cuda (GPU: {torch.cuda.get_device_name(0)})")
        except Exception as e:
            # Catch all exceptions (RuntimeError, CudaError, etc.)
            print(f"⚠️  CUDA available but incompatible: {e}")
            print(f"⚠️  Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"🖥️  Device: cpu")
    
    # Create output directory
    output_dir = Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_dir}")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Vocab size: {len(tokenizer)}")
    
    # Prepare data
    train_dataset = prepare_wikitext2_data(
        tokenizer, 
        max_length=data_cfg["max_seq_length"],
        split="train"
    )
    val_dataset = prepare_wikitext2_data(
        tokenizer,
        max_length=data_cfg["max_seq_length"],
        split="validation"
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Create model
    model = create_model(cfg)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    global_step = 0
    best_val_loss = float('inf')
    
    train_iterator = iter(train_loader)
    
    with tqdm(total=train_cfg["max_steps"], desc="Training") as pbar:
        while global_step < train_cfg["max_steps"]:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            # Training step
            loss, aux = train_step(
                model, batch, device, optimizer,
                train_cfg["max_grad_norm"]
            )
            
            global_step += 1
            pbar.update(1)
            
            # Log
            if global_step % cfg["logging"]["log_every"] == 0:
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'writes': f"{aux['n_writes']}/{aux.get('seq_len', 0)}"
                })
            
            # Evaluate
            if global_step % train_cfg["eval_every"] == 0:
                print(f"\n📊 Evaluation at step {global_step}...")
                val_loss, val_ppl = evaluate(
                    model, val_loader, device,
                    max_batches=eval_cfg.get("max_eval_batches")
                )
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Val PPL:  {val_ppl:.2f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = output_dir / "best_model.pt"
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_ppl': val_ppl,
                        'config': cfg
                    }, checkpoint_path)
                    print(f"   💾 Saved best model (PPL: {val_ppl:.2f})")
            
            # Save checkpoint
            if global_step % train_cfg["save_every"] == 0:
                checkpoint_path = output_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg
                }, checkpoint_path)
                print(f"\n💾 Saved checkpoint at step {global_step}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
