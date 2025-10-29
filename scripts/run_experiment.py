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
import json
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.model_config import SMTConfig, load_config
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


def load_config_file(path: str):
    """Load YAML configuration"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results(output_dir, results):
    """Save experiment results to JSON"""
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved results to {results_path}")


def save_metrics_history(output_dir, metrics_history):
    """Save training metrics history"""
    metrics_path = output_dir / "metrics_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"📊 Saved metrics history to {metrics_path}")


def prepare_wikitext2_data(config, tokenizer, split="train"):
    """Load and tokenize WikiText-2 dataset"""
    print(f"\n📚 Loading WikiText-2 ({split})...")
    
    data_config = config['data']
    
    # Load dataset
    dataset = load_dataset(
        data_config['dataset_name'], 
        data_config['dataset_config'], 
        split=split
    )
    
    # Filter empty lines
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_config['max_seq_length'],
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


def create_model(config: Dict[str, Any]):
    """Create SMT model from config"""
    print("\n🔧 Creating SMT model...")
    
    # Create config wrapper
    smt_config = SMTConfig(config)
    
    # Create model
    model = StrideMemoryTransformer(smt_config)
    
    # Count parameters
    params = model.count_parameters()
    print(f"\n📊 Model size: {params['total']/1e6:.1f}M parameters")
    
    return model


def train_step(model, batch, device, optimizer, grad_clip, pad_token_id, chunk_size=None, scaler=None):
    """
    Single training step with proper TBPTT support.
    
    If chunk_size is provided, processes sequence in chunks with
    immediate backward pass for each chunk (true TBPTT).
    """
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    B, S = input_ids.shape
    
    # Without chunking: standard training
    if chunk_size is None or S <= chunk_size:
        return train_step_standard(model, input_ids, device, optimizer, 
                                   grad_clip, pad_token_id, scaler)
    
    # With chunking: TBPTT training
    return train_step_tbptt(model, input_ids, device, optimizer,
                           grad_clip, pad_token_id, chunk_size, scaler)


def train_step_standard(model, input_ids, device, optimizer, grad_clip, pad_token_id, scaler):
    """Standard training without TBPTT."""
    # Forward pass
    if scaler is not None:
        with torch.amp.autocast('cuda'):
            logits, aux = model(input_ids, chunk_size=None, return_aux=True)
            loss = compute_loss(logits, input_ids, pad_token_id)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, aux = model(input_ids, chunk_size=None, return_aux=True)
        loss = compute_loss(logits, input_ids, pad_token_id)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
    
    return loss.item(), aux


def train_step_tbptt(model, input_ids, device, optimizer, grad_clip, pad_token_id, chunk_size, scaler):
    """
    Training with TBPTT: process chunks sequentially with immediate backward.
    
    This is the memory-efficient version that processes long sequences.
    """
    B, S = input_ids.shape
    
    # Initialize model state
    model.ssm.init_cache(batch_size=B, device=device)
    from src.models.window_manager import WindowManager
    window_mgr = WindowManager(
        batch_size=B,
        n_ssm_outputs=model.n,
        m_input_tokens=model.m,
        d_model=model.d_model,
        device=device,
    )
    
    total_loss = 0.0
    total_tokens = 0
    n_chunks = 0
    all_write_steps = []
    
    # Process each chunk
    for chunk_start in range(0, S, chunk_size):
        chunk_end = min(chunk_start + chunk_size, S)
        chunk_input_ids = input_ids[:, chunk_start:chunk_end]
        
        # Embed chunk
        chunk_embeddings = model.embedding(chunk_input_ids)
        
        # Process chunk
        chunk_logits, write_steps = model._process_chunk(
            chunk_embeddings, window_mgr, global_offset=chunk_start
        )
        all_write_steps.extend(write_steps)
        
        # Compute loss for this chunk
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                chunk_loss = compute_loss(chunk_logits, chunk_input_ids, pad_token_id)
        else:
            chunk_loss = compute_loss(chunk_logits, chunk_input_ids, pad_token_id)
        
        # Count tokens in this chunk
        chunk_tokens = (chunk_input_ids[:, 1:] != pad_token_id).sum().item()
        total_loss += chunk_loss.item() * chunk_tokens
        total_tokens += chunk_tokens
        
        # Backward for this chunk only!
        if scaler is not None:
            scaler.scale(chunk_loss).backward()
        else:
            chunk_loss.backward()
        
        # Truncate gradients (detach state for next chunk)
        if chunk_end < S:
            model.ssm.detach_cache()
            window_mgr.detach()
        
        n_chunks += 1
    
    # Gradient clipping and optimizer step (after all chunks)
    if scaler is not None:
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    
    optimizer.zero_grad()
    
    # Cleanup
    model.ssm.clear_cache()
    
    # Compute average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    
    aux = {
        'write_steps': all_write_steps,
        'n_writes': len(all_write_steps),
        'n_chunks': n_chunks,
    }
    
    return avg_loss, aux


def compute_loss(logits, input_ids, pad_token_id):
    """Compute language modeling loss."""
    # Shift for next token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Cross entropy loss
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_token_id,
        reduction="mean"
    )
    
    return loss


@torch.no_grad()
def evaluate(model, dataloader, device, pad_token_id, max_batches=None):
    """Evaluate on validation set"""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        
        # Forward pass (return_aux=False, only returns logits)
        logits = model(input_ids, return_aux=False)
        
        # Shift for language modeling
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Calculate loss (ignore padding tokens)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction="sum"
        )
        
        # Count non-padding tokens
        batch_tokens = (shift_labels != pad_token_id).sum().item()
        total_loss += loss.item()
        total_tokens += batch_tokens
        
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    print("=" * 80)
    print("SMT (STRIDE MEMORY TRANSFORMER) - EXPERIMENT RUNNER")
    print("=" * 80)
    
    # Load config
    cfg = load_config_file(args.config)
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    
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
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(exp_cfg["output_dir"])
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_dir}")
    
    # Save config
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"💾 Saved config to {config_save_path}")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    print(f"✅ Vocab size: {len(tokenizer)}")
    print(f"✅ Pad token ID: {pad_token_id}")
    
    # Prepare data
    train_dataset = prepare_wikitext2_data(cfg, tokenizer, split="train")
    val_dataset = prepare_wikitext2_data(cfg, tokenizer, split="validation")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["train_batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", True)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["eval_batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", True)
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Create model
    model = create_model(cfg)
    model = model.to(device)
    
    # AMP scaler (if enabled)
    use_amp = cfg['model'].get('use_amp', False)
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    if scaler:
        print(f"✅ Enabled AMP (Automatic Mixed Precision)")
    
    # Get chunk_size for TBPTT
    chunk_size = train_cfg.get('chunk_size', None)
    if chunk_size:
        print(f"✅ TBPTT enabled with chunk_size={chunk_size}")
    else:
        print(f"⚠️  TBPTT disabled (processing full sequences)")
    
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
    best_val_ppl = float('inf')
    start_time = time.time()
    
    # Metrics tracking
    metrics_history = {
        "train_loss": [],
        "train_steps": [],
        "val_loss": [],
        "val_ppl": [],
        "eval_steps": []
    }
    
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
                train_cfg["max_grad_norm"], pad_token_id, chunk_size, scaler
            )
            
            global_step += 1
            pbar.update(1)
            
            # Log
            if global_step % cfg["logging"]["logging_steps"] == 0:
                metrics_history["train_loss"].append(loss)
                metrics_history["train_steps"].append(global_step)
                
                train_ppl = np.exp(loss) if loss < 100 else float('inf')
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'ppl': f'{train_ppl:.2f}',
                    'writes': aux.get('n_writes', 0)
                })
            
            # Evaluate
            if global_step % train_cfg["eval_steps"] == 0:
                print(f"\n📊 Evaluation at step {global_step}...")
                val_loss, val_ppl = evaluate(
                    model, val_loader, device, pad_token_id,
                    max_batches=cfg["evaluation"].get("max_eval_batches")
                )
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Val PPL:  {val_ppl:.2f}")
                
                # Track metrics
                metrics_history["val_loss"].append(val_loss)
                metrics_history["val_ppl"].append(val_ppl)
                metrics_history["eval_steps"].append(global_step)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_ppl = val_ppl
                    checkpoint_path = output_dir / "best_model.pt"
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_ppl': val_ppl,
                        'config': cfg
                    }, checkpoint_path)
                    print(f"   💾 Saved best model (Loss: {val_loss:.4f}, PPL: {val_ppl:.2f})")
            
            # Save checkpoint
            if global_step % train_cfg["save_steps"] == 0:
                checkpoint_path = output_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg
                }, checkpoint_path)
                print(f"\n💾 Saved checkpoint at step {global_step}")
    
    # Final evaluation on test set (using validation set as proxy)
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    # Load best model
    best_checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print("📊 Evaluating best model on full validation set...")
    final_val_loss, final_val_ppl = evaluate(
        model, val_loader, device, pad_token_id, max_batches=None
    )
    print(f"   Final Val Loss: {final_val_loss:.4f}")
    print(f"   Final Val PPL:  {final_val_ppl:.2f}")
    
    # Calculate training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    # Prepare results summary
    model_cfg = cfg['model']
    window_cfg = model_cfg['window']
    
    results = {
        "experiment": {
            "name": exp_cfg["name"],
            "timestamp": timestamp,
            "seed": exp_cfg["seed"],
            "total_steps": global_step,
            "training_time": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "training_time_seconds": total_time
        },
        "model": {
            "model_type": model_cfg["model_type"],
            "parameters": model.count_parameters()["total"],
            "d_model": model_cfg["d_model"],
            "vocab_size": model_cfg["vocab_size"],
            "window": {
                "n_memory_tokens": window_cfg["n_memory_tokens"],
                "n_input_tokens": window_cfg["n_input_tokens"],
                "stride": window_cfg["stride"],
                "total_size": window_cfg["n_memory_tokens"] + window_cfg["n_input_tokens"]
            },
            "transformer": cfg['model']['transformer'],
            "ssm": cfg['model']['ssm']
        },
        "data": {
            "dataset_name": data_cfg["dataset_name"],
            "dataset_config": data_cfg["dataset_config"],
            "max_seq_length": data_cfg["max_seq_length"],
            "train_batch_size": data_cfg["train_batch_size"],
            "eval_batch_size": data_cfg["eval_batch_size"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        },
        "training": {
            "max_steps": train_cfg["max_steps"],
            "chunk_size": train_cfg.get("chunk_size"),
            "learning_rate": train_cfg["learning_rate"],
            "weight_decay": train_cfg["weight_decay"],
            "max_grad_norm": train_cfg["max_grad_norm"]
        },
        "results": {
            "best_val_loss": float(best_val_loss),
            "best_val_ppl": float(best_val_ppl),
            "best_step": int(best_checkpoint['step']),
            "final_val_loss": float(final_val_loss),
            "final_val_ppl": float(final_val_ppl)
        },
        "config": cfg
    }
    
    # Save results
    save_results(output_dir, results)
    save_metrics_history(output_dir, metrics_history)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"✅ Best validation PPL:  {best_val_ppl:.2f}")
    print(f"✅ Final validation loss: {final_val_loss:.4f}")
    print(f"✅ Final validation PPL:  {final_val_ppl:.2f}")
    print(f"⏱️  Training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📁 Results saved to: {output_dir}")
    print(f"   - config.yaml")
    print(f"   - results.json")
    print(f"   - metrics_history.json")
    print(f"   - best_model.pt")


if __name__ == "__main__":
    main()
