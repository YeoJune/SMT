"""
Plot training results from experiment
Usage: python scripts\plot_results.py --result_dir experiments\results\wikitext2_smt\20241028_123456
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(metrics_history, output_path):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax = axes[0]
    if metrics_history["train_steps"]:
        ax.plot(metrics_history["train_steps"], metrics_history["train_loss"], 
                label="Train Loss", alpha=0.7, linewidth=1)
    if metrics_history["eval_steps"]:
        ax.plot(metrics_history["eval_steps"], metrics_history["val_loss"], 
                label="Val Loss", marker='o', linewidth=2)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot perplexity
    ax = axes[1]
    if metrics_history["eval_steps"]:
        ax.plot(metrics_history["eval_steps"], metrics_history["val_ppl"], 
                label="Val Perplexity", marker='o', linewidth=2, color='orange')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved plot to {output_path}")
    plt.close()


def print_results_summary(results):
    """Print formatted results summary"""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n📋 Experiment: {results['experiment']['name']}")
    print(f"   Timestamp: {results['experiment']['timestamp']}")
    print(f"   Seed: {results['experiment']['seed']}")
    print(f"   Training Time: {results['experiment']['training_time']}")
    
    print(f"\n🔧 Model Configuration:")
    model_info = results['model']
    
    # Handle both old and new config formats
    if "type" in model_info:
        print(f"   Type: {model_info['type']}")
    elif "model_type" in model_info:
        print(f"   Type: {model_info['model_type']}")
    
    print(f"   Parameters: {model_info['parameters']/1e6:.1f}M")
    print(f"   d_model: {model_info['d_model']}")
    
    if "n_layers" in model_info:
        print(f"   n_layers: {model_info['n_layers']}")
    elif "transformer" in model_info:
        print(f"   n_layers: {model_info['transformer']['num_layers']}")
    
    if "stride" in model_info:
        print(f"   stride: {model_info['stride']}")
        print(f"   n_ssm: {model_info['n_ssm']}")
        print(f"   m_input: {model_info['m_input']}")
    elif "window" in model_info:
        print(f"   stride: {model_info['window']['stride']}")
        print(f"   n_memory_tokens: {model_info['window']['n_memory_tokens']}")
        print(f"   n_input_tokens: {model_info['window']['n_input_tokens']}")
        print(f"   total_window_size: {model_info['window']['total_size']}")
    
    print(f"\n📚 Dataset:")
    print(f"   Name: {results['data']['dataset']}")
    print(f"   Max Seq Length: {results['data']['max_seq_length']}")
    print(f"   Train Samples: {results['data']['train_samples']}")
    print(f"   Val Samples: {results['data']['val_samples']}")
    
    print(f"\n📊 Results:")
    print(f"   Best Val Loss: {results['results']['best_val_loss']:.4f}")
    print(f"   Best Val PPL:  {results['results']['best_val_ppl']:.2f}")
    print(f"   Best Step:     {results['results']['best_step']}")
    print(f"   Final Val Loss: {results['results']['final_val_loss']:.4f}")
    print(f"   Final Val PPL:  {results['results']['final_val_ppl']:.2f}")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True, help="Path to results directory")
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    if not result_dir.exists():
        print(f"❌ Error: Directory {result_dir} does not exist")
        return
    
    # Load results
    results_path = result_dir / "results.json"
    metrics_path = result_dir / "metrics_history.json"
    
    if not results_path.exists():
        print(f"❌ Error: {results_path} not found")
        return
    
    if not metrics_path.exists():
        print(f"❌ Error: {metrics_path} not found")
        return
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    with open(metrics_path, "r") as f:
        metrics_history = json.load(f)
    
    # Print summary
    print_results_summary(results)
    
    # Plot curves
    plot_path = result_dir / "training_curves.png"
    plot_training_curves(metrics_history, plot_path)
    
    print(f"✅ Results visualization complete")
    print(f"📁 Saved to: {result_dir}")


if __name__ == "__main__":
    main()
