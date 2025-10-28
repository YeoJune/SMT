"""
Compare multiple experiment results
Usage: python scripts\compare_experiments.py experiments\results\wikitext2_smt
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_experiment_results(result_dir):
    """Load all experiment results from a directory"""
    experiments = []
    
    for exp_dir in sorted(result_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        results_path = exp_dir / "results.json"
        if not results_path.exists():
            continue
        
        with open(results_path, "r") as f:
            results = json.load(f)
            results["dir_name"] = exp_dir.name
            experiments.append(results)
    
    return experiments


def print_comparison_table(experiments):
    """Print comparison table"""
    print("\n" + "=" * 120)
    print("EXPERIMENT COMPARISON")
    print("=" * 120)
    
    # Header
    print(f"\n{'Timestamp':<17} {'Steps':<8} {'Params':<8} {'Stride':<7} {'n/m':<8} "
          f"{'Best PPL':<10} {'Final PPL':<10} {'Time':<10}")
    print("-" * 120)
    
    # Sort by final perplexity
    experiments.sort(key=lambda x: x["results"]["final_val_ppl"])
    
    for exp in experiments:
        timestamp = exp["dir_name"]
        steps = exp["experiment"]["total_steps"]
        params = f"{exp['model']['parameters']/1e6:.1f}M"
        stride = exp["model"]["stride"]
        n_m = f"{exp['model']['n_ssm']}/{exp['model']['m_input']}"
        best_ppl = exp["results"]["best_val_ppl"]
        final_ppl = exp["results"]["final_val_ppl"]
        time = exp["experiment"]["training_time"]
        
        print(f"{timestamp:<17} {steps:<8} {params:<8} {stride:<7} {n_m:<8} "
              f"{best_ppl:<10.2f} {final_ppl:<10.2f} {time:<10}")
    
    print("=" * 120 + "\n")


def plot_comparison(experiments, output_path):
    """Plot comparison of experiments"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort by timestamp for consistent colors
    experiments.sort(key=lambda x: x["dir_name"])
    
    # Load metrics for each experiment
    for i, exp in enumerate(experiments):
        exp_dir = Path(args.result_dir) / exp["dir_name"]
        metrics_path = exp_dir / "metrics_history.json"
        
        if not metrics_path.exists():
            continue
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        label = f"{exp['dir_name']} (stride={exp['model']['stride']})"
        
        # Plot loss
        if metrics["eval_steps"]:
            axes[0].plot(metrics["eval_steps"], metrics["val_loss"], 
                        label=label, marker='o', alpha=0.7)
        
        # Plot perplexity
        if metrics["eval_steps"]:
            axes[1].plot(metrics["eval_steps"], metrics["val_ppl"], 
                        label=label, marker='o', alpha=0.7)
    
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Validation Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Validation Perplexity")
    axes[1].set_title("Validation Perplexity Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved comparison plot to {output_path}")
    plt.close()


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", help="Path to results directory containing experiments")
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    if not result_dir.exists():
        print(f"❌ Error: Directory {result_dir} does not exist")
        return
    
    # Load all experiments
    experiments = load_experiment_results(result_dir)
    
    if not experiments:
        print(f"❌ Error: No experiment results found in {result_dir}")
        return
    
    print(f"✅ Found {len(experiments)} experiments")
    
    # Print comparison table
    print_comparison_table(experiments)
    
    # Plot comparison
    if len(experiments) > 1:
        plot_path = result_dir / "experiment_comparison.png"
        plot_comparison(experiments, plot_path)
    
    print(f"✅ Comparison complete")


if __name__ == "__main__":
    main()
