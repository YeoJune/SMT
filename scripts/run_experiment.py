"""Simple experiment runner that reads a YAML config and runs a tiny toy experiment.

Usage (PowerShell):
python .\scripts\run_experiment.py --config experiments\configs\example_experiment.yaml
"""
import argparse
import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, output_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: (B, S) token ids
        emb = self.embed(x)  # (B, S, E)
        logits = self.proj(emb)  # (B, S, output_dim)
        return logits


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    name = cfg.get("name", "experiment")
    seed = int(cfg.get("seed", 0))
    device = cfg.get("device", "cpu")

    set_seed(seed)

    data_cfg = cfg.get("data", {})
    batch = int(data_cfg.get("batch_size", 2))
    seq_len = int(data_cfg.get("seq_len", 32))
    vocab_size = int(data_cfg.get("vocab_size", 1000))

    model_cfg = cfg.get("model", {})
    embed_dim = int(model_cfg.get("embed_dim", 64))
    output_dim = int(model_cfg.get("output_dim", 1000))

    training_cfg = cfg.get("training", {})
    steps = int(training_cfg.get("steps", 1))
    save_dir = training_cfg.get("save_dir", f"experiments/results/{name}")

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # Build a simple model
    model = SimpleModel(vocab_size=vocab_size, embed_dim=embed_dim, output_dim=output_dim)
    model.to(device)
    model.eval()

    # Create a tiny random dataset (token ids)
    inputs = torch.randint(low=0, high=vocab_size, size=(batch, seq_len), dtype=torch.long, device=device)

    # Run a few steps (here, only forward passes)
    all_outputs = []
    for step in range(steps):
        with torch.no_grad():
            logits = model(inputs)  # (B, S, output_dim)
        print(f"Step {step+1}/{steps}: logits shape = {tuple(logits.shape)}")
        all_outputs.append(logits.cpu().numpy())

    # Save final logits as numpy file
    out_arr = np.stack(all_outputs, axis=0)  # (steps, B, S, output_dim)
    out_path = os.path.join(save_dir, "output.npy")
    np.save(out_path, out_arr)
    print(f"Saved outputs to {out_path}")


if __name__ == "__main__":
    main()
