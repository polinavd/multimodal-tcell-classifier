#!/usr/bin/env python3
"""
Step 3: Train the ensemble of FullGenesVJClassifier models.

Trains 8 diverse models with varied hyperparameters and selects
the best ensemble combination (top-3, top-5, or all-8).

Input:  data/processed/ml_data_full_genes.npz (with TCR embeddings)
        data/processed/label_encoder.pkl
Output: results/ensemble/*.pt (model weights)
        results/ensemble/results.json

Usage:
    python scripts/train_ensemble.py --data-dir data/processed --save-dir results/ensemble
    python scripts/train_ensemble.py --data-dir data/processed --epochs 70 --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.ensemble import train_ensemble, DEFAULT_MEMBERS, FIXED_DEFAULTS


def main():
    parser = argparse.ArgumentParser(description="Train ensemble")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--save-dir", type=str, default="results/ensemble")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
    data = np.load(Path(args.data_dir) / "ml_data_full_genes.npz", allow_pickle=True)

    fixed = FIXED_DEFAULTS.copy()
    fixed["epochs"] = args.epochs

    results = train_ensemble(
        data=data,
        save_dir=args.save_dir,
        members=DEFAULT_MEMBERS,
        fixed_config=fixed,
        device=device,
    )

    print(f"\nFinal: Acc={results['final_accuracy']:.4f}, F1={results['final_f1']:.4f}")
    print(f"Method: {results['method']}")


if __name__ == "__main__":
    main()
