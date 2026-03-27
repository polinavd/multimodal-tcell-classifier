#!/usr/bin/env python3
"""
Step 2: Compute TCR-BERT embeddings for CDR3 sequences.

Requires GPU for reasonable speed. Processes alpha and beta chains
through TCR-BERT, extracting [CLS] token embeddings (768-dim).

Input:  data/processed/combined_with_tcr.h5ad
        data/processed/ml_splits.pkl
Output: Adds train/val/test_tcr_a and train/val/test_tcr_b to
        data/processed/ml_data_full_genes.npz

Usage:
    python scripts/extract_tcr_embeddings.py --data-dir data/processed
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pickle
import anndata
import torch
from transformers import AutoTokenizer, AutoModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.preprocessing import compute_tcr_bert_embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract TCR-BERT embeddings")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--bert-model", type=str, default="wukevin/tcr-bert")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=32)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load splits
    with open(data_dir / "ml_splits.pkl", "rb") as f:
        splits = pickle.load(f)

    # Load AnnData obs for CDR3 sequences
    print("Loading AnnData...")
    adata = anndata.read_h5ad(data_dir / "combined_with_tcr.h5ad", backed="r")
    obs = adata.obs[["cdr3a", "cdr3b"]].copy()
    adata.file.close()

    # Load TCR-BERT
    print(f"Loading {args.bert_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model).to(device)

    # Load existing data
    npz_path = data_dir / "ml_data_full_genes.npz"
    existing = dict(np.load(npz_path, allow_pickle=True))

    # Process each split
    for split_name, barcodes in splits.items():
        print(f"\n{split_name}: {len(barcodes)} cells")

        cdr3a = [str(obs.loc[bc, "cdr3a"]) if bc in obs.index and isinstance(obs.loc[bc, "cdr3a"], str) else "" for bc in barcodes]
        cdr3b = [str(obs.loc[bc, "cdr3b"]) if bc in obs.index and isinstance(obs.loc[bc, "cdr3b"], str) else "" for bc in barcodes]

        print("  CDR3-alpha embeddings...")
        emb_a = compute_tcr_bert_embeddings(
            cdr3a, tokenizer, model, device, args.batch_size, args.max_length
        )
        print("  CDR3-beta embeddings...")
        emb_b = compute_tcr_bert_embeddings(
            cdr3b, tokenizer, model, device, args.batch_size, args.max_length
        )

        existing[f"{split_name}_tcr_a"] = emb_a
        existing[f"{split_name}_tcr_b"] = emb_b

    # Save
    print(f"\nSaving to {npz_path}...")
    np.savez_compressed(npz_path, **existing)
    print(f"  File size: {npz_path.stat().st_size / 1024**2:.1f} MB")

    # Verify
    check = np.load(npz_path, allow_pickle=True)
    print("\nKeys:")
    for k in sorted(check.keys()):
        print(f"  {k}: shape={check[k].shape}, dtype={check[k].dtype}")

    del model
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
