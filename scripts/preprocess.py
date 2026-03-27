#!/usr/bin/env python3
"""
Step 1: Preprocess raw scRNA-seq data into ML-ready features.

This script assumes you have already:
  1. Downloaded raw data from GEO (GSE144469, GSE179994, GSE181061, GSE108989)
  2. Converted to h5ad format (see data/README.md)
  3. Performed QC, normalization, HVG selection, and Harmony batch correction
  4. Combined datasets into a single AnnData with TCR annotations

Input:  data/processed/combined_with_tcr.h5ad
Output: data/processed/ml_data_full_genes.npz
        data/processed/ml_data_with_vj.npz
        data/processed/vj_encoder.pkl
        data/processed/label_encoder.pkl
        data/processed/ml_splits.pkl

Usage:
    python scripts/preprocess.py --data-dir data/processed
"""

import argparse
import numpy as np
import pickle
import anndata
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Preprocess scRNA-seq data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with combined_with_tcr.h5ad",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15, help="Test split fraction"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.15, help="Validation split fraction"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    adata_path = data_dir / "combined_with_tcr.h5ad"

    print(f"Loading {adata_path}...")
    adata = anndata.read_h5ad(adata_path)
    print(f"  Shape: {adata.shape}")
    print(f"  Obs columns: {list(adata.obs.columns[:20])}")

    # Functional state labels
    label_col = "functional_state"
    assert label_col in adata.obs.columns, f"Column '{label_col}' not found in obs"

    le = LabelEncoder()
    labels = le.fit_transform(adata.obs[label_col].values)
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Train/val/test split (stratified, by barcode)
    barcodes = list(adata.obs_names)
    train_bc, test_bc, train_y, test_y = train_test_split(
        barcodes, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    train_bc, val_bc, train_y, val_y = train_test_split(
        train_bc, train_y,
        test_size=args.val_size / (1 - args.test_size),
        random_state=args.seed, stratify=train_y,
    )

    splits = {"train": train_bc, "val": val_bc, "test": test_bc}
    print(f"\n  Splits: train={len(train_bc)}, val={len(val_bc)}, test={len(test_bc)}")

    with open(data_dir / "ml_splits.pkl", "wb") as f:
        pickle.dump(splits, f)
    with open(data_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # Extract V/J genes
    print("\nExtracting V/J genes...")
    vj_cols = ["v_alpha", "j_alpha", "v_beta", "j_beta"]
    obs = adata.obs[vj_cols].copy()

    all_barcodes = train_bc + val_bc + test_bc
    vj_data = []
    for bc in all_barcodes:
        row = []
        for col in vj_cols:
            val = obs.loc[bc, col]
            row.append(str(val) if isinstance(val, str) else "UNKNOWN")
        vj_data.append(row)

    vj_array = np.array(vj_data)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    vj_encoded = encoder.fit_transform(vj_array)
    print(f"  V/J one-hot shape: {vj_encoded.shape}")

    with open(data_dir / "vj_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    # Extract full gene expression
    print("\nExtracting gene expression (all genes)...")
    bc_to_idx = {bc: i for i, bc in enumerate(adata.obs_names)}

    data_dict = {}
    idx_offset = 0
    for split_name, split_bcs in [("train", train_bc), ("val", val_bc), ("test", test_bc)]:
        adata_indices = [bc_to_idx[bc] for bc in split_bcs]
        gex = np.array(adata.X[adata_indices], dtype=np.float32)

        split_size = len(split_bcs)
        vj_split = vj_encoded[idx_offset : idx_offset + split_size].astype(np.float32)
        idx_offset += split_size

        data_dict[f"{split_name}_gex_full"] = gex
        data_dict[f"{split_name}_vj"] = vj_split
        data_dict[f"{split_name}_y"] = le.transform(
            adata.obs.loc[split_bcs, label_col].values
        )

        print(f"  {split_name}: {gex.shape}")

    # TCR-BERT embeddings need to be computed separately (requires GPU + model)
    # See scripts/extract_tcr_embeddings.py
    print(
        "\nNote: TCR-BERT embeddings must be computed separately."
        "\nRun: python scripts/extract_tcr_embeddings.py --data-dir data/processed"
    )

    output_path = data_dir / "ml_data_full_genes.npz"
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(output_path, **data_dict)
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
