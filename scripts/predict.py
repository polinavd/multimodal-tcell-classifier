#!/usr/bin/env python3
"""
Step 4: Run inference on new data and generate HTML report.

Takes an h5ad file with scRNA-seq + TCR annotations, computes features,
runs the ensemble, and produces a self-contained interactive HTML report.

Input:  Any h5ad file with gene expression and TCR annotations
Output: prediction_results/report.html
        prediction_results/predictions.csv
        prediction_results/annotated.h5ad

Usage:
    python scripts/predict.py --input new_data.h5ad --output ./results
    python scripts/predict.py --input new_data.h5ad --true-labels functional_state
"""

import argparse
import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import torch
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.predict import load_ensemble, ensemble_predict
from src.inference.report import generate_report
from src.data.preprocessing import prepare_inference_features
from src.data.dataset import InferenceDataset


def main():
    parser = argparse.ArgumentParser(description="T-Cell Functional State Predictor")
    parser.add_argument("--input", required=True, help="Input h5ad file")
    parser.add_argument("--output", default="./prediction_results", help="Output directory")
    parser.add_argument("--true-labels", default=None, help="Obs column with true labels")
    parser.add_argument("--model-dir", default="results/ensemble", help="Ensemble weights directory")
    parser.add_argument("--data-dir", default="data/processed", help="Preprocessed data directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    # Load encoders
    print("\n[1/5] Loading encoders...")
    with open(data_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(data_dir / "vj_encoder.pkl", "rb") as f:
        vj_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    # Load ensemble
    print("\n[2/5] Loading ensemble...")
    models = load_ensemble(model_dir, device)

    # Load input data
    print(f"\n[3/5] Loading: {args.input}")
    adata = anndata.read_h5ad(args.input)
    n_cells, n_genes = adata.shape
    barcodes = list(adata.obs_names)

    # Prepare features
    print("\n[4/5] Preparing features...")
    gex, tcr_a, tcr_b, vj_encoded, vj_raw = prepare_inference_features(
        adata, "wukevin/tcr-bert", vj_encoder, device
    )
    dataset = InferenceDataset(gex, tcr_a, tcr_b, vj_encoded)

    # Predict
    print("\n[5/5] Predicting...")
    predictions, probabilities, agreement = ensemble_predict(models, dataset, device)
    pred_labels = [class_names[p] for p in predictions]

    # True labels (optional)
    true_labels = None
    if args.true_labels and args.true_labels in adata.obs.columns:
        try:
            true_labels = label_encoder.transform(adata.obs[args.true_labels].values)
        except ValueError:
            print("  Warning: label mismatch, skipping evaluation")

    # Save CSV
    csv_path = output_dir / "predictions.csv"
    df = pd.DataFrame({
        "barcode": barcodes,
        "predicted_state": pred_labels,
        "confidence": probabilities.max(axis=1),
        "model_agreement": agreement,
    })
    for i, name in enumerate(class_names):
        df[f"prob_{name}"] = probabilities[:, i]
    df.to_csv(csv_path, index=False)

    # Save annotated h5ad
    adata.obs["predicted_state"] = pd.Categorical(pred_labels, categories=class_names)
    adata.obs["prediction_confidence"] = probabilities.max(axis=1)
    adata.obs["model_agreement"] = agreement
    h5ad_path = output_dir / "annotated.h5ad"
    adata.write(h5ad_path)

    # Generate HTML report
    generate_report(
        barcodes=barcodes,
        predictions=predictions,
        probabilities=probabilities,
        agreement=agreement,
        class_names=class_names,
        true_labels=true_labels,
        n_cells=n_cells,
        n_genes=n_genes,
        input_file=args.input,
        vj_raw=vj_raw,
        obs_df=adata.obs,
        output_path=str(output_dir / "report.html"),
        csv_filename="predictions.csv",
    )

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  {csv_path}")
    print(f"  {h5ad_path}")
    print(f"  {output_dir / 'report.html'}")
    if true_labels is not None:
        print(f"  Accuracy: {accuracy_score(true_labels, predictions):.1%}")
    print("\nOpen report.html in browser!")


if __name__ == "__main__":
    main()
