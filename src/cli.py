"""
CLI entrypoint: tcell-predict

Usage:
    tcell-predict input.h5ad
    tcell-predict input.h5ad -o results/
    tcell-predict input.h5ad --true-labels functional_state
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import torch
from sklearn.metrics import accuracy_score

from .hub import ensure_weights
from .inference.predict import load_ensemble, ensemble_predict
from .inference.report import generate_report
from .data.preprocessing import prepare_inference_features
from .data.dataset import InferenceDataset


def main():
    parser = argparse.ArgumentParser(
        prog="tcell-predict",
        description="Predict T-cell functional states from scRNA-seq + TCR data.",
        epilog="Output: report.html (interactive dashboard), predictions.csv, annotated.h5ad",
    )
    parser.add_argument("input", help="Input .h5ad file with gene expression and TCR annotations")
    parser.add_argument("-o", "--output", default="./tcell_results", help="Output directory (default: ./tcell_results)")
    parser.add_argument("--true-labels", default=None, help="Column in adata.obs with true labels (enables evaluation)")
    parser.add_argument("--cache-dir", default=None, help="Override model cache directory (default: ~/.cache/tcell-classifier)")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, or auto (default: auto)")
    args = parser.parse_args()

    # Device
    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Ensure weights are downloaded
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    model_dir = ensure_weights(cache_dir)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoders
    print("\n[1/5] Loading encoders...")
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(model_dir / "vj_encoder.pkl", "rb") as f:
        vj_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    # Load ensemble
    print("\n[2/5] Loading ensemble (top-5)...")
    models = load_ensemble(model_dir, device)

    # Load input
    print(f"\n[3/5] Loading: {args.input}")
    adata = anndata.read_h5ad(args.input)
    n_cells, n_genes = adata.shape
    barcodes = list(adata.obs_names)
    print(f"  {n_cells:,} cells, {n_genes:,} genes")

    # Features
    print("\n[4/5] Computing features (TCR-BERT embeddings + V/J encoding)...")
    gex, tcr_a, tcr_b, vj_encoded, vj_raw = prepare_inference_features(
        adata, "wukevin/tcr-bert", vj_encoder, device
    )
    dataset = InferenceDataset(gex, tcr_a, tcr_b, vj_encoded)

    # Predict
    print("\n[5/5] Running ensemble prediction...")
    predictions, probabilities, agreement = ensemble_predict(models, dataset, device)
    pred_labels = [class_names[p] for p in predictions]

    # True labels
    true_labels = None
    if args.true_labels and args.true_labels in adata.obs.columns:
        try:
            true_labels = label_encoder.transform(adata.obs[args.true_labels].values)
        except ValueError:
            print("  Warning: some labels not in training set, skipping evaluation")

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

    # Generate report
    report_path = generate_report(
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

    # Summary
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Report:      {output_dir / 'report.html'}")
    print(f"  Predictions: {csv_path}")
    print(f"  Annotated:   {h5ad_path}")
    if true_labels is not None:
        acc = accuracy_score(true_labels, predictions)
        print(f"  Accuracy:    {acc:.1%}")
    print(f"\nOpen report.html in your browser!")


if __name__ == "__main__":
    main()
