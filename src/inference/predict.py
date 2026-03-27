"""
Ensemble inference: load trained models, predict, compute agreement.
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.amp import autocast
from torch.utils.data import DataLoader

from ..models import FullGenesVJClassifier
from ..data import InferenceDataset


def load_ensemble(
    model_dir: str | Path,
    device: torch.device,
    gex_dim: int = 3000,
    vj_dim: int = 161,
    top_k: int = 5,
) -> list[FullGenesVJClassifier]:
    """
    Load top-K models from ensemble directory.

    Uses results.json to determine the best models by accuracy.
    Falls back to loading all m*.pt files if results.json is missing.

    Args:
        model_dir: Directory containing .pt weights and results.json
        device: Torch device
        gex_dim: Gene expression input dimension
        vj_dim: V/J one-hot encoding dimension
        top_k: Number of top models to load

    Returns:
        List of loaded FullGenesVJClassifier models
    """
    model_dir = Path(model_dir)

    # Determine which models to load
    results_path = model_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        ranked = sorted(results["individual"], key=lambda x: x["acc"], reverse=True)
        model_files = [model_dir / f"{m['name']}.pt" for m in ranked[:top_k]]
    else:
        model_files = sorted(model_dir.glob("m*.pt"))[:top_k]

    # Load models with architecture detection from name
    models = []
    for mf in model_files:
        name = mf.stem
        hidden_dim = 512 if any(
            x in name for x in ["h512", "8heads", "lowdrop", "highdrop", "lr3e4", "bigbatch"]
        ) else 384
        n_heads = 8 if "8heads" in name else 4
        dropout = 0.25 if "lowdrop" in name else (0.35 if "highdrop" in name else 0.3)

        model = FullGenesVJClassifier(
            gex_dim=gex_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            vj_dim=vj_dim,
        ).to(device)
        model.load_state_dict(
            torch.load(mf, map_location=device, weights_only=True)
        )
        model.eval()
        models.append(model)
        print(f"  Loaded {name} (h={hidden_dim}, heads={n_heads})")

    return models


@torch.no_grad()
def ensemble_predict(
    models: list[FullGenesVJClassifier],
    dataset: InferenceDataset,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run ensemble prediction with probability averaging.

    Args:
        models: List of trained models
        dataset: InferenceDataset with features
        device: Torch device
        batch_size: Inference batch size

    Returns:
        predictions: (N,) predicted class indices
        probabilities: (N, n_classes) softmax probabilities (averaged)
        agreement: (N,) fraction of models agreeing with ensemble prediction
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    all_logits = []
    for model in models:
        model.eval()
        model_logits = []
        for gex, tcr_a, tcr_b, vj in loader:
            gex = gex.to(device)
            tcr_a = tcr_a.to(device)
            tcr_b = tcr_b.to(device)
            vj = vj.to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(gex, tcr_a, tcr_b, vj)

            model_logits.append(logits.float().cpu())
        all_logits.append(torch.cat(model_logits, dim=0))

    # Average logits, softmax, predict
    avg_logits = torch.stack(all_logits).mean(dim=0)
    probabilities = torch.softmax(avg_logits, dim=1).numpy()
    predictions = avg_logits.argmax(dim=1).numpy()

    # Compute model agreement
    individual_preds = [l.argmax(dim=1).numpy() for l in all_logits]
    agreement = np.zeros(len(predictions))
    for p in individual_preds:
        agreement += (p == predictions).astype(float)
    agreement /= len(models)

    return predictions, probabilities, agreement
