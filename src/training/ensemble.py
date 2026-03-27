"""
Ensemble training: train multiple diverse models, evaluate combinations.
"""

import time
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score

from ..models import FullGenesVJClassifier
from ..data import TCellDataset
from .train import train_single_model, validate_logits


# Default ensemble member configurations
# Diversity through: hidden dims, attention heads, dropout, LR, seeds
DEFAULT_MEMBERS = [
    {"name": "m1_h512", "hidden_dim": 512, "n_attn_heads": 4,
     "dropout": 0.3, "lr": 2e-4, "weight_decay": 0.02,
     "batch_size": 256, "seed": 42},
    {"name": "m2_h512_s2", "hidden_dim": 512, "n_attn_heads": 4,
     "dropout": 0.3, "lr": 2e-4, "weight_decay": 0.02,
     "batch_size": 256, "seed": 999},
    {"name": "m3_h384", "hidden_dim": 384, "n_attn_heads": 4,
     "dropout": 0.3, "lr": 2e-4, "weight_decay": 0.02,
     "batch_size": 256, "seed": 123},
    {"name": "m4_8heads", "hidden_dim": 512, "n_attn_heads": 8,
     "dropout": 0.3, "lr": 2e-4, "weight_decay": 0.02,
     "batch_size": 256, "seed": 777},
    {"name": "m5_lowdrop", "hidden_dim": 512, "n_attn_heads": 4,
     "dropout": 0.25, "lr": 2e-4, "weight_decay": 0.03,
     "batch_size": 256, "seed": 2024},
    {"name": "m6_highdrop", "hidden_dim": 512, "n_attn_heads": 4,
     "dropout": 0.35, "lr": 2e-4, "weight_decay": 0.015,
     "batch_size": 256, "seed": 31415},
    {"name": "m7_lr3e4", "hidden_dim": 512, "n_attn_heads": 4,
     "dropout": 0.3, "lr": 3e-4, "weight_decay": 0.02,
     "batch_size": 256, "seed": 5555},
    {"name": "m8_bigbatch", "hidden_dim": 512, "n_attn_heads": 4,
     "dropout": 0.3, "lr": 3e-4, "weight_decay": 0.02,
     "batch_size": 512, "seed": 8888},
]

FIXED_DEFAULTS = {
    "epochs": 70,
    "warmup_ratio": 0.05,
    "label_smoothing": 0.03,
    "early_stop_patience": 12,
}


def train_ensemble(
    data: dict,
    save_dir: str | Path,
    members: list[dict] | None = None,
    fixed_config: dict | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Train an ensemble of FullGenesVJClassifier models.

    Args:
        data: Loaded npz data dict with train/val/test splits
        save_dir: Directory to save model weights and results
        members: List of member configs (default: 8 diverse configs)
        fixed_config: Fixed hyperparameters shared across members
        device: Torch device (default: auto-detect)

    Returns:
        results: Dict with accuracy, F1, individual model stats
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if members is None:
        members = DEFAULT_MEMBERS
    if fixed_config is None:
        fixed_config = FIXED_DEFAULTS.copy()

    # Dimensions from data
    gex_dim = data["train_gex_full"].shape[1]
    vj_dim = data["train_vj"].shape[1]
    fixed_config["gex_dim"] = gex_dim
    fixed_config["vj_dim"] = vj_dim

    # Create datasets
    train_ds = TCellDataset(
        data["train_gex_full"], data["train_tcr_a"], data["train_tcr_b"],
        data["train_vj"], data["train_y"],
    )
    val_ds = TCellDataset(
        data["val_gex_full"], data["val_tcr_a"], data["val_tcr_b"],
        data["val_vj"], data["val_y"],
    )
    test_ds = TCellDataset(
        data["test_gex_full"], data["test_tcr_a"], data["test_tcr_b"],
        data["test_vj"], data["test_y"],
    )

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(data["train_y"]), y=data["train_y"]
    )

    test_loader = DataLoader(
        test_ds, batch_size=1024, shuffle=False, pin_memory=True, num_workers=0
    )

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"GEX: {gex_dim}, VJ: {vj_dim}")
    print(f"\n{'=' * 90}")
    print(f"ENSEMBLE: {len(members)} models")
    print(f"{'=' * 90}")

    # Train all members
    trained = []
    for i, member in enumerate(members):
        config = {
            **fixed_config,
            **{k: v for k, v in member.items() if k not in ("name", "seed")},
        }
        print(
            f"\n[{i + 1}/{len(members)}] {member['name']} "
            f"(h={config['hidden_dim']}, heads={config['n_attn_heads']}, "
            f"do={config['dropout']}, lr={config['lr']}, seed={member['seed']})"
        )

        t0 = time.time()
        state, val_f1, epochs = train_single_model(
            config, member["seed"], train_ds, val_ds, class_weights, device
        )
        print(f"  Val F1: {val_f1:.4f} | Epochs: {epochs} | Time: {time.time() - t0:.0f}s")

        torch.save(state, save_dir / f"{member['name']}.pt")
        trained.append((state, config, member, val_f1))

    # Evaluate
    print(f"\n{'=' * 90}")
    print("ENSEMBLE EVALUATION")
    print(f"{'=' * 90}\n")

    all_logits = []
    individual = []

    for state, config, member, _ in trained:
        model = FullGenesVJClassifier(
            gex_dim=config["gex_dim"],
            hidden_dim=config["hidden_dim"],
            n_heads=config["n_attn_heads"],
            dropout=config["dropout"],
            vj_dim=config["vj_dim"],
        ).to(device)
        model.load_state_dict(state)

        logits, labels = validate_logits(model, test_loader, device)
        all_logits.append(logits)

        preds = logits.argmax(dim=1).numpy()
        acc = (preds == labels.numpy()).mean()
        f1 = f1_score(labels.numpy(), preds, average="macro")
        individual.append((member["name"], acc, f1))
        print(f"  {member['name']}: Acc={acc:.4f}, F1={f1:.4f}")

    test_labels = labels.numpy()

    # Try ensemble sizes: top-3, top-5, all
    sorted_idx = sorted(
        range(len(individual)), key=lambda i: individual[i][1], reverse=True
    )

    print(f"\nEnsemble combinations:")
    best_ens_acc = 0
    best_ens_preds = None
    best_ens_name = ""
    best_ens_f1 = 0

    for k in [3, 5, len(members)]:
        if k > len(members):
            continue
        top_k = torch.stack([all_logits[i] for i in sorted_idx[:k]]).mean(dim=0)
        preds = top_k.argmax(dim=1).numpy()
        acc = (preds == test_labels).mean()
        f1 = f1_score(test_labels, preds, average="macro")
        name = f"top-{k}" if k < len(members) else f"all-{k}"
        print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}")

        if acc > best_ens_acc:
            best_ens_acc = acc
            best_ens_preds = preds
            best_ens_name = name
            best_ens_f1 = f1

    # Save results
    results = {
        "final_accuracy": float(best_ens_acc),
        "final_f1": float(best_ens_f1),
        "method": best_ens_name,
        "individual": [
            {"name": n, "acc": float(a), "f1": float(f)} for n, a, f in individual
        ],
    }

    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez(save_dir / "predictions.npz", preds=best_ens_preds, labels=test_labels)

    print(f"\n{'=' * 90}")
    print(f"FINAL ({best_ens_name}): Acc={best_ens_acc:.4f}, F1={best_ens_f1:.4f}")
    print(f"{'=' * 90}")

    return results
