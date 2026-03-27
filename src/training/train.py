"""
Single-model training loop with mixed precision, cosine schedule, and early stopping.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ..models import FullGenesVJClassifier


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine annealing with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    """One training epoch with AMP and gradient clipping."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for gex, tcr_a, tcr_b, vj, labels in loader:
        gex = gex.to(device)
        tcr_a = tcr_a.to(device)
        tcr_b = tcr_b.to(device)
        vj = vj.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(gex, tcr_a, tcr_b, vj)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        _, pred = logits.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and labels from validation/test set."""
    model.eval()
    all_logits, all_labels = [], []

    for gex, tcr_a, tcr_b, vj, labels in loader:
        gex = gex.to(device)
        tcr_a = tcr_a.to(device)
        tcr_b = tcr_b.to(device)
        vj = vj.to(device)

        with autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(gex, tcr_a, tcr_b, vj)

        all_logits.append(logits.float().cpu())
        all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def train_single_model(
    config: dict,
    seed: int,
    train_ds,
    val_ds,
    class_weights: np.ndarray,
    device: torch.device,
) -> tuple[dict, float, int]:
    """
    Train a single FullGenesVJClassifier with given config.

    Args:
        config: Training hyperparameters (hidden_dim, n_attn_heads,
                dropout, lr, weight_decay, batch_size, epochs, etc.)
        seed: Random seed for reproducibility
        train_ds: Training dataset
        val_ds: Validation dataset
        class_weights: Class weights for imbalanced data
        device: Torch device

    Returns:
        best_state: State dict of best model (by val macro F1)
        best_val_f1: Best validation macro F1
        n_epochs: Number of epochs trained
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    model = FullGenesVJClassifier(
        gex_dim=config["gex_dim"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_attn_heads"],
        dropout=config["dropout"],
        vj_dim=config["vj_dim"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device),
        label_smoothing=config.get("label_smoothing", 0.03),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.02),
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler("cuda")

    best_val_f1 = 0.0
    best_state = None
    patience = 0
    early_stop = config.get("early_stop_patience", 12)

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )

        val_logits, val_labels = validate_logits(model, val_loader, device)
        val_f1 = f1_score(
            val_labels.numpy(), val_logits.argmax(dim=1).numpy(), average="macro"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= early_stop:
            break

    return best_state, best_val_f1, epoch + 1
