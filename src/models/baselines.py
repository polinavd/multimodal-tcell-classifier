"""
Unimodal baseline models for ablation study.

These models isolate individual modalities to quantify
the contribution of each to the final multimodal performance.
"""

import torch
import torch.nn as nn


class GEXOnlyClassifier(nn.Module):
    """Gene expression-only baseline (PCA-reduced)."""

    def __init__(
        self, gex_dim: int = 50, hidden_dim: int = 256, n_classes: int = 7, dropout: float = 0.3
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gex_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, gex: torch.Tensor) -> torch.Tensor:
        return self.encoder(gex)


class TCROnlyClassifier(nn.Module):
    """TCR-BERT embedding-only baseline (alpha + beta concatenated)."""

    def __init__(
        self, tcr_dim: int = 768, hidden_dim: int = 256, n_classes: int = 7, dropout: float = 0.3
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(tcr_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, tcr_a: torch.Tensor, tcr_b: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([tcr_a, tcr_b], dim=1))
