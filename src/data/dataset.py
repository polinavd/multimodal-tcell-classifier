"""
PyTorch Dataset classes for T-cell multimodal data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class TCellDataset(Dataset):
    """
    Dataset for the full multimodal pipeline.

    Holds pre-computed features: gene expression (3,000 HVGs),
    TCR-BERT embeddings (alpha/beta), and V/J one-hot encoding.
    """

    def __init__(
        self,
        gex: np.ndarray,
        tcr_a: np.ndarray,
        tcr_b: np.ndarray,
        vj: np.ndarray,
        labels: np.ndarray,
    ):
        self.gex = torch.FloatTensor(gex)
        self.tcr_a = torch.FloatTensor(tcr_a)
        self.tcr_b = torch.FloatTensor(tcr_b)
        self.vj = torch.FloatTensor(vj)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.gex[idx],
            self.tcr_a[idx],
            self.tcr_b[idx],
            self.vj[idx],
            self.labels[idx],
        )


class InferenceDataset(Dataset):
    """Dataset for inference (no labels)."""

    def __init__(
        self,
        gex: np.ndarray,
        tcr_a: np.ndarray,
        tcr_b: np.ndarray,
        vj: np.ndarray,
    ):
        self.gex = torch.FloatTensor(gex)
        self.tcr_a = torch.FloatTensor(tcr_a)
        self.tcr_b = torch.FloatTensor(tcr_b)
        self.vj = torch.FloatTensor(vj)

    def __len__(self) -> int:
        return len(self.gex)

    def __getitem__(self, idx: int) -> tuple:
        return self.gex[idx], self.tcr_a[idx], self.tcr_b[idx], self.vj[idx]
