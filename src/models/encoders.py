"""
Encoder modules for different modalities
"""

import torch
import torch.nn as nn


class GEXEncoder(nn.Module):
    """Gene Expression Encoder"""
    def __init__(self, input_dim=50, hidden_dim=128, output_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class TCREncoder(nn.Module):
    """TCR Embedding Encoder"""
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class GEXOnlyClassifier(nn.Module):
    """GEX-only baseline for ablation study"""
    def __init__(self, gex_dim=50, hidden_dim=256, n_classes=7, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gex_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, gex):
        return self.encoder(gex)


class TCROnlyClassifier(nn.Module):
    """TCR-only baseline for ablation study"""
    def __init__(self, tcr_dim=768, hidden_dim=256, n_classes=7, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(tcr_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, tcr_a, tcr_b):
        tcr = torch.cat([tcr_a, tcr_b], dim=1)
        return self.encoder(tcr)
