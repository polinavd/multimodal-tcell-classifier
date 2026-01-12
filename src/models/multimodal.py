"""
Multimodal T-Cell Classifier Models
"""

import torch
import torch.nn as nn


class MultimodalTCellClassifier(nn.Module):
    """
    Basic multimodal classifier with concatenation fusion.
    Combines GEX (gene expression) and TCR (T-cell receptor) embeddings.
    """
    def __init__(self, gex_dim=50, tcr_dim=768, hidden_dim=256, n_classes=7, dropout=0.3):
        super().__init__()
        
        self.gex_encoder = nn.Sequential(
            nn.Linear(gex_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )
        
        self.tcr_a_encoder = nn.Sequential(
            nn.Linear(tcr_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim)
        )
        
        self.tcr_b_encoder = nn.Sequential(
            nn.Linear(tcr_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, gex, tcr_a, tcr_b):
        gex_emb = self.gex_encoder(gex)
        tcr_a_emb = self.tcr_a_encoder(tcr_a)
        tcr_b_emb = self.tcr_b_encoder(tcr_b)
        
        combined = torch.cat([gex_emb, tcr_a_emb, tcr_b_emb], dim=1)
        return self.classifier(combined)
    
    def get_latent(self, gex, tcr_a, tcr_b):
        """Extract latent representation before classifier"""
        gex_emb = self.gex_encoder(gex)
        tcr_a_emb = self.tcr_a_encoder(tcr_a)
        tcr_b_emb = self.tcr_b_encoder(tcr_b)
        return torch.cat([gex_emb, tcr_a_emb, tcr_b_emb], dim=1)


class MultiTaskTCellModel(nn.Module):
    """
    Multi-task model: classification + activation score regression.
    """
    def __init__(self, gex_dim=50, tcr_dim=768, hidden_dim=256, n_classes=7, dropout=0.3):
        super().__init__()
        
        # Shared encoders
        self.gex_encoder = nn.Sequential(
            nn.Linear(gex_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )
        
        self.tcr_a_encoder = nn.Sequential(
            nn.Linear(tcr_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim)
        )
        
        self.tcr_b_encoder = nn.Sequential(
            nn.Linear(tcr_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim)
        )
        
        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.regressor = nn.Linear(hidden_dim, 1)
    
    def forward(self, gex, tcr_a, tcr_b):
        gex_emb = self.gex_encoder(gex)
        tcr_a_emb = self.tcr_a_encoder(tcr_a)
        tcr_b_emb = self.tcr_b_encoder(tcr_b)
        
        combined = torch.cat([gex_emb, tcr_a_emb, tcr_b_emb], dim=1)
        shared_repr = self.shared(combined)
        
        class_out = self.classifier(shared_repr)
        activation_out = self.regressor(shared_repr).squeeze(-1)
        
        return class_out, activation_out, shared_repr
