"""
Cross-Attention and Advanced Multimodal Architectures
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Cross-attention between two modalities"""
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        attn_out, attn_weights = self.attention(query, key_value, key_value)
        out = self.norm(query + self.dropout(attn_out))
        return out, attn_weights


class AdvancedMultimodalClassifier(nn.Module):
    """
    Advanced architecture with cross-attention fusion between GEX and TCR.
    """
    def __init__(self, gex_dim=50, tcr_dim=768, hidden_dim=256, n_classes=7, n_heads=4, dropout=0.3):
        super().__init__()
        
        # Projection layers
        self.gex_proj = nn.Sequential(
            nn.Linear(gex_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.tcr_a_proj = nn.Sequential(
            nn.Linear(tcr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.tcr_b_proj = nn.Sequential(
            nn.Linear(tcr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention
        self.gex_to_tcr_attn = CrossAttention(hidden_dim, n_heads, dropout)
        self.tcr_to_gex_attn = CrossAttention(hidden_dim, n_heads, dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.attn_weights = {}
    
    def forward(self, gex, tcr_a, tcr_b):
        # Project to common dimension
        gex_emb = self.gex_proj(gex).unsqueeze(1)
        tcr_a_emb = self.tcr_a_proj(tcr_a).unsqueeze(1)
        tcr_b_emb = self.tcr_b_proj(tcr_b).unsqueeze(1)
        
        tcr_emb = torch.cat([tcr_a_emb, tcr_b_emb], dim=1)
        
        # Cross-attention
        gex_attended, attn_g2t = self.gex_to_tcr_attn(gex_emb, tcr_emb)
        tcr_attended, attn_t2g = self.tcr_to_gex_attn(tcr_emb, gex_emb)
        
        self.attn_weights['gex_to_tcr'] = attn_g2t
        self.attn_weights['tcr_to_gex'] = attn_t2g
        
        # Combine
        combined = torch.cat([
            gex_attended.squeeze(1),
            tcr_attended[:, 0, :],
            tcr_attended[:, 1, :]
        ], dim=1)
        
        class_out = self.classifier(combined)
        activation_out = self.regressor(combined).squeeze(-1)
        
        return class_out, activation_out, combined
