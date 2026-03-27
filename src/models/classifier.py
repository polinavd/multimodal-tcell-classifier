"""
FullGenesVJClassifier — final multimodal architecture.

Integrates three modalities via cross-attention fusion:
  1. Gene expression (3,000 HVGs) — learned dimensionality reduction
  2. TCR-BERT embeddings (768-dim CLS tokens for alpha/beta chains)
  3. V/J gene usage (one-hot encoded, 161-dim)

Architecture:
  GEX (3000) -> [Linear 512 -> GELU -> Linear hidden] + ResidualBlock -> (hidden,)
  TCR-alpha/beta (768) + VJ context (64) -> [Linear hidden] + ResidualBlock -> (hidden,)
  VJ (161) -> [Linear hidden] -> (hidden,)

  Cross-Attention Fusion:
    GEX (1 token) attends to TCR tokens (alpha, beta, VJ) and vice versa
    Output: 4 * hidden_dim

  Post-fusion ResidualBlock -> Classifier head -> 7 classes
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-norm residual block with two linear layers."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.net(x)))


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between GEX and TCR modalities."""

    def __init__(self, dim: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gex_to_tcr = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.tcr_to_gex = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_gex = nn.LayerNorm(dim)
        self.norm_tcr = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, gex_emb: torch.Tensor, tcr_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gex_emb: (B, 1, dim) — single GEX token
            tcr_tokens: (B, 3, dim) — alpha, beta, VJ tokens

        Returns:
            gex_out: (B, 1, dim)
            tcr_out: (B, 3, dim)
        """
        gex_att, _ = self.gex_to_tcr(gex_emb, tcr_tokens, tcr_tokens)
        gex_out = self.norm_gex(gex_emb + self.dropout(gex_att))

        tcr_att, _ = self.tcr_to_gex(tcr_tokens, gex_emb, gex_emb)
        tcr_out = self.norm_tcr(tcr_tokens + self.dropout(tcr_att))

        return gex_out, tcr_out


class FullGenesVJClassifier(nn.Module):
    """
    Multimodal T-cell functional state classifier.

    Uses all 3,000 highly variable genes (no PCA), TCR-BERT embeddings
    for alpha/beta CDR3 sequences, and one-hot encoded V/J gene usage.
    Cross-attention fusion combines modalities before classification.

    Args:
        gex_dim: Number of input genes (default: 3000)
        tcr_dim: TCR-BERT embedding dimension (default: 768)
        vj_dim: V/J one-hot encoding dimension (default: 161)
        hidden_dim: Internal representation dimension (default: 384)
        n_classes: Number of functional state classes (default: 7)
        n_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        gex_dim: int = 3000,
        tcr_dim: int = 768,
        vj_dim: int = 161,
        hidden_dim: int = 384,
        n_classes: int = 7,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # GEX encoder: deeper network for high-dimensional input
        # Acts as learned dimensionality reduction (better than PCA for the task)
        self.gex_encoder = nn.Sequential(
            nn.Linear(gex_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gex_residual = ResidualBlock(hidden_dim, dropout)

        # V/J context encoder (injected into TCR pathway)
        self.vj_encoder = nn.Sequential(
            nn.Linear(vj_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # TCR encoder: takes BERT embedding + VJ context
        self.tcr_encoder = nn.Sequential(
            nn.Linear(tcr_dim + 64, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tcr_residual = ResidualBlock(hidden_dim, dropout)

        # V/J as a separate token for cross-attention
        self.vj_token_proj = nn.Sequential(
            nn.Linear(vj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(hidden_dim, n_heads, dropout)

        # Post-fusion processing
        fused_dim = hidden_dim * 4  # GEX + TCR-alpha + TCR-beta + VJ
        self.post_fusion = ResidualBlock(fused_dim, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        gex: torch.Tensor,
        tcr_a: torch.Tensor,
        tcr_b: torch.Tensor,
        vj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gex: (B, gex_dim) gene expression profile
            tcr_a: (B, 768) TCR-BERT embedding for CDR3-alpha
            tcr_b: (B, 768) TCR-BERT embedding for CDR3-beta
            vj: (B, vj_dim) one-hot encoded V/J gene usage

        Returns:
            logits: (B, n_classes) classification logits
            fused: (B, hidden_dim * 4) latent representation
        """
        # Encode V/J context
        vj_context = self.vj_encoder(vj)

        # Encode GEX
        gex_emb = self.gex_residual(self.gex_encoder(gex))

        # Encode TCR chains with V/J context
        tcr_a_emb = self.tcr_residual(
            self.tcr_encoder(torch.cat([tcr_a, vj_context], dim=1))
        )
        tcr_b_emb = self.tcr_residual(
            self.tcr_encoder(torch.cat([tcr_b, vj_context], dim=1))
        )

        # V/J as separate attention token
        vj_token = self.vj_token_proj(vj)

        # Prepare sequences for cross-attention
        gex_seq = gex_emb.unsqueeze(1)  # (B, 1, hidden)
        tcr_seq = torch.stack([tcr_a_emb, tcr_b_emb, vj_token], dim=1)  # (B, 3, hidden)

        # Cross-attention fusion
        gex_out, tcr_out = self.fusion(gex_seq, tcr_seq)

        # Concatenate all attended representations
        fused = torch.cat(
            [
                gex_out.squeeze(1),   # (B, hidden) — GEX
                tcr_out[:, 0, :],     # (B, hidden) — TCR-alpha
                tcr_out[:, 1, :],     # (B, hidden) — TCR-beta
                tcr_out[:, 2, :],     # (B, hidden) — VJ
            ],
            dim=1,
        )

        # Post-fusion + classify
        fused = self.post_fusion(fused)
        logits = self.classifier(fused)

        return logits, fused
