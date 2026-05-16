"""
Data loading, preprocessing, and feature extraction utilities.

Pipeline:
  1. Load raw scRNA-seq data (h5ad) from multiple GEO datasets
  2. Quality control and filtering
  3. Normalization and HVG selection (3,000 genes)
  4. Batch correction with Harmony
  5. Extract CDR3 sequences and V/J gene annotations
  6. Compute TCR-BERT embeddings
  7. One-hot encode V/J genes
  8. Split into train/val/test
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path

import anndata
import torch
from torch.amp import autocast
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


def load_splits(data_dir: str | Path) -> dict:
    """Load preprocessed ML data with all features."""
    data_dir = Path(data_dir)
    data = np.load(data_dir / "ml_data_full_genes.npz", allow_pickle=True)

    with open(data_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open(data_dir / "vj_encoder.pkl", "rb") as f:
        vj_encoder = pickle.load(f)

    return data, label_encoder, vj_encoder


def compute_tcr_bert_embeddings(
    sequences: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 32,
) -> np.ndarray:
    """
    Compute TCR-BERT [CLS] embeddings for CDR3 sequences.

    Args:
        sequences: List of CDR3 amino acid strings
        tokenizer: TCR-BERT tokenizer
        model: TCR-BERT model (already on device)
        device: Torch device
        batch_size: Batch size for encoding
        max_length: Max token length (CDR3 is typically 12-20 AA)

    Returns:
        embeddings: (N, 768) array of CLS token embeddings
    """
    model.eval()
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]

        # TCR-BERT expects space-separated amino acids
        spaced = []
        for seq in batch:
            if not seq or str(seq) == "nan":
                spaced.append("")
            else:
                spaced.append(" ".join(list(str(seq))))

        encoded = tokenizer(
            spaced,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**encoded)

        cls_emb = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        all_embeddings.append(cls_emb)

        if (i // batch_size) % 20 == 0:
            print(f"    {min(i + batch_size, len(sequences))}/{len(sequences)}")

    return np.concatenate(all_embeddings, axis=0)


def encode_vj_genes(
    adata: anndata.AnnData,
    encoder: OneHotEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, OneHotEncoder]:
    """
    Extract and one-hot encode V/J gene annotations.

    Looks for columns: v_alpha, j_alpha, v_beta, j_beta
    or AIRR-format alternatives: IR_VDJ_1_v_call, etc.

    Args:
        adata: AnnData object with obs containing V/J columns
        encoder: Pre-fitted OneHotEncoder (None to fit new)

    Returns:
        vj_encoded: (N, vj_dim) one-hot encoded array
        vj_raw: (N, 4) string array of raw gene names
        encoder: The fitted OneHotEncoder
    """
    n_cells = adata.n_obs

    # Column mapping with AIRR fallbacks
    col_map = {
        "v_alpha": "IR_VDJ_1_v_call",
        "j_alpha": "IR_VDJ_1_j_call",
        "v_beta": "IR_VDJ_2_v_call",
        "j_beta": "IR_VDJ_2_j_call",
    }

    vj_data = []
    for i in range(n_cells):
        row = []
        for primary, fallback in col_map.items():
            if primary in adata.obs.columns:
                val = adata.obs.iloc[i][primary]
            elif fallback in adata.obs.columns:
                val = adata.obs.iloc[i][fallback]
            else:
                val = None

            row.append(str(val) if pd.notna(val) else "UNKNOWN")
        vj_data.append(row)

    vj_raw = np.array(vj_data)

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        vj_encoded = encoder.fit_transform(vj_raw).astype(np.float32)
    else:
        vj_encoded = encoder.transform(vj_raw).astype(np.float32)

    return vj_encoded, vj_raw, encoder


def prepare_gene_expression(
    adata: anndata.AnnData,
    gene_list_path: str | Path | None = None,
    gene_scaling_path: str | Path | None = None,
    clip_value: float = 10.0,
) -> np.ndarray:
    """
    Extract gene expression and optionally align it to the training gene space.

    If gene_list_path is provided, adata.X is reordered to match the exact
    training HVG order and missing genes are filled with zeros. If
    gene_scaling_path is provided, training mean/std scaling is applied.
    """
    from scipy import sparse

    gene_list_path = Path(gene_list_path) if gene_list_path else None
    gene_scaling_path = Path(gene_scaling_path) if gene_scaling_path else None

    if gene_list_path and gene_list_path.exists():
        expected_genes = [
            line.strip()
            for line in gene_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gene_to_idx: dict[str, int] = {}
        for idx, gene in enumerate(map(str, adata.var_names)):
            gene_to_idx.setdefault(gene, idx)

        out_cols = []
        in_cols = []
        missing = []
        for out_idx, gene in enumerate(expected_genes):
            in_idx = gene_to_idx.get(gene)
            if in_idx is None:
                missing.append(gene)
            else:
                out_cols.append(out_idx)
                in_cols.append(in_idx)

        gex = np.zeros((adata.n_obs, len(expected_genes)), dtype=np.float32)
        if in_cols:
            subset = adata.X[:, in_cols]
            if sparse.issparse(subset):
                subset = subset.toarray()
            gex[:, out_cols] = np.asarray(subset, dtype=np.float32)

        if missing:
            print(
                f"    Gene alignment: {len(expected_genes) - len(missing)}/"
                f"{len(expected_genes)} genes found; {len(missing)} filled with zero"
            )
        else:
            print(f"    Gene alignment: all {len(expected_genes)} genes found")
    else:
        if sparse.issparse(adata.X):
            gex = np.asarray(adata.X.todense(), dtype=np.float32)
        else:
            gex = np.asarray(adata.X, dtype=np.float32)

    if gene_scaling_path and gene_scaling_path.exists():
        scaling = np.load(gene_scaling_path)
        means = scaling["means"].astype(np.float32)
        stds = scaling["stds"].astype(np.float32)
        if gex.shape[1] != means.shape[0]:
            raise ValueError(
                f"Gene scaling expects {means.shape[0]} genes, got {gex.shape[1]}"
            )
        stds = np.where(stds == 0, 1.0, stds)
        gex = (gex - means) / stds
        gex = np.clip(gex, -clip_value, clip_value).astype(np.float32)
        print(f"    Gene scaling: applied z-score and clipped to +/-{clip_value:g}")

    return gex.astype(np.float32, copy=False)


def prepare_inference_features(
    adata: anndata.AnnData,
    bert_model_name: str,
    vj_encoder: OneHotEncoder,
    device: torch.device,
    gene_list_path: str | Path | None = None,
    gene_scaling_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare all features for inference from a new h5ad file.

    Returns:
        gex: (N, n_genes) gene expression
        tcr_a_emb: (N, 768) TCR-BERT embeddings for alpha
        tcr_b_emb: (N, 768) TCR-BERT embeddings for beta
        vj_encoded: (N, vj_dim) one-hot V/J
        vj_raw: (N, 4) raw V/J gene names
    """
    from transformers import AutoTokenizer, AutoModel

    # Gene expression
    print("  Extracting gene expression...")
    gex = prepare_gene_expression(
        adata,
        gene_list_path=gene_list_path,
        gene_scaling_path=gene_scaling_path,
    )
    print(f"    {gex.shape[0]} cells, {gex.shape[1]} genes")

    # TCR-BERT embeddings
    print("  Computing TCR-BERT embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert = AutoModel.from_pretrained(bert_model_name).to(device)

    cdr3a = adata.obs.get(
        "cdr3a", adata.obs.get("IR_VDJ_1_junction_aa", pd.Series([""] * adata.n_obs))
    ).values
    cdr3b = adata.obs.get(
        "cdr3b", adata.obs.get("IR_VDJ_2_junction_aa", pd.Series([""] * adata.n_obs))
    ).values

    tcr_a_emb = compute_tcr_bert_embeddings(list(cdr3a), tokenizer, bert, device)
    tcr_b_emb = compute_tcr_bert_embeddings(list(cdr3b), tokenizer, bert, device)

    del bert
    torch.cuda.empty_cache()

    # V/J genes
    print("  Encoding V/J genes...")
    vj_encoded, vj_raw, _ = encode_vj_genes(adata, encoder=vj_encoder)

    return gex, tcr_a_emb, tcr_b_emb, vj_encoded, vj_raw
