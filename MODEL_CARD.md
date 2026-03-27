---
license: gpl-3.0
library_name: pytorch
tags:
  - biology
  - single-cell
  - T-cell
  - TCR
  - immunology
  - scRNA-seq
  - multimodal
  - cross-attention
  - ensemble
datasets:
  - custom
metrics:
  - accuracy
  - f1
pipeline_tag: tabular-classification
language:
  - en
---

# Multimodal T-Cell Functional State Classifier

A multimodal deep learning ensemble for predicting T-cell functional states from scRNA-seq data. Integrates gene expression (3,000 HVGs), TCR sequences (via [TCR-BERT](https://github.com/wukevin/tcr-bert)), and V/J gene usage through bidirectional cross-attention fusion.

**89.6% accuracy** | **macro F1 0.88** | **7 functional states** | **top-5 ensemble**

## Model Description

This repository contains the weights for a top-5 ensemble of `FullGenesVJClassifier` models. Each model takes three input modalities:

- **Gene expression**: 3,000 highly variable genes (learned dimensionality reduction, no PCA)
- **TCR sequences**: CDR3-alpha and CDR3-beta encoded via TCR-BERT (768-dim CLS embeddings)
- **V/J gene usage**: one-hot encoded TRAV/TRAJ/TRBV/TRBJ segments (161-dim)

Cross-attention fusion allows each modality to attend to the others before classification into 7 functional states: Effector, Exhausted, Memory, Naive, Proliferating, Th_effector, Treg.

### Architecture

```
GEX (3000) → [Linear 512 → GELU → Linear hidden] + ResidualBlock → (hidden,)
TCR-α/β (768) + VJ context (64) → [Linear hidden] + ResidualBlock → (hidden,)
VJ (161) → [Linear hidden] → (hidden,)

Cross-Attention Fusion:
  GEX (1 token) ↔ TCR-α + TCR-β + VJ (3 tokens)
  4 attention heads, LayerNorm, residual connections

→ Concat (4 × hidden) → ResidualBlock → Linear → 7 classes
```

### Ensemble Diversity

| Model | Hidden dim | Heads | Dropout | Acc |
|---|---|---|---|---|
| m7_lr3e4 | 512 | 4 | 0.30 | 88.8% |
| m2_h512_s2 | 512 | 4 | 0.30 | 88.3% |
| m6_highdrop | 512 | 4 | 0.35 | 88.3% |
| m4_8heads | 512 | 8 | 0.30 | 88.3% |
| m1_h512 | 512 | 4 | 0.30 | 88.1% |

Ensemble averaging of these 5 models yields **89.6% accuracy** (macro F1 0.88).

## Intended Use

Classification of T-cell functional states from paired scRNA-seq + TCR-seq data. Designed for research use in immunology, immuno-oncology, and single-cell analysis pipelines.

**Not intended** for clinical decision-making or diagnostic use.

## Training Data

~290,000 T-cells from 4 public scRNA-seq datasets:

| Dataset | Platform | Cells | Tissue |
|---|---|---|---|
| GSE144469 | 10x Genomics | ~60,000 | Colitis (colon) |
| GSE179994 | 10x Genomics | ~77,000 | PBMC (exhaustion study) |
| GSE181061 | 10x Genomics | ~31,000 | ccRCC (tumor-infiltrating) |
| GSE108989 | Smart-seq2 | ~12,000 | CRC (tumor + blood) |

Preprocessing: QC → normalization (scanpy) → 3,000 HVGs → Harmony batch correction → CDR3/V/J extraction via scirpy.

## Evaluation

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Effector | 0.91 | 0.92 | 0.91 | 6,685 |
| Exhausted | 0.84 | 0.82 | 0.83 | 2,245 |
| Memory | 0.89 | 0.88 | 0.89 | 4,979 |
| Naive | 0.87 | 0.85 | 0.86 | 2,441 |
| Proliferating | 0.92 | 0.89 | 0.90 | 764 |
| Th_effector | 0.76 | 0.74 | 0.75 | 393 |
| Treg | 0.93 | 0.94 | 0.94 | 2,329 |

### Ablation Study

| Configuration | Accuracy |
|---|---|
| TCR-only (BERT embeddings) | 33.7% |
| GEX-only (PCA-50) | 69.9% |
| Multimodal (PCA-50, concat) | 79.3% |
| End-to-end BERT fine-tuning | 77.4% |
| Hybrid + VJ + PCA-200 | 84.9% |
| **Ensemble + VJ + 3000 genes** | **89.6%** |

## How to Use

### Quick Start (CLI)

```bash
pip install tcell-classifier
tcell-predict your_data.h5ad
```

Model weights (~300 MB) download automatically on first run.

```bash
tcell-predict data.h5ad -o results/                # custom output dir
tcell-predict data.h5ad --true-labels cell_type     # evaluate vs ground truth
tcell-predict data.h5ad --device cpu                # force CPU
```

Output: interactive HTML report, predictions.csv, annotated .h5ad.

### Python API

```python
from src.hub import ensure_weights
from src.inference import load_ensemble, ensemble_predict
from src.data import InferenceDataset, prepare_inference_features

model_dir = ensure_weights()  # auto-downloads from this repo
models = load_ensemble(model_dir, device)
dataset = InferenceDataset(gex, tcr_a_emb, tcr_b_emb, vj_encoded)
predictions, probabilities, agreement = ensemble_predict(models, dataset, device)
```

### Manual Download

```python
from huggingface_hub import snapshot_download
snapshot_download("VirialyD/tcell-classifier", local_dir="./weights")
```

## Files

| File | Description |
|---|---|
| `m1_h512.pt` | Model 1: hidden=512, heads=4, dropout=0.30 |
| `m2_h512_s2.pt` | Model 2: hidden=512, heads=4, dropout=0.30, seed=2 |
| `m4_8heads.pt` | Model 4: hidden=512, heads=8, dropout=0.30 |
| `m6_highdrop.pt` | Model 6: hidden=512, heads=4, dropout=0.35 |
| `m7_lr3e4.pt` | Model 7: hidden=512, heads=4, dropout=0.30, lr=3e-4 |
| `results.json` | Individual model metrics and ensemble config |
| `label_encoder.pkl` | sklearn LabelEncoder for 7 functional states |
| `vj_encoder.pkl` | V/J gene one-hot encoder (161-dim) |

## Technical Details

- **Optimizer**: AdamW (lr=2e-4, weight_decay=0.02)
- **Schedule**: Cosine annealing with 5% linear warmup
- **Loss**: CrossEntropyLoss with balanced class weights + label smoothing (0.03)
- **Mixed precision**: FP16 with gradient clipping (max_norm=1.0)
- **Early stopping**: on validation macro F1 (patience=12)
- **Hardware**: NVIDIA RTX 5070 (8 GB VRAM)

## Limitations

- Trained on human T-cells only; not validated on other species or non-T immune cells.
- Requires paired scRNA-seq + TCR-seq data (CDR3 alpha/beta + V/J genes).
- Gene expression input must be from the same 3,000 HVG feature space. The preprocessing pipeline handles this, but heavily divergent protocols may reduce accuracy.
- Th_effector class has the lowest performance (F1 0.75), likely due to small training sample (393 cells).

## Citation

```bibtex
@software{levchenko2026multimodal,
  author = {Shirokikh, Polina},
  title = {Multimodal T-Cell Functional State Classifier},
  year = {2026},
  url = {https://github.com/polinavd/multimodal-tcell-classifier}
}
```

## License

GPL-3.0
