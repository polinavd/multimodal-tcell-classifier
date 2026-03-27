# Multimodal T-Cell Functional State Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/VirialyD/tcell-classifier)

A multimodal deep learning model for predicting T-cell functional states by integrating single-cell gene expression, TCR sequences, and V/J gene usage. Achieves **89.6% accuracy** (macro F1 0.88) across 7 functional states using a top-5 ensemble with cross-attention fusion.

## Overview

T-cell functional state classification is critical for understanding immune responses in cancer, autoimmunity, and infection. This project integrates three complementary data modalities:

- **Gene expression** — 3,000 highly variable genes from scRNA-seq (no PCA; learned dimensionality reduction)
- **TCR sequences** — CDR3-alpha and CDR3-beta encoded via [TCR-BERT](https://github.com/wukevin/tcr-bert) (768-dim embeddings)
- **V/J gene usage** — One-hot encoded TRAV/TRAJ/TRBV/TRBJ segments (161-dim)

The model uses **bidirectional cross-attention** to learn interactions between modalities and classifies cells into 7 functional states: Effector, Exhausted, Memory, Naive, Proliferating, Th_effector, and Treg.

## Key Results

### Ablation Study

| Model Configuration | Test Accuracy | Macro F1 |
|---|---|---|
| TCR-only (BERT embeddings) | 33.7% | — |
| GEX-only (PCA-50) | 69.9% | — |
| Multimodal (PCA-50, concat) | 79.3% | — |
| End-to-end BERT fine-tuning | 77.4% | — |
| Hybrid + VJ + PCA-200 | 84.9% | — |
| **Ensemble + VJ + 3000 genes** | **89.6%** | **0.88** |

Using all 3,000 HVGs instead of PCA-reduced representations provides a **+4.7%** improvement — the learned GEX encoder outperforms fixed PCA for this task.

### Per-Class Performance (Top-5 Ensemble)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Effector | 0.91 | 0.92 | 0.91 | 6,685 |
| Exhausted | 0.84 | 0.82 | 0.83 | 2,245 |
| Memory | 0.89 | 0.88 | 0.89 | 4,979 |
| Naive | 0.87 | 0.85 | 0.86 | 2,441 |
| Proliferating | 0.92 | 0.89 | 0.90 | 764 |
| Th_effector | 0.76 | 0.74 | 0.75 | 393 |
| Treg | 0.93 | 0.94 | 0.94 | 2,329 |

## Architecture

```
                    FullGenesVJClassifier

  GEX (3000 genes)        TCR-BERT (768-dim)         V/J (161-dim)
       |                   |           |                   |
  [Linear 512]        [CDR3-α]    [CDR3-β]          [Linear 64]
  [GELU + Drop]            |           |             (VJ context)
  [Linear hidden]          |           |                   |
  [ResidualBlock]     [cat with VJ context]                |
       |              [Linear hidden]                [Linear hidden]
       |              [ResidualBlock]                 (VJ token)
       |                   |           |                   |
       v                   v           v                   v
  ┌─────────────────────────────────────────────────────────────┐
  │            Bidirectional Cross-Attention Fusion              │
  │                                                             │
  │  GEX (1 token) ←→ TCR-α + TCR-β + VJ (3 tokens)           │
  │  4 attention heads, LayerNorm, residual connections         │
  └─────────────────────────────────────────────────────────────┘
       |                   |           |                   |
       └───────────────────┴───────────┴───────────────────┘
                           |
                    [Concat: 4 × hidden]
                    [ResidualBlock]
                    [Linear hidden → 7]
                           |
                    Functional State
```

The ensemble uses the **top-5 models** (by validation F1) with diverse hyperparameters (hidden dim 512, 4/8 attention heads, dropout 0.25–0.35, varied learning rates and seeds), averaged at inference time.

## Data

Trained on **136,667 T-cells** (after QC filtering) from 4 public scRNA-seq datasets:

| Dataset | Source | Cells* | Tissue |
|---|---|---|---|
| GSE144469 | 10x Genomics | ~60,000 | Colitis (colon) |
| GSE179994 | 10x Genomics | ~77,000 | PBMC (exhaustion study) |
| GSE181061 | 10x Genomics | ~31,000 | ccRCC (tumor-infiltrating) |
| GSE108989 | Smart-seq2 | ~12,000 | CRC (tumor + blood) |

*Cell counts are pre-QC; 136,667 cells remain after quality control filtering.

Preprocessing: quality control → normalization (scanpy) → 3,000 HVGs → Harmony batch correction → CDR3/V/J extraction via scirpy.

## Installation

```bash
git clone https://github.com/polinavd/multimodal-tcell-classifier.git
cd multimodal-tcell-classifier
pip install -r requirements.txt
```

PyTorch with GPU support (optional but ~10x faster):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Inference on New Data

```bash
python predict_report.py --input your_data.h5ad --output ./results
```

Model weights (~300 MB) are downloaded automatically from HuggingFace on first run.

The report is a self-contained interactive HTML file (FastQC-style) with:
confidence distributions, model agreement, per-class metrics, V/J gene usage patterns, confusion matrix, and exportable predictions.

### Full Pipeline (Reproduce from Scratch)

```bash
# Step 1: Preprocess raw data
python scripts/preprocess.py --data-dir data/processed

# Step 2: Compute TCR-BERT embeddings (requires GPU)
python scripts/extract_tcr_embeddings.py --data-dir data/processed

# Step 3: Train ensemble
python scripts/train_ensemble.py --data-dir data/processed --save-dir results/ensemble
```

### Manual Weight Download

```python
from huggingface_hub import snapshot_download
snapshot_download("VirialyD/tcell-classifier", local_dir="./weights")
```

## Project Structure

```
multimodal-tcell-classifier/
├── src/
│   ├── download.py              # Auto-download weights from HuggingFace Hub
│   ├── models/
│   │   ├── classifier.py      # FullGenesVJClassifier (main architecture)
│   │   └── baselines.py       # GEX-only, TCR-only ablation baselines
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset classes
│   │   └── preprocessing.py    # Feature extraction, TCR-BERT, V/J encoding
│   ├── training/
│   │   ├── train.py            # Single-model training loop (AMP, cosine LR)
│   │   └── ensemble.py         # Ensemble training and evaluation
│   ├── inference/
│   │   ├── predict.py          # Ensemble inference with agreement scoring
│   │   └── report.py           # Interactive HTML report generator
│   └── utils/
│       ├── metrics.py          # Evaluation metrics
│       └── visualization.py    # Plotting utilities
├── scripts/                    # Reproducibility scripts (preprocess, train)
├── configs/                    # Training configurations
├── results/figures/            # Generated figures
├── predict_report.py            # Main inference + HTML report script
├── QUICKSTART.md               # 2-command usage guide
├── LICENSE                     # MIT
└── README.md
```

## Technical Details

**Training:**
- Optimizer: AdamW (lr=2e-4, weight_decay=0.02)
- Schedule: Cosine annealing with 5% linear warmup
- Loss: CrossEntropyLoss with balanced class weights + label smoothing (0.03)
- Mixed precision (FP16) with gradient clipping (max_norm=1.0)
- Early stopping on validation macro F1 (patience=12)
- Hardware: NVIDIA RTX 5070 (8 GB VRAM)

**Ensemble diversity** achieved through variation in:
- Hidden dimensions (384, 512)
- Attention heads (4, 8)
- Dropout rates (0.25, 0.30, 0.35)
- Learning rates (2e-4, 3e-4)
- Batch sizes (256, 512)
- Random seeds

## Citation

```bibtex
@software{shirokikh2025multimodal,
  author = {Shirokikh, Polina},
  title = {Multimodal T-Cell Functional State Classifier},
  year = {2025},
  url = {https://github.com/polinavd/multimodal-tcell-classifier}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Polina Shirokikh**
Master's Thesis, 2026
