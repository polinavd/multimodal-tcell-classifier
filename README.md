# Multimodal T-Cell Functional State Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/VirialyD/tcell-classifier)

A multimodal deep learning model for classifying T-cell functional states from paired scRNA-seq and scTCR-seq data. Integrates gene expression profiles, TCR sequence embeddings, and V/J gene usage through bidirectional cross-attention fusion.

**89.6% accuracy** · **macro F1 0.88** · **7 functional states** · **top-5 ensemble**

| Internal test | NSCLC (GSE99254) | Glioblastoma (GSE163108) | Skin SCC (GSE123813) |
|:---:|:---:|:---:|:---:|
| 89.6% | 86.4% | 67.2% | 62.6% |


> **Weights**: [VirialyD/tcell-classifier](https://huggingface.co/VirialyD/tcell-classifier) on HuggingFace Hub

---

## Quick Start

```bash
pip install -e .

# Predict from an h5ad file — weights download automatically (~500 MB, once)
python predict_report.py --input your_data.h5ad --output results/
```

This produces:
- `predictions.csv` — per-cell class labels, confidence scores, model agreement
- `annotated.h5ad` — input file with predictions added to `.obs`
- `report.html` — interactive single-file report (FastQC-style)

The report includes confidence distributions, model agreement heatmap, per-class metrics, V/J gene usage patterns, and confusion matrix (when ground-truth labels are present).

### From Python

```python
from src.inference.predict import EnsemblePredictor

predictor = EnsemblePredictor()  # auto-downloads weights
predictions = predictor.predict("your_data.h5ad")
```

---

## What the Model Does

Classifies individual T-cells into 7 functional states:

| State | Key markers | Internal F1 |
|---|---|:---:|
| **Treg** | FOXP3, IL2RA, CTLA4 | 0.94 |
| **Effector** | GZMB, PRF1, IFNG | 0.91 |
| **Proliferating** | MKI67, TOP2A, STMN1 | 0.90 |
| **Memory** | IL7R, TCF7, CCR7 | 0.89 |
| **Naive** | CCR7, SELL, LEF1 | 0.86 |
| **Exhausted** | PDCD1, LAG3, HAVCR2, TOX | 0.83 |
| **Th_effector** | CD4+ helper effector program | 0.75 |

### Three Input Modalities

1. **Gene expression** — 3,000 highly variable genes (learned dimensionality reduction via 2-layer encoder, no PCA)
2. **TCR sequences** — CDR3α and CDR3β encoded via frozen [TCR-BERT](https://github.com/wukevin/tcr-bert) (768-dim CLS embeddings per chain)
3. **V/J gene usage** — one-hot encoded TRAV/TRAJ/TRBV/TRBJ segments (161-dim vector)

### Why Multimodal?

TCR sequence alone achieves only 33.7% accuracy — a TCR defines antigen specificity, not functional state. The same TCR can belong to effector, memory, or exhausted cells depending on context. Gene expression carries the primary signal (69.9% alone), but integrating TCR embeddings adds +9.4 pp by capturing complementary clonal history information.

---

## Architecture

```
GEX (3000-dim) → Linear(512) → GELU → Linear(hidden) → ResidualBlock → 1 token
TCR-α (768-dim) + VJ context (64-dim) → Linear(hidden) → ResidualBlock → 1 token  ┐ weight
TCR-β (768-dim) + VJ context (64-dim) → Linear(hidden) → ResidualBlock → 1 token  ┘ sharing
VJ (161-dim) → Linear(hidden) → 1 token

Bidirectional Cross-Attention:
  GEX→TCR: GEX as query, [TCR-α, TCR-β, VJ] as key/value
  TCR→GEX: [TCR-α, TCR-β, VJ] as query, GEX as key/value
  4 heads, LayerNorm, residual connections

Concat(4 enriched tokens) → ResidualBlock → Linear(hidden→7) → softmax
```

### Ablation Results

| Configuration | Accuracy |
|---|:---:|
| TCR-only (TCR-BERT) | 33.7% |
| GEX-only (PCA-50) | 69.9% |
| Multimodal GEX+TCR (PCA-50, concat) | 79.3% |
| + VJ genes + PCA-200 | 84.9% |
| + Full 3000 genes (no PCA) | 88.1% (+3.2 pp) |
| + Cross-attention fusion | 88.8% (+0.7 pp) |
| + Ensemble top-5 | **89.6%** (+0.8 pp) |

### Comparison with Classical ML

On internal test data, XGBoost on concatenated features achieves 90.6% (vs 89.6% for the neural ensemble). However, on external cohorts the neural architecture generalizes substantially better:

| Cohort | XGBoost | Neural ensemble | Δ Acc |
|---|:---:|:---:|:---:|
| GSE99254 (NSCLC) | 66.8% | 75.0% | **+8.2** |
| GSE163108 (glioma) | 64.9% | 66.9% | **+2.0** |
| GSE123813 (skin SCC) | 48.5% | 51.5% | **+3.0** |

> Note: external cohort numbers above are from a simplified comparison pipeline. The full inference pipeline (used in `predict_report.py`) achieves higher absolute performance: 86.4% / 67.2% / 62.6% respectively, due to proper gene mapping and normalization.

---

## External Validation

Validated on 3 independent cohorts not used during training:

### GSE99254 — NSCLC (Guo et al., 2018)
8,950 T-cells · partially overlapping tissue context · **86.4% accuracy, macro F1 0.84**

Best: Treg (F1=0.95), Effector (F1=0.90). Weakest: Th_effector (F1=0.78). Proliferating correctly absent from predictions (zero false positives).

### GSE163108 — Glioblastoma (Xie et al., 2021)
24,804 T-cells · tissue context absent from training · **67.2% accuracy, macro F1 0.52**

Domain shift selectively degrades context-dependent states: Exhausted (F1=0.24, 73% misclassified as Effector), Th_effector (F1=0.03). Universal-signature classes remain robust: Effector (F1=0.80), Proliferating (F1=0.74), Treg (F1=0.75).

### GSE123813 — Skin Carcinoma (Yost et al., 2019)
59,122 T-cells · tissue context absent from training · **62.6% accuracy, macro F1 0.55**

Different degradation pattern vs glioma: Exhausted performs well (F1=0.74), but Naive↔Memory and Th_effector↔Naive boundaries blur. Treg remains most robust (F1=0.88). 2,015 false-positive Proliferating predictions reveal a limitation of fixed 7-class softmax.

### Key Insight

Generalization failure is **class-specific and tissue-dependent**, not uniform. Classes defined by universal programs (Treg/FOXP3, Proliferating/MKI67, Effector/GZMB) transfer well. Classes defined by context-dependent combinations (Exhausted, Th_effector) degrade when the tissue microenvironment shifts. This is a fundamental limitation of flat 7-class classification that mixes lineage, functional state, and cell-cycle axes.

---

## Training Data

**136,667 T-cells** from 4 public datasets after QC filtering:

| Dataset | Platform | Tissue | Reference |
|---|---|---|---|
| GSE144469 | 10x Genomics | Colon, blood (ICI-colitis) | Luoma et al., 2020 |
| GSE179994 | 10x Genomics | NSCLC (tumor, blood; anti-PD-1) | Liu et al., 2022 |
| GSE181061 | 10x Genomics | ccRCC (TIL) | Braun et al., 2021 |
| GSE108989 | Smart-seq2 | CRC (tumor + blood) | Zhang et al., 2018 |

Preprocessing: scanpy QC (>200 genes, <20% MT) → library-size normalization → log1p → 3,000 HVGs (seurat_v3) → Harmony batch correction.

---

## Reproduce Training

```bash
# Step 1: Preprocess raw data
python scripts/preprocess.py --data-dir data/processed

# Step 2: Compute TCR-BERT embeddings (requires GPU)
python scripts/extract_tcr_embeddings.py --data-dir data/processed

# Step 3: Train ensemble (8 models, selects top-5)
python scripts/train_ensemble.py --data-dir data/processed --save-dir results/ensemble
```

### Training Details

- **Split**: 70/15/15 stratified
- **Optimizer**: AdamW (lr=2×10⁻⁴, weight_decay=0.02)
- **Schedule**: cosine annealing with 5% linear warmup
- **Loss**: CrossEntropyLoss + balanced class weights + label smoothing (0.03)
- **Regularization**: dropout (0.25–0.35), gradient clipping (max_norm=1.0)
- **Early stopping**: macro F1 on validation, patience=12
- **Precision**: FP16 via PyTorch AMP
- **Hardware**: NVIDIA RTX 5070 (12 GB VRAM)

### Ensemble Members

| Model | hidden | heads | dropout | lr | batch | Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| m7_lr3e4 | 512 | 4 | 0.30 | 3e-4 | 256 | 88.8% |
| m2_h512_s2 | 512 | 4 | 0.30 | 2e-4 | 256 | 88.3% |
| m6_highdrop | 512 | 4 | 0.35 | 2e-4 | 256 | 88.3% |
| m4_8heads | 512 | 8 | 0.30 | 2e-4 | 256 | 88.3% |
| m1_h512 | 512 | 4 | 0.30 | 2e-4 | 256 | 88.1% |

Soft voting (logit averaging) → **89.6% accuracy**, macro F1 0.88, weighted F1 0.90.

---

## Project Structure

```
multimodal-tcell-classifier/
├── src/
│   ├── models/
│   │   ├── classifier.py        # FullGenesVJClassifier architecture
│   │   └── baselines.py         # GEX-only, TCR-only ablation models
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset classes
│   │   └── preprocessing.py     # Feature extraction, TCR-BERT, V/J encoding
│   ├── training/
│   │   ├── train.py             # Single-model training loop
│   │   └── ensemble.py          # Ensemble training and selection
│   ├── inference/
│   │   ├── predict.py           # Ensemble inference + agreement scoring
│   │   └── report.py            # Interactive HTML report generator
│   ├── hub/
│   │   └── download.py          # Auto-download weights from HF Hub / GitHub
│   └── utils/
│       ├── metrics.py           # Evaluation metrics
│       └── visualization.py     # Plotting utilities
├── scripts/                     # Reproducibility scripts
├── configs/                     # Training hyperparameter configs
├── results/figures/             # Generated figures
├── predict_report.py            # CLI: inference + HTML report
├── QUICKSTART.md
├── LICENSE                      # MIT
└── README.md
```

---

## Known Limitations

- **Domain shift**: performance drops on tissue types not seen during training, especially for Exhausted and Th_effector classes whose transcriptomic signatures are microenvironment-dependent
- **Flat classification**: the 7-class scheme mixes lineage (Treg, Th_effector), functional state (Naive→Effector→Exhausted), and cell cycle (Proliferating) — a multi-axis formulation would improve transferability
- **Exhausted heterogeneity**: public annotations for "Exhausted" vary across datasets (checkpoint-like CD8, cytotoxic exhausted, CD4/Tfh-like, cycling exhausted), limiting what any model can learn from this label
- **TCR-α coverage**: 10x Genomics has incomplete α-chain recovery; missing chains are zero-padded
- **Softmax constraint**: the model always distributes probability across all 7 classes, producing false positives for classes absent in the target cohort (e.g., Proliferating in Yost data)

---

## Citation

```bibtex
@mastersthesis{shirokikh2026multimodal,
  author  = {Shirokikh, Polina G.},
  title   = {Multimodal Deep Learning Model for Classification of T-Cell
             Functional States Based on Single-Cell Sequencing Data Integration},
  school  = {ITMO University},
  year    = {2026},
  address = {Saint Petersburg, Russia},
  url     = {https://github.com/polinavd/multimodal-tcell-classifier}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Polina Shirokikh** · ITMO University, 2026

Scientific advisor: Mikhail P. Raiko (ITMO) · Consultant: Gaukhar M. Yusubalieva (FMBA)
