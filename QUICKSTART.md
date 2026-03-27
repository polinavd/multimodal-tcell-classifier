# Quick Start

## Install

```bash
git clone https://github.com/polinavd/multimodal-tcell-classifier.git
cd multimodal-tcell-classifier
pip install -r requirements.txt
```

PyTorch with GPU support (optional but recommended):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Run

```bash
python predict_report.py --input your_data.h5ad --output ./results
```

On first run, model weights (~300 MB) download automatically from [HuggingFace](https://huggingface.co/VirialyD/tcell-classifier).

## Output

```
results/
├── report.html       ← open in browser (interactive dashboard)
├── predictions.csv   ← per-cell predictions + probabilities
└── annotated.h5ad    ← your data + predicted_state column
```

### report.html

Self-contained interactive report. No internet needed. Includes:
class distributions, confidence histograms, model agreement, V/J gene usage per class, low-confidence cells, and confusion matrix (if true labels provided).

### predictions.csv

| barcode | predicted_state | confidence | model_agreement | prob_Effector | ... | prob_Treg |
|---|---|---|---|---|---|---|
| ACGTACGT-1 | Effector | 0.94 | 1.0 | 0.94 | ... | 0.01 |

### annotated.h5ad

```python
import anndata
adata = anndata.read_h5ad("tcell_results/annotated.h5ad")
print(adata.obs["predicted_state"].value_counts())
```

## Input Requirements

Your `.h5ad` file needs:

| What | Where | Notes |
|---|---|---|
| Gene expression | `adata.X` | Raw or normalized counts |
| CDR3-alpha | `adata.obs["cdr3a"]` | Also accepts `IR_VDJ_1_junction_aa` |
| CDR3-beta | `adata.obs["cdr3b"]` | Also accepts `IR_VDJ_2_junction_aa` |
| V/J genes | `adata.obs["v_alpha"]` etc. | Also accepts scirpy column names |

Missing TCR/V/J values are handled automatically (encoded as unknowns).

## Manual Weight Download

```python
from huggingface_hub import snapshot_download
snapshot_download("VirialyD/tcell-classifier", local_dir="./weights")
```

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | Add `--device cpu` or reduce dataset size |
| Slow on CPU | Expected: ~5 min/10K cells on GPU vs ~50 min on CPU |
| `KeyError` on TCR columns | Check column names — scirpy format (`IR_VDJ_*`) auto-detected |
| Download fails | Manually download weights with `snapshot_download` (see above) |
| Low confidence everywhere | Normal if tissue/disease differs from training data (colitis, PBMC, ccRCC, CRC) |
