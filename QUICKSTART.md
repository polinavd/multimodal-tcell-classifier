# Quick Start

## Install

```bash
pip install tcell-classifier
```

Or from source:

```bash
git clone https://github.com/polinavd/multimodal-tcell-classifier.git
cd multimodal-tcell-classifier
pip install .
```

PyTorch with GPU support (optional but recommended):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Run

```bash
tcell-predict your_data.h5ad
```

That's it. On first run, model weights (~500 MB) download automatically to `~/.cache/tcell-classifier/`.

### Options

```bash
tcell-predict your_data.h5ad -o results/              # custom output directory
tcell-predict your_data.h5ad --true-labels cell_type   # evaluate against ground truth
tcell-predict your_data.h5ad --device cpu               # force CPU
```

## Output

```
tcell_results/
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

## Python API

```python
from src.hub import ensure_weights
from src.inference import load_ensemble, ensemble_predict
from src.data import InferenceDataset, prepare_inference_features
import anndata, pickle, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = ensure_weights()  # auto-downloads if needed

# Load
adata = anndata.read_h5ad("your_data.h5ad")
with open(model_dir / "vj_encoder.pkl", "rb") as f:
    vj_enc = pickle.load(f)

# Features + predict
gex, tcr_a, tcr_b, vj, vj_raw = prepare_inference_features(
    adata, "wukevin/tcr-bert", vj_enc, device
)
models = load_ensemble(model_dir, device)
preds, probs, agreement = ensemble_predict(
    models, InferenceDataset(gex, tcr_a, tcr_b, vj), device
)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | Use `--device cpu` or reduce dataset size |
| Slow on CPU | Expected: ~5 min/10K cells on GPU vs ~50 min on CPU |
| `KeyError` on TCR columns | Check column names — scirpy format (`IR_VDJ_*`) auto-detected |
| Download fails | Set `TCELL_CACHE_DIR=/path/to/dir` or manually place weights there |
| Low confidence everywhere | Normal if tissue/disease differs from training data (colitis, PBMC, ccRCC, CRC) |
