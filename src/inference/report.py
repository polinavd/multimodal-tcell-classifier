"""
Interactive HTML report generator (FastQC-style).

Generates a self-contained HTML report with:
  1. Summary statistics
  2. Dataset information
  3. Predicted class distribution
  4. Confidence and model agreement histograms
  5. Confidence breakdown by class
  6. Confusion matrix (if true labels provided)
  7. Per-class precision/recall/F1 (if true labels provided)
  8. Top V/J gene usage per class
  9. Low-confidence predictions table
 10. Export (CSV link + JSON summary)
"""

import json
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Class colors for 7 functional states
CLASS_COLORS = [
    "#2563eb", "#dc2626", "#059669", "#d97706",
    "#7c3aed", "#db2777", "#0891b2",
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Source+Sans+3:wght@300;400;600;700&display=swap');
:root{--bg:#0a0e17;--sf:#111827;--s2:#1a2234;--bd:#1e3048;--tx:#e2e8f0;--td:#8896ab;
--ac:#3b82f6;--a2:#06b6d4;--ok:#10b981;--wr:#f59e0b;--er:#ef4444;
--gr:linear-gradient(135deg,#3b82f6 0%,#06b6d4 100%)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Source Sans 3',sans-serif;background:var(--bg);color:var(--tx);line-height:1.6}
.hd{background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);border-bottom:1px solid var(--bd);padding:3rem 2rem;text-align:center;position:relative;overflow:hidden}
.hd::before{content:'';position:absolute;inset:0;background:radial-gradient(circle at 50% 0%,rgba(59,130,246,.15) 0%,transparent 60%);pointer-events:none}
.hd h1{font-size:2.2rem;font-weight:700;letter-spacing:-.02em;background:var(--gr);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem;position:relative}
.hd p{color:var(--td);font-size:.95rem;position:relative}
.ct{max-width:1200px;margin:0 auto;padding:2rem}
.nv{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:.7rem 1.2rem;margin-bottom:1.5rem;display:flex;flex-wrap:wrap;gap:.3rem;align-items:center;position:sticky;top:0;z-index:100;backdrop-filter:blur(12px)}
.nv a{color:var(--td);text-decoration:none;font-size:.8rem;padding:.3rem .65rem;border-radius:6px;transition:all .2s}
.nv a:hover{color:var(--tx);background:var(--s2)}.nv .nt{font-weight:600;color:var(--ac);margin-right:auto;font-size:.85rem}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.8rem;margin-bottom:1.5rem}
.sc{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:1.3rem;text-align:center;transition:transform .2s,border-color .2s}
.sc:hover{transform:translateY(-2px);border-color:var(--ac)}
.sv{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:600;background:var(--gr);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sl{color:var(--td);font-size:.75rem;margin-top:.2rem;text-transform:uppercase;letter-spacing:.05em}
.se{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:1.8rem;margin-bottom:1.2rem}
.st{font-size:1.2rem;font-weight:600;margin-bottom:1.2rem;display:flex;align-items:center;gap:.5rem}
.st .ic{width:25px;height:25px;background:var(--gr);border-radius:6px;display:inline-flex;align-items:center;justify-content:center;font-size:.78rem;color:#fff;flex-shrink:0}
.bc{display:flex;flex-direction:column;gap:.45rem}
.br{display:flex;align-items:center;gap:.8rem}
.bl{font-family:'JetBrains Mono',monospace;font-size:.8rem;min-width:110px;text-align:right}
.bk{flex:1;height:28px;background:var(--s2);border-radius:6px;overflow:hidden}
.bf{height:100%;border-radius:6px;display:flex;align-items:center;transition:width 1s}
.bv{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5);margin-left:auto;padding-right:8px;white-space:nowrap}
.hi{display:flex;align-items:flex-end;gap:2px;height:130px;padding-top:.8rem}
.hb{flex:1;border-radius:3px 3px 0 0;position:relative;cursor:pointer;min-width:6px;transition:height .5s}
.hb:hover{opacity:.8}
.hb:hover::after{content:attr(data-t);position:absolute;bottom:100%;left:50%;transform:translateX(-50%);background:var(--s2);border:1px solid var(--bd);padding:3px 6px;border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:.65rem;white-space:nowrap;z-index:10}
.tc{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
@media(max-width:768px){.tc{grid-template-columns:1fr}}
.cmc{overflow-x:auto}.cmt{border-collapse:collapse;margin:0 auto}
.cmt th,.cmt td{width:62px;height:36px;text-align:center;font-family:'JetBrains Mono',monospace;font-size:.7rem;border:1px solid var(--bd)}
.cmt th{background:var(--s2);color:var(--td);font-weight:600;font-size:.65rem}
.dt{width:100%;border-collapse:collapse}
.dt th{text-align:left;padding:.6rem .8rem;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--td);border-bottom:2px solid var(--bd)}
.dt td{padding:.5rem .8rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;border-bottom:1px solid var(--bd)}
.dt tr:hover{background:var(--s2)}
.tg{display:inline-block;padding:2px 8px;border-radius:16px;font-size:.7rem;font-weight:600}
.th{background:rgba(16,185,129,.15);color:var(--ok)}.tm{background:rgba(245,158,11,.15);color:var(--wr)}.tl{background:rgba(239,68,68,.15);color:var(--er)}
.mg{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:.8rem}
.mc{background:var(--s2);border-radius:8px;padding:1rem;display:flex;align-items:center;gap:.7rem}
.md{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.mn{font-weight:600;font-size:.88rem}.mv{font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--td)}
.ig{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:.6rem}
.ii{background:var(--s2);border-radius:8px;padding:.65rem .85rem}
.il{font-size:.68rem;color:var(--td);text-transform:uppercase;letter-spacing:.05em}
.iv{font-family:'JetBrains Mono',monospace;font-size:.82rem;margin-top:.12rem}
.vjb{margin-bottom:1.2rem}.vjt{font-weight:600;font-size:.88rem;margin-bottom:.35rem;display:flex;align-items:center;gap:.4rem}
.vjg{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem}
@media(max-width:900px){.vjg{grid-template-columns:repeat(2,1fr)}}
.vjc{background:var(--s2);border-radius:8px;padding:.6rem}
.vjn{font-size:.65rem;color:var(--a2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.35rem}
.vjr{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:.1rem 0}
.vjr span:last-child{color:var(--td)}
.cct{width:100%;border-collapse:collapse}
.cct th{text-align:left;padding:.5rem .65rem;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--td);border-bottom:2px solid var(--bd)}
.cct td{padding:.5rem .65rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;border-bottom:1px solid var(--bd)}
.cct tr:hover{background:var(--s2)}
.cb{width:100px;height:6px;background:var(--s2);border-radius:3px;display:inline-block;position:relative;vertical-align:middle}
.cff{height:100%;border-radius:3px;position:absolute;top:0;left:0}
.btn{display:inline-flex;align-items:center;gap:.4rem;padding:.6rem 1.3rem;border-radius:8px;font-size:.85rem;font-weight:600;cursor:pointer;border:none;transition:all .2s;text-decoration:none;color:#fff}
.bp{background:var(--gr)}.bp:hover{opacity:.9;transform:translateY(-1px)}
.bo{background:transparent;border:1px solid var(--bd);color:var(--tx)}.bo:hover{border-color:var(--ac);background:var(--s2)}
.ft{text-align:center;padding:2rem;color:var(--td);font-size:.75rem;border-top:1px solid var(--bd);margin-top:2rem}
"""


def generate_report(
    barcodes: list[str],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    agreement: np.ndarray,
    class_names: list[str],
    true_labels: np.ndarray | None = None,
    n_cells: int = 0,
    n_genes: int = 0,
    input_file: str = "",
    vj_raw: np.ndarray | None = None,
    obs_df=None,
    output_path: str = "report.html",
    csv_filename: str = "predictions.csv",
) -> str:
    """
    Generate a self-contained interactive HTML report.

    Args:
        barcodes: Cell barcode identifiers
        predictions: (N,) predicted class indices
        probabilities: (N, n_classes) softmax probabilities
        agreement: (N,) model agreement fractions
        class_names: List of class name strings
        true_labels: (N,) true class indices (optional)
        n_cells: Number of cells
        n_genes: Number of genes
        input_file: Input filename
        vj_raw: (N, 4) raw V/J gene names (optional)
        obs_df: DataFrame with obs metadata (optional)
        output_path: Output HTML file path
        csv_filename: CSV filename for download link

    Returns:
        output_path: Path to generated report
    """
    conf = probabilities.max(axis=1)
    pred_labels = np.array([class_names[p] for p in predictions])
    unique, counts = np.unique(predictions, return_counts=True)
    class_dist = {class_names[u]: int(c) for u, c in zip(unique, counts)}

    conf_stats = {
        "mean": float(np.mean(conf)),
        "median": float(np.median(conf)),
        "std": float(np.std(conf)),
        "below_50": int(np.sum(conf < 0.5)),
        "above_90": int(np.sum(conf > 0.9)),
    }

    # Confidence by class
    conf_by_class = {}
    for ci, cn in enumerate(class_names):
        mask = predictions == ci
        if mask.sum() > 0:
            cc = conf[mask]
            conf_by_class[cn] = {
                "mean": float(np.mean(cc)),
                "median": float(np.median(cc)),
                "q25": float(np.percentile(cc, 25)),
                "q75": float(np.percentile(cc, 75)),
                "min": float(np.min(cc)),
                "n": int(mask.sum()),
                "below_50": int(np.sum(cc < 0.5)),
            }

    # V/J gene usage
    vj_usage = {}
    if vj_raw is not None:
        vj_categories = ["TRAV", "TRAJ", "TRBV", "TRBJ"]
        for ci, cn in enumerate(class_names):
            mask = predictions == ci
            if mask.sum() == 0:
                continue
            cv = vj_raw[mask]
            vj_usage[cn] = {}
            for vi, vc in enumerate(vj_categories):
                ctr = Counter(cv[:, vi])
                ctr.pop("UNKNOWN", None)
                ctr.pop("nan", None)
                total = sum(ctr.values())
                vj_usage[cn][vc] = [
                    (gene, count / total * 100 if total > 0 else 0)
                    for gene, count in ctr.most_common(5)
                ]

    # Low confidence cells
    low_conf_idx = np.argsort(conf)[:20]
    low_conf_data = []
    for idx in low_conf_idx:
        top2 = np.argsort(probabilities[idx])[-2:][::-1]
        low_conf_data.append({
            "barcode": barcodes[idx],
            "predicted": class_names[predictions[idx]],
            "confidence": float(conf[idx]),
            "second_class": class_names[top2[1]],
            "second_prob": float(probabilities[idx, top2[1]]),
            "agreement": float(agreement[idx]),
        })

    # True labels metrics
    has_truth = true_labels is not None
    if has_truth:
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average="macro")
        weighted_f1 = f1_score(true_labels, predictions, average="weighted")
        report_dict = classification_report(
            true_labels, predictions, target_names=class_names, output_dict=True
        )
        cm = confusion_matrix(true_labels, predictions)
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Color map
    color_map = {n: CLASS_COLORS[i % len(CLASS_COLORS)] for i, n in enumerate(class_names)}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Dataset metadata
    dataset_info = {"file": Path(input_file).name}
    if obs_df is not None:
        for col in ["dataset", "sample", "patient", "tissue", "source", "batch", "study"]:
            if col in obs_df.columns:
                dataset_info[col] = list(obs_df[col].unique()[:15])

    # Build HTML
    h = (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        f'<title>T-Cell Prediction Report</title><style>{CSS}</style></head><body>\n'
    )

    # Header
    h += (
        f'<div class="hd"><h1>T-Cell Functional State Prediction Report</h1>'
        f'<p>Generated {timestamp} &middot; {n_cells:,} cells &middot; '
        f'{n_genes:,} genes &middot; {len(class_names)} classes</p></div>\n'
        f'<div class="ct">\n'
    )

    # Navigation
    h += (
        '<div class="nv"><span class="nt">Sections</span>'
        '<a href="#sum">Summary</a><a href="#dsi">Dataset</a>'
        '<a href="#dst">Distribution</a><a href="#cnf">Confidence</a>'
        '<a href="#ccl">Conf/Class</a>'
    )
    if has_truth:
        h += '<a href="#cmx">Confusion</a><a href="#pcl">Per-Class</a>'
    if vj_usage:
        h += '<a href="#vjg">V/J Genes</a>'
    h += '<a href="#lcf">Low Conf</a><a href="#exp">Export</a></div>\n'

    # Summary cards
    h += '<div id="sum" class="sg">\n'
    summary_cards = [(f"{n_cells:,}", "Total Cells"), (f"{len(class_names)}", "Classes")]
    if has_truth:
        summary_cards += [
            (f"{accuracy:.1%}", "Accuracy"),
            (f"{macro_f1:.3f}", "Macro F1"),
            (f"{weighted_f1:.3f}", "Weighted F1"),
        ]
    summary_cards += [
        (f"{conf_stats['mean']:.1%}", "Mean Conf"),
        (f"{conf_stats['above_90']:,}", "High (>90%)"),
        (f"{conf_stats['below_50']:,}", "Low (<50%)"),
    ]
    for val, label in summary_cards:
        h += f'<div class="sc"><div class="sv">{val}</div><div class="sl">{label}</div></div>\n'
    h += '</div>\n'

    # Dataset info
    h += (
        '<div id="dsi" class="se"><div class="st">'
        '<span class="ic">i</span> Dataset Information</div><div class="ig">\n'
    )
    h += f'<div class="ii"><div class="il">Input File</div><div class="iv">{dataset_info["file"]}</div></div>\n'
    h += f'<div class="ii"><div class="il">Cells</div><div class="iv">{n_cells:,}</div></div>\n'
    h += f'<div class="ii"><div class="il">Genes</div><div class="iv">{n_genes:,}</div></div>\n'
    h += '<div class="ii"><div class="il">Model</div><div class="iv">Top-5 Ensemble</div></div>\n'
    h += '<div class="ii"><div class="il">Modalities</div><div class="iv">GEX + TCR-BERT + V/J</div></div>\n'
    h += f'<div class="ii"><div class="il">Classes</div><div class="iv">{", ".join(class_names)}</div></div>\n'
    for key in ["dataset", "sample", "patient", "tissue", "source", "batch", "study"]:
        if key in dataset_info:
            vals = dataset_info[key]
            display = ", ".join(str(x) for x in vals[:5])
            if len(vals) > 5:
                display += f" (+{len(vals) - 5})"
            h += f'<div class="ii"><div class="il">{key.title()}</div><div class="iv">{display}</div></div>\n'
    h += '</div></div>\n'

    # Class distribution
    max_count = max(class_dist.values())
    sorted_dist = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
    h += (
        '<div id="dst" class="se"><div class="st">'
        '<span class="ic">&#9632;</span> Predicted Class Distribution</div><div class="bc">\n'
    )
    for cn, ct in sorted_dist:
        pct = ct / n_cells * 100
        width = ct / max_count * 100
        color = color_map.get(cn, "#3b82f6")
        h += (
            f'<div class="br"><span class="bl">{cn}</span>'
            f'<div class="bk"><div class="bf" style="width:{width:.1f}%;background:{color};">'
            f'<span class="bv">{ct:,} ({pct:.1f}%)</span></div></div></div>\n'
        )
    h += '</div></div>\n'

    # Confidence + Agreement histograms
    hist_bins = np.linspace(0, 1, 21)
    hist_counts, _ = np.histogram(conf, bins=hist_bins)
    hist_max = max(hist_counts) or 1

    h += (
        '<div id="cnf" class="tc"><div class="se"><div class="st">'
        '<span class="ic">&#9650;</span> Confidence Distribution</div><div class="hi">\n'
    )
    for i, c in enumerate(hist_counts):
        height = c / hist_max * 100
        b = hist_bins[i]
        color = "var(--er)" if b < 0.5 else ("var(--wr)" if b < 0.7 else "var(--ok)")
        h += (
            f'<div class="hb" style="height:{max(height, 1)}%;background:{color};" '
            f'data-t="{b:.0%}-{hist_bins[i+1]:.0%}: {c:,}"></div>\n'
        )
    h += '</div><div style="display:flex;gap:2px;margin-top:.3rem">'
    for i in range(0, 21, 5):
        h += f'<span style="flex:1;text-align:center;font-family:\'JetBrains Mono\',monospace;font-size:.6rem;color:var(--td)">{hist_bins[i]:.0%}</span>'
        if i < 20:
            for _ in range(4):
                h += '<span style="flex:1"></span>'
    h += f'</div><p style="color:var(--td);font-size:.78rem;margin-top:.7rem">Median: {conf_stats["median"]:.1%} &middot; Std: {conf_stats["std"]:.3f}</p></div>\n'

    # Agreement histogram
    agr_bins = np.linspace(0, 1, 11)
    agr_counts, _ = np.histogram(agreement, bins=agr_bins)
    agr_max = max(agr_counts) or 1

    h += '<div class="se"><div class="st"><span class="ic">&#9679;</span> Model Agreement</div><div class="hi">\n'
    for i, c in enumerate(agr_counts):
        height = c / agr_max * 100
        b = agr_bins[i]
        color = "var(--ok)" if b >= 0.8 else ("var(--wr)" if b >= 0.6 else "var(--er)")
        h += (
            f'<div class="hb" style="height:{max(height, 1)}%;background:{color};" '
            f'data-t="{agr_bins[i]:.0%}-{agr_bins[i+1]:.0%}: {c:,}"></div>\n'
        )
    full_agr = int(np.sum(agreement == 1.0))
    h += (
        f'</div><p style="color:var(--td);font-size:.78rem;margin-top:.7rem">'
        f'Full agreement: {full_agr:,} ({full_agr / n_cells:.1%})</p></div></div>\n'
    )

    # Confidence by class table
    h += (
        '<div id="ccl" class="se"><div class="st">'
        '<span class="ic">&#9733;</span> Confidence by Class</div>\n'
        '<table class="cct"><tr><th>Class</th><th>N</th><th>Mean</th>'
        '<th>Median</th><th>Q25-Q75</th><th>Min</th><th>Low(&lt;50%)</th><th></th></tr>\n'
    )
    for cn in class_names:
        if cn not in conf_by_class:
            continue
        c = conf_by_class[cn]
        cl = color_map.get(cn, "#3b82f6")
        bar_width = c["mean"] * 100
        h += (
            f'<tr><td><span style="color:{cl}">&#9679;</span> {cn}</td>'
            f'<td>{c["n"]:,}</td><td>{c["mean"]:.1%}</td><td>{c["median"]:.1%}</td>'
            f'<td>{c["q25"]:.1%} - {c["q75"]:.1%}</td><td>{c["min"]:.1%}</td>'
            f'<td>{c["below_50"]:,}</td>'
            f'<td><span class="cb"><span class="cff" style="width:{bar_width:.0f}%;background:{cl}"></span></span></td></tr>\n'
        )
    h += '</table></div>\n'

    # Confusion matrix
    if has_truth:
        h += (
            '<div id="cmx" class="se"><div class="st">'
            '<span class="ic">&#9632;</span> Confusion Matrix (normalized)</div>'
            '<div class="cmc"><table class="cmt"><tr><th></th>'
        )
        for n in class_names:
            h += f'<th>{n[:6]}</th>'
        h += '</tr>\n'
        for i, n in enumerate(class_names):
            h += f'<tr><th>{n[:6]}</th>'
            for j in range(len(class_names)):
                v = cm_normalized[i, j]
                raw = cm[i, j]
                alpha = max(0.05, v)
                bg = (
                    f"rgba(16,185,129,{alpha})" if i == j
                    else f"rgba(239,68,68,{alpha * 0.8})"
                )
                h += f'<td style="background:{bg}" title="{n} -> {class_names[j]}: {raw}">{v:.2f}</td>'
            h += '</tr>\n'
        h += (
            '</table></div><p style="color:var(--td);font-size:.75rem;margin-top:.6rem;text-align:center">'
            'Rows: true &middot; Columns: predicted &middot; Hover for counts</p></div>\n'
        )

        # Per-class performance
        h += (
            '<div id="pcl" class="se"><div class="st">'
            '<span class="ic">&#9733;</span> Per-Class Performance</div><div class="mg">\n'
        )
        for n in class_names:
            r = report_dict[n]
            cl = color_map.get(n, "#3b82f6")
            h += (
                f'<div class="mc"><div class="md" style="background:{cl}"></div><div>'
                f'<div class="mn">{n}</div>'
                f'<div class="mv">P:{r["precision"]:.3f} R:{r["recall"]:.3f} '
                f'F1:{r["f1-score"]:.3f} n={int(r["support"]):,}</div></div></div>\n'
            )
        h += '</div></div>\n'

    # V/J gene usage
    if vj_usage:
        h += (
            '<div id="vjg" class="se"><div class="st">'
            '<span class="ic">&#9830;</span> Top V/J Gene Usage by Class</div>\n'
        )
        for cn in class_names:
            if cn not in vj_usage:
                continue
            cl = color_map.get(cn, "#3b82f6")
            h += (
                f'<div class="vjb"><div class="vjt">'
                f'<span style="color:{cl}">&#9679;</span> {cn} ({class_dist.get(cn, 0):,})</div>'
                f'<div class="vjg">\n'
            )
            for vc in ["TRAV", "TRAJ", "TRBV", "TRBJ"]:
                h += f'<div class="vjc"><div class="vjn">{vc}</div>\n'
                for gene, pct in vj_usage[cn].get(vc, []):
                    h += f'<div class="vjr"><span>{gene}</span><span>{pct:.1f}%</span></div>\n'
                h += '</div>\n'
            h += '</div></div>\n'
        h += '</div>\n'

    # Low confidence table
    h += (
        '<div id="lcf" class="se"><div class="st">'
        '<span class="ic">!</span> Lowest Confidence Predictions (Top 20)</div>'
        '<table class="dt">\n'
        '<tr><th>Barcode</th><th>Predicted</th><th>Confidence</th>'
        '<th>2nd Choice</th><th>2nd Prob</th><th>Agreement</th></tr>\n'
    )
    for cell in low_conf_data:
        tag_class = "th" if cell["confidence"] > 0.7 else ("tm" if cell["confidence"] > 0.5 else "tl")
        h += (
            f'<tr><td>{cell["barcode"][:30]}</td><td>{cell["predicted"]}</td>'
            f'<td><span class="tg {tag_class}">{cell["confidence"]:.1%}</span></td>'
            f'<td>{cell["second_class"]}</td><td>{cell["second_prob"]:.1%}</td>'
            f'<td>{cell["agreement"]:.0%}</td></tr>\n'
        )
    h += '</table></div>\n'

    # Export section
    summary_json = {
        "timestamp": timestamp,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "classes": class_names,
        "distribution": class_dist,
        "confidence": conf_stats,
        "confidence_by_class": conf_by_class,
    }
    if has_truth:
        summary_json.update({
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        })

    json_text = json.dumps(summary_json, indent=2).replace("<", "&lt;").replace(">", "&gt;")
    h += (
        f'<div id="exp" class="se" style="text-align:center">'
        f'<div class="st" style="justify-content:center">'
        f'<span class="ic">&#8615;</span> Export Results</div>'
        f'<p style="color:var(--td);margin-bottom:1.2rem">'
        f'Download predictions table or copy summary as JSON.</p>'
        f'<div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap">'
        f'<a class="btn bp" href="{csv_filename}" download>Download CSV</a>'
        f'<button class="btn bo" onclick="navigator.clipboard.writeText('
        f"document.getElementById('jd').value).then(()=>{{this.textContent='Copied!';"
        f"setTimeout(()=>this.textContent='Copy Summary JSON',2000)}})"
        f'">Copy Summary JSON</button>'
        f'</div><textarea id="jd" style="display:none">{json_text}</textarea></div>\n'
    )

    # Footer
    h += (
        f'</div><div class="ft">T-Cell Functional State Predictor v2 &middot; '
        f'Multimodal Deep Learning &middot; TCR-BERT + GEX ({n_genes:,}) + V/J &middot; '
        f'Top-5 Ensemble &middot; {timestamp}</div></body></html>'
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(h)

    return output_path
