"""Create four single-experiment training notebooks plus a comparison notebook."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import nbformat as nbf

SRC = Path("VERITAS_Kaggle.ipynb")
OUT_DIR = Path("notebooks")

VARIANTS = [
    {
        "file": "VERITAS_01_baseline.ipynb",
        "exp": "baseline_effb7",
        "title": "Notebook 1 of 4 — Baseline EfficientNet-B7",
        "who": "Member A",
    },
    {
        "file": "VERITAS_02_multistream.ipynb",
        "exp": "multistream_no_seg",
        "title": "Notebook 2 of 4 — Multi-stream classification (no segmentation)",
        "who": "Member B",
    },
    {
        "file": "VERITAS_03_segmentation.ipynb",
        "exp": "seg_no_multistream",
        "title": "Notebook 3 of 4 — Segmentation / MTL (single RGB stream)",
        "who": "Member C",
    },
    {
        "file": "VERITAS_04_full.ipynb",
        "exp": "full_veritas",
        "title": "Notebook 4 of 4 — Full VERITAS",
        "who": "Member D",
    },
]


def join_src(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else (src or "")


def set_src(cell, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.split("\n")[:-1]]
    if text.endswith("\n"):
        pass


def patch_training_nb(nb, exp: str, title: str, who: str):
    for cell in nb.cells:
        src = join_src(cell)
        if cell.cell_type == "markdown" and src.startswith("# VERITAS"):
            extra = f"""
# VERITAS — {title}

**Assigned to:** {who}  
**This notebook trains only:** `{exp}`

The other three variants are **not** trained here. After every member finishes, merge results with `VERITAS_05_compare.ipynb`.

## Shared protocol (must be identical on all four notebooks)

Do **not** change these if you want a valid comparison:

- `run_mode` (all `smoke` together, or all `full` together)
- `seed` (keep `42`)
- `split_protocol` / `split_pool`
- `image_size`
- `classification_threshold`
- `face_margin`
- `max_train_images` / `max_val_images` / `max_test_images`

`experiments_to_run` is already locked to `["{exp}"]`.

Dataset: attach `nathanielescuro/veritas-openforensics-compiled`. Follow `KAGGLE_RUN_GUIDE.md`.
"""
            cell["source"] = extra.strip() + "\n"
        if cell.cell_type == "code" and '"experiments_to_run": [' in src:
            src = src.replace(
                '"experiments_to_run": [\n        "baseline_effb7",\n        "multistream_no_seg",\n        "seg_no_multistream",\n        "full_veritas",\n    ],',
                f'"experiments_to_run": ["{exp}"],  # locked: this notebook trains only this variant',
            )
            cell["source"] = src
    # fingerprint cell after CONFIG
    fp = nbf.v4.new_code_cell(
        """
# Shared-protocol fingerprint. All four members must print the SAME hash.
import hashlib, json as _json
_protocol = {k: CONFIG[k] for k in [
    "run_mode", "seed", "image_size", "face_margin", "classification_threshold",
    "split_protocol", "split_pool", "max_train_images", "max_val_images", "max_test_images",
]}
_blob = _json.dumps(_protocol, sort_keys=True, default=str)
PROTOCOL_HASH = hashlib.sha256(_blob.encode()).hexdigest()[:16]
print("THIS NOTEBOOK TRAINS:", CONFIG["experiments_to_run"])
print("PROTOCOL_HASH:", PROTOCOL_HASH)
print(_json.dumps(_protocol, indent=2))
(PATHS["working"] / "logs" / "protocol.json").write_text(
    _json.dumps({"hash": PROTOCOL_HASH, "protocol": _protocol, "experiment": CONFIG["experiments_to_run"]}, indent=2),
    encoding="utf-8",
)
""".strip()
        + "\n"
    )
    # insert after the CONFIG cell (first code cell containing experiments_to_run)
    inserted = False
    new_cells = []
    for cell in nb.cells:
        new_cells.append(cell)
        if (not inserted) and cell.cell_type == "code" and "experiments_to_run" in join_src(cell):
            new_cells.append(fp)
            inserted = True
    nb.cells = new_cells


def make_compare_nb():
    cells = []

    def md(t):
        cells.append(nbf.v4.new_markdown_cell(t.strip() + "\n"))

    def cd(t):
        cells.append(nbf.v4.new_code_cell(t.strip() + "\n"))

    md(
        """
# VERITAS — Notebook 5 of 5: Compare the four experiments

This notebook does **not** train. It merges outputs from the four training notebooks.

## Before you run

Each member downloads their `/kaggle/working` folder after training. You need, from **each** variant:

```text
predictions/<exp>_predictions.csv
logs/<exp>_test_metrics.json      (if present)
logs/image_splits.json
logs/protocol.json                (if present)
experiment_results.csv            (if present)
```

Put all four output folders into **one Kaggle dataset** (or add four datasets) and attach it to this notebook. CPU is enough.

Expected experiment names:

- `baseline_effb7`
- `multistream_no_seg`
- `seg_no_multistream`
- `full_veritas`

Comparison is only valid if all four used the same `PROTOCOL_HASH` (same seed, split, image size, threshold).
"""
    )
    cd(
        r"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

SEARCH_ROOTS = []
if Path("/kaggle/input").exists():
    SEARCH_ROOTS.append(Path("/kaggle/input"))
if Path("/kaggle/working").exists():
    SEARCH_ROOTS.append(Path("/kaggle/working"))
SEARCH_ROOTS.append(Path.cwd())
SEARCH_ROOTS.append(Path.cwd() / "kaggle_working")

EXPECTED = ["baseline_effb7", "multistream_no_seg", "seg_no_multistream", "full_veritas"]
print("search roots:", [str(p) for p in SEARCH_ROOTS if p.exists()])
"""
    )
    cd(
        r"""
def find_files(name_substr: str, suffix: str):
    hits = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() == suffix and name_substr in p.name:
                hits.append(p)
    return sorted(set(hits))


pred_files = {}
for exp in EXPECTED:
    cands = find_files(f"{exp}_predictions", ".csv")
    pred_files[exp] = cands[0] if cands else None
    print(exp, "predictions:", pred_files[exp])

split_files = find_files("image_splits", ".json")
protocol_files = find_files("protocol", ".json")
metric_files = {exp: (find_files(f"{exp}_test_metrics", ".json") or [None])[0] for exp in EXPECTED}
result_csvs = find_files("experiment_results", ".csv")
print("split files", len(split_files))
print("protocol files", len(protocol_files))
print("result csvs", [str(p) for p in result_csvs])
"""
    )
    cd(
        r"""
# --- protocol / split consistency ---
protocols = []
for p in protocol_files:
    try:
        protocols.append((p, json.loads(p.read_text(encoding="utf-8"))))
    except Exception as e:
        print("skip protocol", p, e)

if protocols:
    hashes = {json.dumps(obj, sort_keys=True) and obj.get("hash") for _, obj in protocols}
    print("PROTOCOL hashes found:", hashes)
    if len(hashes) > 1:
        print("WARNING: protocol hashes differ. Do not treat metrics as a controlled ablation.")
        for p, obj in protocols:
            print(" ", p, obj.get("hash"), obj.get("experiment"))
    else:
        print("OK: all protocol.json files share the same hash.")
else:
    print("No protocol.json found. Comparison can still run if prediction IDs match.")

splits = []
for p in split_files:
    try:
        splits.append((p, json.loads(p.read_text(encoding="utf-8"))))
    except Exception as e:
        print("skip split", p, e)

if len(splits) >= 2:
    ref = splits[0][1]
    ok = True
    for p, obj in splits[1:]:
        for key in ("train", "val", "test"):
            if sorted(map(int, obj.get(key, []))) != sorted(map(int, ref.get(key, []))):
                print("SPLIT MISMATCH", key, "in", p)
                ok = False
    print("splits consistent:" if ok else "splits DIFFER:", "compare test IDs only if they match")
"""
    )
    cd(
        r"""
# --- load face-level predictions ---
frames = {}
for exp, path in pred_files.items():
    if path is None:
        print("MISSING predictions for", exp)
        continue
    df = pd.read_csv(path)
    need = {"image_id", "face_id", "ground_truth_label", "predicted_label", "fake_probability"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df["experiment"] = exp
    frames[exp] = df
    print(exp, "n=", len(df), "acc=", float((df.ground_truth_label == df.predicted_label).mean()))

if len(frames) < 2:
    raise RuntimeError("Need at least two experiment prediction CSVs to compare.")
"""
    )
    cd(
        r"""
def cls_metrics(df):
    y = df["ground_truth_label"].astype(int)
    p = df["predicted_label"].astype(int)
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    row = {
        "n_faces": int(len(df)),
        "accuracy": float(accuracy_score(y, p)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if "iou" in df.columns:
        fake = df[df.ground_truth_label == 1]
        row["miou_fake_only"] = float(fake["iou"].mean()) if len(fake) else float("nan")
        row["dice_fake_only"] = float(fake["dice"].mean()) if len(fake) and "dice" in fake.columns else float("nan")
    else:
        row["miou_fake_only"] = "N/A"
        row["dice_fake_only"] = "N/A"
    return row


rows = []
meta = {
    "baseline_effb7": {"multi_stream": False, "segmentation": False},
    "multistream_no_seg": {"multi_stream": True, "segmentation": False},
    "seg_no_multistream": {"multi_stream": False, "segmentation": True},
    "full_veritas": {"multi_stream": True, "segmentation": True},
}
for exp in EXPECTED:
    if exp not in frames:
        rows.append({"experiment": exp, "status": "NOT RUN", **meta[exp]})
        continue
    rec = {"experiment": exp, "status": "ok", **meta[exp], **cls_metrics(frames[exp])}
    mf = metric_files.get(exp)
    if mf is not None:
        blob = json.loads(Path(mf).read_text(encoding="utf-8"))
        compute = blob.get("compute", {})
        rec["mean_latency_ms"] = compute.get("mean_latency_ms_per_face")
        rec["peak_vram_mb"] = compute.get("peak_vram_mb")
        rec["input_resolution"] = compute.get("input_resolution")
        rec["device"] = compute.get("device")
    rows.append(rec)

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working"
out_dir.mkdir(parents=True, exist_ok=True)
summary.to_csv(out_dir / "experiment_results_compared.csv", index=False)
print("wrote", out_dir / "experiment_results_compared.csv")
"""
    )
    cd(
        r"""
def key(df):
    return list(zip(df["image_id"].astype(int), df["face_id"].astype(int)))


def mcnemar(a: pd.DataFrame, b: pd.DataFrame):
    ka = {(int(r.image_id), int(r.face_id)): r for r in a.itertuples()}
    kb = {(int(r.image_id), int(r.face_id)): r for r in b.itertuples()}
    common = sorted(set(ka) & set(kb))
    n01 = n10 = 0
    for k in common:
        aw = int(ka[k].predicted_label != ka[k].ground_truth_label)
        bw = int(kb[k].predicted_label != kb[k].ground_truth_label)
        if aw == 0 and bw == 1:
            n01 += 1
        elif aw == 1 and bw == 0:
            n10 += 1
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n=n, p=0.5).pvalue)
    return {"n_paired": len(common), "a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "pvalue": p}


def wilcoxon_iou(a: pd.DataFrame, b: pd.DataFrame):
    if "iou" not in a.columns or "iou" not in b.columns:
        return {"note": "no IoU column"}
    ka = {(int(r.image_id), int(r.face_id)): r for r in a.itertuples()}
    kb = {(int(r.image_id), int(r.face_id)): r for r in b.itertuples()}
    common = sorted(set(ka) & set(kb))
    diffs = [float(ka[k].iou) - float(kb[k].iou) for k in common]
    if len(diffs) < 10:
        return {"n": len(diffs), "note": "too few paired IoU values"}
    try:
        stat, p = stats.wilcoxon(diffs)
        return {"n": len(diffs), "stat": float(stat), "pvalue": float(p)}
    except Exception as e:
        return {"n": len(diffs), "error": str(e)}


pairs = [
    ("baseline_effb7", "multistream_no_seg"),
    ("baseline_effb7", "seg_no_multistream"),
    ("baseline_effb7", "full_veritas"),
    ("multistream_no_seg", "full_veritas"),
    ("seg_no_multistream", "full_veritas"),
]
stat_rows = []
for a, b in pairs:
    if a not in frames or b not in frames:
        stat_rows.append({"pair": f"{a} vs {b}", "status": "missing"})
        continue
    mc = mcnemar(frames[a], frames[b])
    wx = wilcoxon_iou(frames[a], frames[b])
    stat_rows.append({"pair": f"{a} vs {b}", "mcnemar": mc, "wilcoxon_iou": wx})
    print(f"{a} vs {b}")
    print("  McNemar", mc)
    print("  Wilcoxon IoU", wx)

(out_dir / "statistical_tests_compared.json").write_text(json.dumps(stat_rows, indent=2, default=str), encoding="utf-8")
print("wrote", out_dir / "statistical_tests_compared.json")
"""
    )
    md(
        """
## How to read this

- If `PROTOCOL_HASH` differed across members, stop and rerun the mismatched notebook with the agreed CONFIG.
- If test `image_id` sets differ, McNemar/Wilcoxon are invalid.
- `miou_fake_only` / `dice_fake_only` exist only for the two segmentation variants.
- Classification metrics are face-level on the shared test faces.
"""
    )

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "None",
    }
    nb["cells"] = cells
    return nb


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_nb = nbf.read(SRC, as_version=4)
    for v in VARIANTS:
        nb = deepcopy(src_nb)
        # drop executed outputs so each member starts clean (keeps source)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        patch_training_nb(nb, v["exp"], v["title"], v["who"])
        dest = OUT_DIR / v["file"]
        nbf.write(nb, dest)
        print("wrote", dest, "exp", v["exp"])

    cmp = make_compare_nb()
    dest = OUT_DIR / "VERITAS_05_compare.ipynb"
    nbf.write(cmp, dest)
    print("wrote", dest)


if __name__ == "__main__":
    main()
