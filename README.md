# VERITAS

Vision-based EfficientNet-Reinforced Image Tampering Authentication and ASPP-Segmentation MTL-Framework.

This repository is the **current** implementation: a single Kaggle notebook that trains and compares four EfficientNet-B7 variants on compiled OpenForensics, at **face level** (multiple mixed real/fake faces in one image).

## Run this

**Group (four GPUs in parallel):** each person imports one file from `notebooks/`, then merge with `notebooks/VERITAS_05_compare.ipynb`.  
**Solo:** upload `VERITAS_Kaggle.ipynb` (trains all four variants sequentially).

1. Attach dataset: [nathanielescuro/veritas-openforensics-compiled](https://www.kaggle.com/datasets/nathanielescuro/veritas-openforensics-compiled)
2. Follow **`KAGGLE_RUN_GUIDE.md`** (when to import, attach data, edit `CONFIG`, then run).

Default `run_mode` is `"smoke"` (tiny subset). Set `"full"` only after smoke succeeds on GPU. All four members must use the same `run_mode` and `PROTOCOL_HASH`.

## What is in this repo

| File | Purpose |
| --- | --- |
| `notebooks/VERITAS_01_baseline.ipynb` | Member A — baseline only |
| `notebooks/VERITAS_02_multistream.ipynb` | Member B — multi-stream only |
| `notebooks/VERITAS_03_segmentation.ipynb` | Member C — segmentation MTL only |
| `notebooks/VERITAS_04_full.ipynb` | Member D — full VERITAS only |
| `notebooks/VERITAS_05_compare.ipynb` | Merge the four runs (CPU) |
| `VERITAS_Kaggle.ipynb` | All four experiments in one notebook |
| `KAGGLE_RUN_GUIDE.md` | Step-by-step Kaggle instructions |
| `agent.md` | Original implementation spec |
| `build_notebook*.py` / `assemble_notebook.py` | Regenerates the all-in-one `.ipynb` |

The OpenForensics images are **not** stored here. Use the Kaggle dataset.

## Experiments

| Name | Multi-stream | Segmentation |
| --- | --- | --- |
| `baseline_effb7` | no | no |
| `multistream_no_seg` | yes | no |
| `seg_no_multistream` | no | yes |
| `full_veritas` | yes | yes |
