# VERITAS

Vision-based EfficientNet-Reinforced Image Tampering Authentication and ASPP-Segmentation MTL-Framework.

This repository is the **current** implementation: a single Kaggle notebook that trains and compares four EfficientNet-B7 variants on compiled OpenForensics, at **face level** (multiple mixed real/fake faces in one image).

## Run this

1. Upload `VERITAS_Kaggle.ipynb` to a Kaggle GPU notebook.
2. Attach dataset: [nathanielescuro/veritas-openforensics-compiled](https://www.kaggle.com/datasets/nathanielescuro/veritas-openforensics-compiled)
3. Follow **`KAGGLE_RUN_GUIDE.md`** step by step (when to import the notebook, attach data, edit `CONFIG`, then run).

Default `run_mode` is `"smoke"` (tiny subset). Set `"full"` only after smoke succeeds on GPU.

## What is in this repo

| File | Purpose |
| --- | --- |
| `VERITAS_Kaggle.ipynb` | Full experimental notebook (the thing you run) |
| `KAGGLE_RUN_GUIDE.md` | Step-by-step Kaggle instructions |
| `agent.md` | Original implementation spec |
| `build_notebook*.py` / `assemble_notebook.py` | Regenerates the `.ipynb` if you edit the cell sources |

The OpenForensics images are **not** stored here. Use the Kaggle dataset.

## Experiments

| Name | Multi-stream | Segmentation |
| --- | --- | --- |
| `baseline_effb7` | no | no |
| `multistream_no_seg` | yes | no |
| `seg_no_multistream` | no | yes |
| `full_veritas` | yes | yes |
