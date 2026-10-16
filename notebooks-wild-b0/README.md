# OpenForensics Wild — EfficientNet-B0 baseline

Separate from `notebooks-wild/` (B7 ablations). Same dataset and 600 letterbox; the backbone is **EfficientNet-B0**.

B7 baseline and Full VERITAS both reached ~97% on Wild. This notebook is the weaker-capacity check.

| File | Trains |
| --- | --- |
| `WILD_B0_01_baseline.ipynb` | `baseline_effb0_wild` — RGB B0, no multi-stream, no segmentation |

Regenerate: `python make_wild_notebook.py`

## Kaggle

1. **New** GPU notebook (not the B7 Wild session).
2. Import `WILD_B0_01_baseline.ipynb`.
3. Attach `ahmedgamal/openforensics-wild-dataset` only.
4. Accelerator **GPU T4**. Log must say `CUDA available: True`.
5. Smoke, then `full`. One epoch per Save & Run All. Attach **this** notebook’s latest successful output between epochs.

Do not attach B7 Wild checkpoints. `features` + `cls_head` shapes differ (1280 vs 2560).
