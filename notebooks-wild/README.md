# OpenForensics Wild — VERITAS ablations

Official Wild `Train` / `Validation` / `Test` with folder labels `Real` / `Fake`. No polygons (oval surrogate on fakes).

The other study is [`../notebooks-of/`](../notebooks-of/) (`OF_01` … `OF_05`). The app uses `full_veritas_defacto_best` from `OF_04_full`.

| File | Who | Trains |
| --- | --- | --- |
| `WILD_01_baseline.ipynb` | Member A | `baseline_effb7_wild` |
| `WILD_02_multistream.ipynb` | Member B | `multistream_no_seg_wild` |
| `WILD_03_segmentation.ipynb` | Member C | `seg_no_multistream_wild` |
| `WILD_04_full.ipynb` | Member D | `full_veritas_wild` |
| `WILD_05_compare.ipynb` | Anyone | no training |

Regenerate: `python make_wild_notebook.py`

## Kaggle

1. Import `WILD_01` … `WILD_04` into a **new** GPU session.
2. Attach `ahmedgamal/openforensics-wild-dataset`.
3. GPU T4. `run_mode`: smoke then full. One epoch per Save & Run All.

`WILD_05_compare`: attach Wild dataset + Wild training outputs. To test the **app model** on Wild Test, also attach `OF_04_full` (or drop in `full_veritas_defacto_best.pt`). Compare treats that file as Full VERITAS when `full_veritas_wild` is missing.
