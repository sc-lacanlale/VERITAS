# OpenForensics — VERITAS ablations (GT-mask Full)

Keep `notebooks-wild/` for the official Wild Real/Fake folder protocol (oval masks).

This folder is the **OpenForensics Full** study: same four EfficientNet-B7 variants. Checkpoint names stay `*_defacto` so existing `full_veritas_defacto_best.pt` / `.zip` still match.

| File | Who | Trains (checkpoint name) |
| --- | --- | --- |
| `OF_01_baseline.ipynb` | Member A | `baseline_effb7_defacto` |
| `OF_02_multistream.ipynb` | Member B | `multistream_no_seg_defacto` |
| `OF_03_segmentation.ipynb` | Member C | `seg_no_multistream_defacto` |
| `OF_04_full.ipynb` | Member D | `full_veritas_defacto` |
| `OF_05_compare.ipynb` | Anyone | no training |

Regenerate: `python make_defacto_notebook.py`

The app loads `models/full_veritas_defacto_best.zip` from `OF_04_full`. See `APP_SETUP.md`.

`WILD_05_compare.ipynb` can attach `full_veritas_defacto_best.pt` and score official Wild Test with that Full model.
