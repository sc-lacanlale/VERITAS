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

## Test the 3-epoch Full model (`OF_05_compare`)

1. Import `OF_05_compare.ipynb` into a **new GPU T4** notebook.
2. Attach `full_veritas_defacto_best.pt` or `.zip` (OF_04 output, or upload the zip as a dataset).
3. Attach one or more test dumps. The notebook auto-detects:
   - OpenForensics Wild (`Dataset/Test/Real` + `Fake`)
   - DEFACTO Face (`swapping_img` / `morphing_img` / `reference`)
   - FaceForensics++ (`original_sequences` + `manipulated_sequences`, or `youtube` + `Deepfakes`/…)
   - ForgeryNet-style or any `Test/Real`+`Test/Fake` or `Test/0_real`+`1_fake`
4. Run All. Each set prints accuracy / F1 (labels only). Dice only if `probe_mask` exists.

`WILD_05_compare.ipynb` can still attach the same Full checkpoint and score official Wild Test only.
