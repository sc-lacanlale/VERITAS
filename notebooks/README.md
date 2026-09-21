# Group training notebooks

Four people can train one VERITAS variant each, then merge results.

| File | Who | Trains |
| --- | --- | --- |
| `VERITAS_01_baseline.ipynb` | Member A | `baseline_effb7` |
| `VERITAS_02_multistream.ipynb` | Member B | `multistream_no_seg` |
| `VERITAS_03_segmentation.ipynb` | Member C | `seg_no_multistream` |
| `VERITAS_04_full.ipynb` | Member D | `full_veritas` |
| `VERITAS_05_compare.ipynb` | Anyone (CPU) | no training — comparison table + McNemar / Wilcoxon |

The all-in-one notebook `../VERITAS_Kaggle.ipynb` still exists if one person wants to run every variant sequentially.

## Rules so the four runs are comparable

Every member must use the **same** values for:

- `run_mode` (`smoke` for a test, `full` for the thesis)
- `seed` (`42`)
- `split_protocol` / `split_pool`
- `image_size`
- `classification_threshold`
- `face_margin`
- `max_*_images`

Each notebook prints `PROTOCOL_HASH`. All four hashes must match.

The split is computed from that protocol at **image** level, so the same photos (and all of their faces) land in train/val/test for everyone.

## After training

Each member downloads `/kaggle/working` (or at least):

```text
predictions/<exp>_predictions.csv
logs/image_splits.json
logs/protocol.json
logs/<exp>_test_metrics.json
experiment_results.csv
```

One person creates a Kaggle dataset containing all four folders, opens `VERITAS_05_compare.ipynb` (CPU, internet optional), attaches that dataset, and Run All.

See `../KAGGLE_RUN_GUIDE.md` for the Kaggle UI steps.
