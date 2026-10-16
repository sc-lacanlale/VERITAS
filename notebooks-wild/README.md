# OpenForensics Wild — VERITAS ablations

Separate from `notebooks/` (compiled OpenForensics + polygons). This folder trains the same four EfficientNet-B7 variants on:

https://www.kaggle.com/datasets/ahmedgamal/openforensics-wild-dataset

Images are already face crops. Labels come from folders (`Real` / `Fake`). Splits are the dataset’s **Train / Validation / Test**.

Wild has **no polygons**. Segmentation and Full still train a DeepLabV3+ head, but the GT mask is a **face-oval on fake crops** (zeros on reals). Dice/IoU are not comparable to compiled-poly.

| File | Who | Trains |
| --- | --- | --- |
| `WILD_01_baseline.ipynb` | Member A | `baseline_effb7_wild` (RGB B7; keep this notebook to resume the ~60% run) |
| `WILD_02_multistream.ipynb` | Member B | `multistream_no_seg_wild` |
| `WILD_03_segmentation.ipynb` | Member C | `seg_no_multistream_wild` |
| `WILD_04_full.ipynb` | Member D | `full_veritas_wild` |
| `WILD_05_compare.ipynb` | Anyone (GPU if scoring checkpoints) | no training — Test from CSVs **or** epoch-1 checkpoints; 02/03 optional |

Regenerate: `python make_wild_notebook.py`

## Kaggle

Each training notebook needs its **own** GPU session (not DS18, not VERITAS-finetuning, not another Wild notebook).

1. Import one of `WILD_01` … `WILD_04`.
2. **Add Input → Datasets** → `ahmedgamal/openforensics-wild-dataset`.
3. Settings → Accelerator → **GPU T4** (not CPU / None). Internet **On** (ImageNet on the first run), Persistence Files only. The log must say `CUDA available: True`. `PyTorch: …+cpu` means the session has no GPU — Full will burn 12h and finish ~3% of an epoch.
4. First run: leave `"run_mode": "smoke"`. Then set `"full"`. Full mode raises if CUDA is missing.
5. Each Save & Run All trains **one** epoch (`epochs_per_run: 1`). Attach **that notebook’s latest successful version** for the next epoch (including mid-epoch saves).

Do not attach `veritas-openforensics-compiled` here.

If B7 baseline and Full both sit near 97%, the weaker-backbone check is `../notebooks-wild-b0/` (`WILD_B0_01_baseline.ipynb`, EfficientNet-B0, same 600 crops).

All four must print the same `PROTOCOL_HASH`. To peek at **epoch 1** before more training: import `WILD_05_compare.ipynb` (GPU T4), attach the Wild dataset plus each training notebook’s epoch-1 output. Compare scores official Test from `{exp}_last.pt` when a predictions CSV is missing. New training runs also write Test after every epoch (`test_eval_when: always`).
