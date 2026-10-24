# VERITAS — How to run on a Kaggle Notebook

Two studies, two notebook folders. Pick **one** study per training session.

| Study | Import from | Attach | Experiment names |
| --- | --- | --- | --- |
| OpenForensics Wild | `notebooks-wild/` | `ahmedgamal/openforensics-wild-dataset` | `*_wild` |
| OpenForensics Full | `notebooks-of/` | `defactodataset/defactoface` and `ahmedgamal/openforensics-wild-dataset` | `*_defacto` (kept so existing checkpoints match) |

Wild B0: `notebooks-wild-b0/WILD_B0_01_baseline.ipynb`.

The app test model is `full_veritas_defacto_best` from `OF_04_full`. `WILD_05_compare` will use that checkpoint as Full VERITAS on Wild Test if `full_veritas_wild` is missing.

---

## Group workflow (four people, four GPUs)

Each member uses **their own** Kaggle notebook.

| Member | Wild | OpenForensics Full | Trains |
| --- | --- | --- | --- |
| A | `WILD_01_baseline.ipynb` | `OF_01_baseline.ipynb` | Baseline EfficientNet-B7 |
| B | `WILD_02_multistream.ipynb` | `OF_02_multistream.ipynb` | Multi-stream, no segmentation |
| C | `WILD_03_segmentation.ipynb` | `OF_03_segmentation.ipynb` | Segmentation / MTL, RGB only |
| D | `WILD_04_full.ipynb` | `OF_04_full.ipynb` | Full VERITAS |

Agree on CONFIG before Run All: `run_mode`, `seed` 42, `image_size`, `classification_threshold` 0.5.

Compare `PROTOCOL_HASH` across the four notebooks of the same study.

### After epoch 1

1. `WILD_05_compare.ipynb` or `OF_05_compare.ipynb`.
2. Attach each training output. For Wild Test of the Full model you may attach `OF_04_full` (`full_veritas_defacto_best.pt`) instead of `WILD_04`.
3. GPU T4 if scoring checkpoints.

---

## What you need

1. Kaggle account (GPU quota).
2. The notebook you were assigned.
3. The dataset(s) for that study.

---

## Step 1 — Sign in

Open https://www.kaggle.com and sign in.

---

## Step 2 — Create a blank GPU notebook

1. **Create** → New Notebook.
2. Accelerator **GPU T4**, Internet **On**, Persistence **Files only**, Python.
3. Wait until the session is running.

---

## Step 3 — Import the `.ipynb`

File → Import notebook → upload `WILD_0x` or `OF_0x`.

---

## Step 4 — Attach the dataset

1. Add Input → Datasets.
2. Wild: `openforensics-wild-dataset`.
3. OpenForensics Full (`OF_*`): add `defactodataset/defactoface` **and** `ahmedgamal/openforensics-wild-dataset`.

---

## Step 5 — CONFIG

Set `run_mode` to `"smoke"` first, then `"full"`. Full needs CUDA.

---

## Step 6 — Run All

Save Version → Save & Run All. Check `CUDA available: True` and `START OF RUN`.

One epoch per run. Attach **this** notebook’s latest successful output for the next epoch.

App setup after you have `full_veritas_defacto_best.zip`: **`APP_SETUP.md`**.
