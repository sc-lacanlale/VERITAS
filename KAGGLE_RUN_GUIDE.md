# VERITAS — How to run on a Kaggle Notebook

There is **one** notebook: `VERITAS_Kaggle.ipynb`.  
Do not split it into multiple notebooks. Run it **top to bottom** in a single Kaggle session.

This guide is the full sequence: create the session, **then** import the notebook, **then** attach the dataset, **then** set GPU/internet, **then** edit config, **then** run.

---

## What you need before opening Kaggle

1. A Kaggle account (phone-verified if Kaggle asks you to verify before using GPUs).
2. GPU quota remaining: [Kaggle Account settings](https://www.kaggle.com/settings) → scroll to **Accelerators** / GPU quota.
3. The local file:
   ```text
   VERITAS_Kaggle.ipynb
   ```
   This lives in the `veritas-thesis` project folder.
4. The dataset (already public; you will attach it inside Kaggle, not upload the 100k images yourself):
   ```text
   https://www.kaggle.com/datasets/nathanielescuro/veritas-openforensics-compiled
   ```

---

## Step 1 — Sign in

1. Open [https://www.kaggle.com](https://www.kaggle.com).
2. Sign in.
3. Confirm you can see **Code**, **Datasets**, and **Create** in the left/top navigation.

---

## Step 2 — Create a blank GPU notebook (do this first)

Do **not** import the `.ipynb` yet. Create the Kaggle session first so GPU, internet, and data can be attached to it.

1. Click **Create** (or **Code → New Notebook**).
2. Kaggle opens a new untitled notebook with a default empty cell.
3. At the top, click the notebook title (`Untitled`) and rename it, for example:
   ```text
   VERITAS OpenForensics
   ```
4. Open **Session options** (right sidebar; gear icon / **Settings** depending on the UI):

   | Setting | Required value | Why |
   | --- | --- | --- |
   | **Accelerator** | **GPU T4 x2** (or P100 / GPU) | EfficientNet-B7 will be extremely slow on CPU |
   | **Internet** | **On** | Downloads ImageNet EfficientNet-B7 weights and the MediaPipe face-detector `.tflite` |
   | **Persistence** | **Files only** (recommended) | Keeps `/kaggle/working` checkpoints if the session restarts |
   | **Language** | Python | Default |

5. Wait until the session status shows **running** (not “queued”).  
   If GPU is queued for a long time, leave this tab open; do not import the notebook into a CPU session and hope to switch later without a restart.

**Checkpoint:** the right sidebar should show a GPU accelerator, Internet On, and a live session.

---

## Step 3 — Import `VERITAS_Kaggle.ipynb` into this session

Import **after** the GPU session exists. Importing first into a CPU notebook and then turning GPU on often requires a restart and can drop unsaved cells.

1. In the notebook menu: **File → Import notebook**.
2. Choose **Upload**.
3. Select your local file:
   ```text
   VERITAS_Kaggle.ipynb
   ```
4. Confirm. Kaggle replaces (or adds) the cells from that file.
5. Scroll the notebook. You should see sections starting with:

   ```text
   VERITAS — Vision-based EfficientNet-Reinforced Image Tampering Authentication
   ```

   then **1. Environment**, **2. Imports**, **3. Dataset discovery**, **5. Central configuration**, and so on.

6. If Kaggle asks whether to keep the current notebook or overwrite, choose the imported VERITAS notebook as the active notebook.

**Do not click Run All yet.**

---

## Step 4 — Attach the dataset (before any cell that loads images)

The notebook reads from `/kaggle/input/`. If the dataset is not attached, discovery fails.

1. In the right sidebar click **Add Input** / **+ Add data**.
2. Open the **Datasets** tab.
3. Search:
   ```text
   veritas-openforensics-compiled
   ```
   or paste:
   ```text
   nathanielescuro/veritas-openforensics-compiled
   ```
4. Click **Add** / **+** on that dataset.
5. Wait until it appears under **Input**. You should see a folder similar to:
   ```text
   /kaggle/input/veritas-openforensics-compiled/
   ```
6. Expand it and confirm these exist somewhere under that folder (names may be nested one extra level):
   - `compiled_index.parquet` (preferred index)
   - `compiled_poly.json`
   - `Images/` (flat `.jpg` files)

You do **not** copy the local `dataset/` folder onto Kaggle. The public dataset is the same compiled OpenForensics set.

**Do not click Run All yet.**

---

## Step 5 — Edit configuration (the only cell you must change)

Find the markdown heading **5. Central configuration** and the code cell under it that starts with:

```python
CONFIG: Dict[str, Any] = {
    "run_mode": "smoke",
    ...
}
```

Edit **this cell only**, then stop. Do not edit model code.

### 5A. First run on Kaggle (do this)

Leave smoke mode on so you can prove GPU + dataset + training work:

```python
"run_mode": "smoke",
```

Smoke automatically uses:

- 24 train / 8 val / 8 test **images**
- `image_size = 320`
- `epochs = 1`
- all four experiment names
- no loss-weight sweep

This is the correct first run. It is **not** the thesis result.

### 5B. Thesis / full research run (only after smoke succeeds)

Change:

```python
"run_mode": "full",
```

Then confirm / set:

```python
"image_size": 600,          # EfficientNet-B7 native size
"batch_size": 4,            # drop to 2 or 1 if you OOM
"gradient_accumulation": 1, # raise to 4 if batch_size is 1
"epochs": 8,
"split_protocol": "70_15_15",
"split_pool": "train_val_testdev",
"skip_training": False,
"resume": True,
"experiments_to_run": [
    "baseline_effb7",
    "multistream_no_seg",
    "seg_no_multistream",
    "full_veritas",
],
"run_loss_sweeps": False,   # turn True only after the four variants finish
```

If 8 hours is not enough for all four EfficientNet-B7 models, run **one variant per session**:

```python
"experiments_to_run": ["baseline_effb7"],
```

Next session use `multistream_no_seg`, then `seg_no_multistream`, then `full_veritas`.  
Keep `"resume": True` so existing files in `/kaggle/working/checkpoints/` are reused.

Optional official OpenForensics split instead of SOP 70/15/15:

```python
"split_protocol": "official_openforensics",
```

---

## Step 6 — Save a version (optional but useful)

Before a long run:

1. **Save Version** (top right).
2. Version type: **Quick Save** is enough before smoke. Use **Save & Run All** only when you already trust the config.
3. Add a note: `smoke first run` or `full baseline only`.

This stores the notebook snapshot. Training artifacts still go to `/kaggle/working`.

---

## Step 7 — Run, in this order

You have two options. Option A is safer the first time.

### Option A — Run section by section (recommended first time)

Use **Shift+Enter** (run current cell and move to the next).  
Do **not** skip cells. Later cells depend on earlier ones.

| When | What to run | What you must see before continuing |
| --- | --- | --- |
| 1 | **1. Environment** | `CUDA available: True` and a GPU name. If you see `GPU: CPU fallback`, **stop**. Set Accelerator to GPU, restart the session, come back to Step 2/4 (dataset stays attached). |
| 2 | **2. Imports** | No import error. MediaPipe version prints. `MediaPipe Tasks API not importable` is a warning only if the detector later falls back. |
| 3 | **3. Dataset discovery** and **4. Dataset structure** | Prints `parquet:`, `poly.json:`, `Images:` under `/kaggle/input/...`. Sample `.jpg` names appear. If `FileNotFoundError`, go back to Step 4 and attach the dataset, then re-run discovery. |
| 4 | **5. Central configuration** | Prints `run_mode: smoke` or `full`, `device: cuda`, and the rest of CONFIG. Confirm this matches what you intended. |
| 5 | **6. Reproducibility** through **7. Annotation inspection** | `category_id` counts, `0 => real`, `1 => fake`, mixed real/fake image count. |
| 6 | **8–12** geometry, detector, matching, masks | Detector line like `Using MediaPipe Tasks FaceDetector`. First time this may download a `.tflite` (needs Internet On). |
| 7 | **13–15** dataset class + split | `Using SOP image-level 70/15/15...` (or official splits). Train/val/test image counts. Face samples built with `skipped missing 0` (or a small number). |
| 8 | **14b. Preprocessing before / after samples** | Figures: raw image vs boxes/polygons, then 8-panel face strips (crop → letterbox → mask → RGB / residual / FFT). Inspect these **before** training. |
| 9 | **16–18** models, losses, dataloaders | Variant list prints. Loaders show a batch count. |
| 10 | **19–22** then **23. Automated tests** | Tests should print `PASS` lines. Training has not fully run yet unless you already executed the experiment cell. |
| 11 | **24–26** inference helpers | Defines `predict_multi_face` and `debug_image`. |
| 12 | **27. Run experiments** | This is the long cell. It trains every name in `experiments_to_run`, evaluates, writes CSV/checkpoints. Leave the tab open. |
| 13 | **28. Qualitative visualization** | Debug overlays on held-out images. |

### Option B — Run All (after smoke has already worked)

1. Menu: **Run → Run All** (or the Run All button).
2. Stay on the page until the last cell finishes.
3. If the session disconnects, reconnect; with **Persistence: Files only** and `"resume": True`, it can continue from checkpoints.

---

## Step 8 — While training is running

- Do not close the Kaggle tab if you can avoid it.
- Do not turn **Internet** off until EfficientNet-B7 weights and the MediaPipe model have finished downloading (usually during the first model build).
- If you see **CUDA out of memory**:
  1. Stop the run.
  2. In CONFIG set `"batch_size": 1` and `"gradient_accumulation": 4`.
  3. Restart the session (keep dataset + GPU + internet).
  4. Re-run from the top. `"resume": True` will pick up `*_best.pt` / `*_last.pt` if they exist.
- Kaggle GPU sessions are time-limited (often ~9 hours). Plan `experiments_to_run` so one session can finish.

---

## Step 9 — After the last cell

Confirm these files under `/kaggle/input` is read-only; **outputs** are under `/kaggle/working`:

```text
/kaggle/working/
├── checkpoints/
│   ├── baseline_effb7_best.pt
│   ├── baseline_effb7_last.pt
│   ├── multistream_no_seg_best.pt
│   ├── seg_no_multistream_best.pt
│   └── full_veritas_best.pt
├── predictions/
│   ├── baseline_effb7_predictions.csv
│   └── ...
├── debug/
│   ├── preproc_image_*.png
│   ├── preproc_face_*.png
│   └── debug_*.png
├── logs/
│   ├── image_splits.json
│   ├── unit_tests.json
│   └── *_history.json
└── experiment_results.csv
```

Unrun experiments are written as `NOT RUN` or `N/A`. The notebook does not invent metrics.

### Download results

1. In the right sidebar open **Output** / `/kaggle/working`.
2. Download `experiment_results.csv`, `predictions/`, and `checkpoints/` at minimum.
3. **Save Version → Quick Save** so the executed notebook (plots in cells) is stored on Kaggle.

---

## Step 10 — Next session (full run or the next variant)

1. Open the **same** Kaggle notebook (not a brand-new blank one), so `/kaggle/working` persistence can reload.
2. Confirm **GPU + Internet + dataset** are still attached (repeat Steps 2 and 4 if this is a new notebook copy).
3. Change CONFIG (`run_mode`, `experiments_to_run`) as needed.
4. Run All again. With `"resume": True`, finished variants keep their checkpoints.

To force a clean retrain of one variant, delete its files in `/kaggle/working/checkpoints/` first.

If you change the face detector or confidence threshold, delete:

```text
/kaggle/working/face_metadata.json
```

so detection cache is rebuilt.

---

## What not to do

- Do not import the notebook **before** creating the GPU session if you can avoid it.
- Do not hit Run All before the dataset is attached.
- Do not start `"run_mode": "full"` until smoke has shown `CUDA available: True`, discovery paths, preprocess figures, and at least one training epoch.
- Do not clone a private GitHub repo or point the notebook at your PC `C:\Users\...` paths. Kaggle must use `/kaggle/input/` and `/kaggle/working/`.
- Do not use deprecated `mediapipe.solutions` code. The notebook already uses MediaPipe **Tasks**.
- Do not run cells out of order.

---

## Short checklist (print this)

- [ ] Kaggle signed in, GPU quota available
- [ ] **Create → New Notebook**
- [ ] Rename notebook
- [ ] Accelerator = GPU, Internet = On, Persistence = Files only
- [ ] Wait until session is running
- [ ] **File → Import notebook → Upload `VERITAS_Kaggle.ipynb`**
- [ ] **Add Input** → `nathanielescuro/veritas-openforensics-compiled`
- [ ] Edit **5. Central configuration** (`smoke` first, then `full`)
- [ ] Run Environment cell → confirm CUDA
- [ ] Run discovery → confirm `/kaggle/input/...` paths
- [ ] Inspect preprocess before/after figures
- [ ] Run All / remaining cells through experiment table
- [ ] Download `/kaggle/working/experiment_results.csv` and checkpoints
- [ ] Save Version
