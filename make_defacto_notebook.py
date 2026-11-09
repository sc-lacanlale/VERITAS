"""Build notebooks-of/ — OpenForensics-protocol notebooks (GT-mask Full VERITAS)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DEST_DIR = Path("notebooks-of")

VARIANTS = [
    {
        "file": "OF_01_baseline.ipynb",
        "exp": "baseline_effb7_defacto",
        "title": "Baseline EfficientNet-B7",
        "who": "Member A",
        "n": "1 of 4",
        "multi_stream": False,
        "segmentation": False,
        "legacy_baseline": True,
        "backbone": "efficientnet_b7",
        "run_mode": "full",
        "epochs_per_run": 3,
        "classification_wild_only": False,
        "blurb": "RGB EfficientNet-B7, no multi-stream, no segmentation.",
    },
    {
        "file": "OF_02_multistream.ipynb",
        "exp": "multistream_no_seg_defacto",
        "title": "Multi-stream classification (no segmentation)",
        "who": "Member B",
        "n": "2 of 4",
        "multi_stream": True,
        "segmentation": False,
        "legacy_baseline": False,
        "backbone": "efficientnet_b7",
        "blurb": "RGB + SRM residual + log-FFT, fused into EfficientNet-B7. Classification only.",
    },
    {
        "file": "OF_03_segmentation.ipynb",
        "exp": "seg_no_multistream_defacto",
        "title": "Segmentation / MTL (single RGB stream)",
        "who": "Member C",
        "n": "3 of 4",
        "multi_stream": False,
        "segmentation": True,
        "legacy_baseline": False,
        "backbone": "efficientnet_b7",
        "blurb": "RGB EfficientNet-B7 + DeepLabV3+ head. Fakes use DEFACTO probe_mask GT; reals are zeros.",
    },
    {
        "file": "OF_04_full.ipynb",
        "exp": "full_veritas_defacto",
        "title": "Full VERITAS",
        "who": "Member D",
        "n": "4 of 4",
        "multi_stream": True,
        "segmentation": True,
        "legacy_baseline": False,
        "backbone": "efficientnet_b7",
        "blurb": "Multi-stream + DeepLabV3+ MTL. Same probe_mask GT as notebook 3.",
    },
]


def md(src: str):
    return nbf.v4.new_markdown_cell(src.strip() + "\n")


def cd(src: str):
    return nbf.v4.new_code_cell(src.strip() + "\n")


def _nb(cells):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "GPU",
    }
    nb["cells"] = cells
    return nb


def architecture_md(variant_line: str) -> str:
    return f"""
## Network: what the baseline is, and what changed

The trunk is **EfficientNet-B7** (torchvision). Counted on this architecture with weights left unloaded:

- **273** `Conv2d` layers in the trunk, of which **55** are depthwise (one MBConv block each)
- **7** MBConv stages after a stem convolution, then a 1×1 head convolution (9 children of `features`)
- Inside each MBConv: expand 1×1 (skipped on the first block), depthwise convolution, squeeze-excitation (two 1×1 convolutions), project 1×1
- Nonlinearity in the trunk is **SiLU**, with batch norm. There is **no step function** and **no softmax** in the forward pass

**Baseline (EfficientNet-B7 only)**

- Input is RGB, 3×600×600, ImageNet normalization
- Adaptive average pool, dropout 0.5, **one** linear unit (`Linear(2560 → 1)`). One logit, not two class scores
- **Sigmoid** turns that logit into a fake probability. Softmax is the wrong function here: softmax needs two or more logits that compete. This head has a single logit
- The label is applied **after** the network: fake if the probability is at least 0.5. That cutoff is a hard decision rule, not a layer

{variant_line}

**Full VERITAS relative to that baseline**

- **Multi-stream stem.** Three frozen 5×5 SRM high-pass filters and a log-magnitude FFT (3 channels) are concatenated with RGB, so the stem convolution changes from 3 input channels to **9**. Its weights start as the ImageNet RGB kernel copied three times and divided by 3. The 55 MBConv blocks are unchanged
- **DeepLabV3+ head** (only the segmentation notebooks). ASPP is a 1×1 convolution, three dilated 3×3 convolutions (rates 6, 12, 18), and a pooled 1×1, then a 1×1 projection. A low-level 1×1 reduces the skip to 48 channels. The decoder is two 3×3 convolutions and a 1×1 to one mask channel. Those are **10** learned convolutions on top of the 273, with **ReLU** in ASPP and the decoder. The mask uses a **sigmoid**, same as the classifier
- Classification loss is binary cross-entropy on the logit. The segmentation loss runs only when a DEFACTO `probe_mask` exists. OpenForensics Wild fakes have no mask, so they train the classifier and skip localization
"""


def resume_section(v: dict, exp: str, imagenet_note: str) -> str:
    if v.get("legacy_baseline"):
        return f"""
## Finish this baseline

The saved baseline completed **1 of 3** epochs, and that accuracy counted OpenForensics Wild only (3,990 images). The other three saved files completed 3 epochs and count the mixed test (6,000 images). This run finishes the remaining epochs and scores that same mixed test.

This notebook resumes the 1e-4 checkpoint and trains the remaining epochs in one run. `run_mode` is already `full`.

1. New GPU notebook. Import this file. Do **not** import it into a Wild training session or reuse the old 1.0 DEFACTO output.
2. **Add Input → Datasets** → `defactodataset/defactoface` and `ahmedgamal/openforensics-wild-dataset`.
3. **Add Input → Notebooks** → the successful baseline run whose progress file says `epochs_completed` 1. Use the 1e-4 run, not the 5e-5 or 2e-4 run.
4. Settings → Accelerator → **GPU T4**. Internet On (first run downloads {imagenet_note}).
5. Leave `learning_rate` at 1e-4. Save & Run All.
6. The log must show `START OF RUN | {exp}: 1/3` and, when the run finishes, `END OF RUN | {exp}: 3/3`.

The run stops itself at 11 hours and writes a mid-epoch checkpoint every 30 minutes. If it stops early, attach that output and run again. The classification row counts every test image: up to 2,000 reals, 2,000 DEFACTO fakes, and 2,000 Wild fakes.
"""
    return f"""
## Attach (before Run All)

1. New GPU notebook. Do **not** import this into a Wild training session or reuse the old 1.0 DEFACTO output.
2. **Add Input → Datasets** → `defactodataset/defactoface` **and** `ahmedgamal/openforensics-wild-dataset`.
3. Settings → Accelerator → **GPU T4**. Internet On (first run downloads {imagenet_note}), Persistence Files only.
4. Set `run_mode: full`. One epoch is ~2–4 hours (DEFACTO fakes + Wild Real/Fake). Smoke is only a pipeline check.

## One epoch per run

Each Save & Run All trains **one** more epoch (`epochs_per_run: 1`), writes
`checkpoints/{exp}_last.pt` and `_best.pt`, then scores official **Test**
(`test_eval_when: always`) so you can stop after epoch 1. To score an already-saved
epoch without training more: `"skip_training": True` and attach that run’s output.

Between runs:

1. Output → `logs/{exp}_training_progress.json` → `epochs_completed` went up by 1.
2. **Add Input → Notebooks** → **this** notebook, latest **successful** version.
3. Save & Run All. Log must say `START OF RUN | {exp}: N/3`.
"""


def build_training_nb(v: dict):
    exp = v["exp"]
    ms = "True" if v["multi_stream"] else "False"
    sg = "True" if v["segmentation"] else "False"
    lg = "True" if v["legacy_baseline"] else "False"
    bb = v.get("backbone", "efficientnet_b7")
    imagenet_note = "ImageNet B0" if bb == "efficientnet_b0" else "ImageNet B7"
    merge_note = "The other OF variants are **not** trained here. Merge later with `OF_05_compare.ipynb`."
    if v["legacy_baseline"]:
        variant_line = "**This notebook is the baseline.** It does not add a stream or a segmentation head. The later variants are measured against this network, including the extra inference time."
    else:
        added = []
        if v["multi_stream"]:
            added.append("the 9-channel stem (RGB + SRM + log-FFT)")
        if v["segmentation"]:
            added.append("the DeepLabV3+ localization head")
        variant_line = "**What this notebook changes from the baseline:** " + " and ".join(added) + "."
    run_mode = v.get("run_mode", "smoke")
    epochs_per_run = int(v.get("epochs_per_run", 1))
    wild_only = "True" if v.get("classification_wild_only", True) else "False"
    protocol_extra = ', "backbone"' if bb == "efficientnet_b0" else ""
    seg_note = ""
    if v["segmentation"]:
        seg_note = """
**Masks:** DEFACTO ships a binary **probe_mask** for every fake (forged region).
Fake images are cropped from that mask bbox (+10% margin) and the mask is letterboxed
with the RGB crop. Reals get Haar crop + an all-zero mask. Dice/IoU **are** GT
localization scores.
"""
    cells = [
        md(
            f"""
# VERITAS OpenForensics — {v['title']}

**Assigned to:** {v['who']} ({v['n']})  
**This notebook trains only:** `{exp}`

**Datasets (attach both):**

- [defactodataset/defactoface](https://www.kaggle.com/datasets/defactodataset/defactoface) — swap/morph fakes with GT `probe_mask` (this is the **only** localization supervision)
- [ahmedgamal/openforensics-wild-dataset](https://www.kaggle.com/datasets/ahmedgamal/openforensics-wild-dataset) — official **Real and Fake** folders for binary classification. Wild has no polygons, so those fakes skip the seg loss.

Classification = predicted label vs ground-truth label only. Segmentation loss and dice run **only** on images that have a `probe_mask`. Haar crop when there is no GT box.

Do **not** mix Wild **checkpoints**. Do not attach the F1=1.0 DEFACTO-vs-Wild-Real run — this protocol is different.

{v['blurb']}
{seg_note}
{merge_note}

{resume_section(v, exp, imagenet_note)}
"""
        ),
        md(architecture_md(variant_line)),
        md("## 1. Environment"),
        cd(
            r"""
import time
NOTEBOOK_START_TIME = time.time()

import os
import sys
import platform
import subprocess
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

_required = {
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
}
missing = []
for mod, pip_name in _required.items():
    try:
        __import__(mod if mod != "PIL" else "PIL")
    except Exception:
        missing.append(pip_name)
if missing:
    _pip_install(missing)

import torch
print("Python:", sys.version.split()[0], platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM: %.2f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
"""
        ),
        md("## 2. Imports"),
        cd(
            r"""
import gc
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B7_Weights, efficientnet_b0, efficientnet_b7
from tqdm.auto import tqdm
"""
        ),
        md("## 3. Dataset discovery"),
        cd(
            r"""
def discover_defacto_root() -> Path:
    roots = []
    if Path("/kaggle/input").exists():
        roots.append(Path("/kaggle/input"))
    roots.extend([Path.cwd() / "defacto-face", Path.cwd() / "dataset", Path.cwd()])
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            if p in seen:
                continue
            seen.add(p)
            kids = {d.lower() for d in dirnames}
            if "swapping_img" in kids or "morphing_img" in kids:
                return p
            if "probe_mask" in kids and ("swapping_img" in str(p).lower() or "morphing" in str(p).lower() or "swapping" in str(p).lower()):
                return p.parent
            if "reference" in kids and ("swapping_img" in kids or "morphing_img" in kids):
                return p
    raise FileNotFoundError(
        "Could not find DEFACTO Face. Attach defactodataset/defactoface "
        "(expect swapping_img/, morphing_img/, reference/, and probe_mask/)."
    )


DEFACTO_ROOT = discover_defacto_root()
IN_KAGGLE = Path("/kaggle/input").exists()
WORKING = Path("/kaggle/working") if IN_KAGGLE else Path.cwd() / "kaggle_working"
WORKING.mkdir(parents=True, exist_ok=True)
for sub in ("checkpoints", "predictions", "logs", "debug"):
    (WORKING / sub).mkdir(parents=True, exist_ok=True)
print("defacto root:", DEFACTO_ROOT)
print("working:", WORKING)
print("children:", sorted(p.name for p in DEFACTO_ROOT.iterdir() if p.is_dir())[:20])
"""
        ),
        md("## 4. Configuration"),
        cd(
            f"""
CONFIG: Dict[str, Any] = {{
    "run_mode": "{run_mode}",
    "seed": 42,
    "backbone": "{bb}",
    "use_pretrained": True,
    "image_size": 600,
    "batch_size": 4,
    "num_workers": 2 if os.name != "nt" else 0,
    "epochs": 3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "classification_threshold": 0.5,
    "mask_threshold": 0.5,
    "lambda_cls": 1.0,
    "lambda_seg": 1.0,
    "dice_weight": 1.0,
    "use_amp": True,
    "gradient_accumulation": 1,
    "skip_training": False,
    "resume": True,
    "resume_from_input": True,
    "time_budget_hours": 11.0,
    "multi_gpu": False,
    "epochs_per_run": {epochs_per_run},
    "classification_wild_only": {wild_only},
    "mid_epoch_checkpoint_minutes": 30,
    "test_eval_when": "always",
    "experiment": "{exp}",
    "max_per_split": None,
    "split_protocol": "70_15_15",
    "face_margin": 0.10,
    "methods": ["swapping", "morphing"],
    "max_train_fakes": 12000,
    "max_val_fakes": 2000,
    "max_test_fakes": 2000,
    "max_train_reals": 12000,
    "max_val_reals": 2000,
    "max_test_reals": 2000,
    "use_extra_reals": True,
    "use_extra_fakes": True,
    "max_train_extra_fakes": 12000,
    "max_val_extra_fakes": 2000,
    "max_test_extra_fakes": 2000,
    "balance_batches": True,
}}

EXP_CFG = {{
    "multi_stream": {ms},
    "segmentation": {sg},
    "legacy_baseline": {lg},
    "backbone": "{bb}",
}}

if CONFIG["run_mode"] == "smoke":
    CONFIG.update(
        {{
            "image_size": 320,
            "batch_size": 4,
            "epochs": 1,
            "max_per_split": None,
            "max_train_fakes": 64,
            "max_val_fakes": 64,
            "max_test_fakes": 64,
            "max_train_reals": 64,
            "max_val_reals": 64,
            "max_test_reals": 64,
            "max_train_extra_fakes": 64,
            "max_val_extra_fakes": 64,
            "max_test_extra_fakes": 64,
            "latency_repeats": 3,
        }}
    )
    print("SMOKE MODE: 64 images per class per split, image_size=320, 1 epoch. Set run_mode='full' for the real run.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
print("PyTorch build:", torch.__version__, "CUDA:", torch.cuda.is_available())
if CONFIG["run_mode"] == "full" and DEVICE.type != "cuda":
    raise RuntimeError(
        "Full DEFACTO training needs a GPU session. "
        "Kaggle: Settings → Accelerator → GPU T4 (or P100). "
        f"This process is {{torch.__version__}} CUDA={{torch.cuda.is_available()}}. "
        "Leave run_mode='smoke' to try the pipeline on CPU."
    )
print("THIS NOTEBOOK TRAINS:", CONFIG["experiment"], EXP_CFG)
print("CONFIG:")
for k, v in CONFIG.items():
    print(f"  {{k}}: {{v}}")
"""
        ),
        cd(
            f"""
import hashlib

_protocol = {{k: CONFIG[k] for k in [
    "run_mode", "seed", "image_size", "classification_threshold",
    "epochs", "batch_size", "learning_rate", "lambda_cls", "lambda_seg", "dice_weight",
    "split_protocol", "face_margin",
    "max_train_fakes", "max_val_fakes", "max_test_fakes",
    "max_train_reals", "max_val_reals", "max_test_reals",
    "max_train_extra_fakes", "max_val_extra_fakes", "max_test_extra_fakes",
    "use_extra_reals", "use_extra_fakes"{protocol_extra}
]}}
PROTOCOL_HASH = hashlib.sha256(json.dumps(_protocol, sort_keys=True, default=str).encode()).hexdigest()[:16]
print("THIS NOTEBOOK TRAINS:", CONFIG["experiment"])
print("PROTOCOL_HASH:", PROTOCOL_HASH)
print(json.dumps(_protocol, indent=2))
(WORKING / "logs" / "protocol.json").write_text(
    json.dumps({{"hash": PROTOCOL_HASH, "protocol": _protocol, "experiment": CONFIG["experiment"]}}, indent=2),
    encoding="utf-8",
)
"""
        ),
        md("## 5. Index DEFACTO Face images (70/15/15 + probe_mask GT)"),
        cd(
            r"""
import hashlib

EXTS_IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _dir(root: Optional[Path], name: str) -> Optional[Path]:
    if root is None or not root.is_dir():
        return None
    direct = root / name
    if direct.is_dir():
        return direct
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None


def _as_img_dir(p: Optional[Path]) -> Optional[Path]:
    if p is None or not p.is_dir():
        return None
    inner = p / "img"
    return inner if inner.is_dir() else p


def list_images(folder: Optional[Path]) -> List[Path]:
    if folder is None or not folder.is_dir():
        return []
    out = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS_IMG]
    return sorted(out)


def _norm_stem(stem: str) -> str:
    s = stem.lower()
    for tok in ("_probe_mask", "_probemask", "_probe", "_mask", "_gt", "-mask"):
        s = s.replace(tok, "")
    return s


def _id_key(stem: str) -> str:
    digits = "".join(ch for ch in _norm_stem(stem) if ch.isdigit())
    return digits.lstrip("0") or digits


def _add_mask_keys(idx: Dict[str, Path], p: Path, mask_root: Path) -> None:
    idx[p.stem] = p
    idx[p.stem.lower()] = p
    idx[_norm_stem(p.stem)] = p
    kid = _id_key(p.stem)
    if kid:
        idx[kid] = p
    try:
        rel = p.relative_to(mask_root).with_suffix("").as_posix().lower()
        idx[rel] = p
        idx[Path(rel).name] = p
    except ValueError:
        pass
    parent = p.parent
    if parent != mask_root and parent.name.lower() not in {"img", "probe_mask", "masks", "images"}:
        idx[parent.name] = p
        idx[parent.name.lower()] = p
        idx[_norm_stem(parent.name)] = p
        pid = _id_key(parent.name)
        if pid:
            idx[pid] = p


def index_mask_dir(mask_root: Optional[Path]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    if mask_root is None or not Path(mask_root).is_dir():
        print("  mask dir missing", mask_root)
        return idx
    mask_root = Path(mask_root)
    samples = []
    n_files = 0
    skip_meta = {".json", ".txt", ".xml", ".csv", ".md"}
    files = [p for p in mask_root.rglob("*") if p.is_file()]
    image_files = [p for p in files if p.suffix.lower() in EXTS_IMG]
    if not image_files:
        image_files = [p for p in files if p.suffix.lower() not in skip_meta]
        if image_files:
            print("  no image-ext masks; using", len(image_files), "other files")
    for p in image_files:
        n_files += 1
        _add_mask_keys(idx, p, mask_root)
        if len(samples) < 8:
            samples.append(str(p.relative_to(mask_root)))
    print("  mask files", n_files, "samples", samples)
    if n_files == 0:
        kids = [c.name for c in mask_root.iterdir()][:20]
        print("  mask dir is empty of usable files. children:", kids)
    return idx


def lookup_mask(image_path: Path, img_dir: Path, mask_root: Optional[Path], mask_idx: Dict[str, Path]) -> Optional[Path]:
    if not mask_idx:
        return None
    keys = [image_path.stem, image_path.stem.lower(), _norm_stem(image_path.stem), _id_key(image_path.stem)]
    try:
        rel = image_path.relative_to(img_dir).with_suffix("").as_posix().lower()
        keys.extend([rel, Path(rel).name, Path(rel).parent.as_posix(), Path(rel).parent.name])
    except ValueError:
        pass
    for k in keys:
        if not k:
            continue
        hit = mask_idx.get(k)
        if hit is not None:
            return hit
    return None


def assign_split(key: str) -> str:
    h = hashlib.md5(f"{CONFIG['seed']}:{key}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < 0.70:
        return "train"
    if x < 0.85:
        return "val"
    return "test"


SWAPPING_IMG = _as_img_dir(_dir(DEFACTO_ROOT, "swapping_img"))
MORPHING_IMG = _as_img_dir(_dir(DEFACTO_ROOT, "morphing_img"))
REFERENCE_IMG = _as_img_dir(_dir(DEFACTO_ROOT, "reference"))
SWAP_MASK = _as_img_dir(_dir(_dir(DEFACTO_ROOT, "swapping_annotations"), "probe_mask"))
MORPH_MASK = _as_img_dir(_dir(_dir(DEFACTO_ROOT, "morphing_annotations"), "probe_mask"))
print("swapping_img:", SWAPPING_IMG, "probe_mask:", SWAP_MASK)
print("morphing_img:", MORPHING_IMG, "probe_mask:", MORPH_MASK)
print("reference:", REFERENCE_IMG)

SOURCES = []
if "swapping" in CONFIG["methods"] and SWAPPING_IMG is not None:
    SOURCES.append(("swapping", SWAPPING_IMG, SWAP_MASK))
if "morphing" in CONFIG["methods"] and MORPHING_IMG is not None:
    SOURCES.append(("morphing", MORPHING_IMG, MORPH_MASK))
if not SOURCES:
    raise RuntimeError("No swapping_img/ or morphing_img/ found. Attach defactodataset/defactoface.")

rows_all = []
skipped_nomask = 0
for method, img_dir, mask_dir in SOURCES:
    imgs = list_images(img_dir)
    print(method, "images", len(imgs), "mask dir", mask_dir)
    print("  image samples", [p.name for p in imgs[:8]])
    mask_idx = index_mask_dir(mask_dir)
    paired = []
    unmatched = []
    for p in imgs:
        mp = lookup_mask(p, img_dir, mask_dir, mask_idx)
        if mp is None:
            unmatched.append(p)
        else:
            paired.append((p, mp))
    if not paired and unmatched and mask_dir is not None:
        masks = list_images(Path(mask_dir))
        if not masks:
            masks = [p for p in Path(mask_dir).rglob("*") if p.is_file() and p.suffix.lower() not in {".json", ".txt", ".xml", ".csv", ".md"}]
            masks = sorted(masks)
        if len(masks) == len(imgs):
            print(method, "stem names differ; pairing", len(masks), "files by sorted order")
            paired = list(zip(imgs, masks))
            unmatched = []
        elif masks:
            print(method, "WARNING: cannot pair. image e.g.", unmatched[0].name, "mask e.g.", masks[0].name, "n_img", len(imgs), "n_mask", len(masks))
    skipped_nomask += len(unmatched)
    for p, mp in paired:
        rows_all.append(
            {
                "path": str(p),
                "mask_path": str(mp),
                "split": assign_split(p.stem),
                "label": 1,
                "class_name": "fake",
                "method": method,
                "video_id": p.stem,
                "frame_idx": 0,
                "mask_frame_idx": 0,
                "file_name": p.name,
            }
        )
print("fakes skipped (no probe_mask):", skipped_nomask)

real_imgs = list_images(REFERENCE_IMG)
print("reference reals", len(real_imgs))
if not real_imgs:
    raise RuntimeError("No reference/ real images found under DEFACTO Face.")
for p in real_imgs:
    rows_all.append(
        {
            "path": str(p),
            "mask_path": "",
            "split": assign_split("real:" + p.stem),
            "label": 0,
            "class_name": "real",
            "method": "reference",
            "video_id": p.stem,
            "frame_idx": 0,
            "mask_frame_idx": 0,
            "file_name": p.name,
        }
    )


def discover_extra_openforensics() -> Dict[str, Dict[str, List[Path]]]:
    out: Dict[str, Dict[str, List[Path]]] = {
        "real": {"train": [], "val": [], "test": []},
        "fake": {"train": [], "val": [], "test": []},
    }
    want_real = bool(CONFIG.get("use_extra_reals", True))
    want_fake = bool(CONFIG.get("use_extra_fakes", True))
    if not want_real and not want_fake:
        print("extra OpenForensics disabled")
        return out
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }
    roots = []
    inp = Path("/kaggle/input")
    if inp.is_dir():
        for child in sorted(inp.iterdir()):
            if child.is_dir() and "defacto" not in child.name.lower():
                roots.append(child)
    for local in (Path.cwd() / "openforensics-wild-dataset", Path.cwd() / "Dataset"):
        if local.is_dir():
            roots.append(local)
    seen = set()
    for root in roots:
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            cls = p.name.lower()
            if cls not in {"real", "fake"}:
                continue
            if cls == "real" and not want_real:
                continue
            if cls == "fake" and not want_fake:
                continue
            split = aliases.get(p.parent.name.lower())
            if split is None:
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            imgs = list_images(p)
            if imgs:
                out[cls][split].extend(imgs)
                print("extra OpenForensics", cls, split, len(imgs), p)
    for cls in out:
        for k in out[cls]:
            out[cls][k] = sorted(set(out[cls][k]))
        print("extra", cls, "totals", {k: len(v) for k, v in out[cls].items()})
    return out


extra_of = discover_extra_openforensics()
for split_name, paths in extra_of["real"].items():
    for p in paths:
        rows_all.append(
            {
                "path": str(p),
                "mask_path": "",
                "split": split_name,
                "label": 0,
                "class_name": "real",
                "method": "extra_real",
                "video_id": p.stem,
                "frame_idx": 0,
                "mask_frame_idx": 0,
                "file_name": p.name,
            }
        )
for split_name, paths in extra_of["fake"].items():
    for p in paths:
        rows_all.append(
            {
                "path": str(p),
                "mask_path": "",
                "split": split_name,
                "label": 1,
                "class_name": "fake",
                "method": "extra_fake",
                "video_id": p.stem,
                "frame_idx": 0,
                "mask_frame_idx": 0,
                "file_name": p.name,
            }
        )
n_extra_r = sum(len(v) for v in extra_of["real"].values())
n_extra_f = sum(len(v) for v in extra_of["fake"].values())
if n_extra_r == 0:
    print("WARNING: no Wild Real folders. Classification reals are only DEFACTO reference/", len(real_imgs))
if n_extra_f == 0:
    print("WARNING: no Wild Fake folders. Classification fakes would be DEFACTO-only (dataset shortcut).")
    print("Attach ahmedgamal/openforensics-wild-dataset so Fake/ is used for labels; probe_mask still used for localization.")

ALL = pd.DataFrame(rows_all)


def list_split(split_name: str) -> pd.DataFrame:
    df = ALL[ALL.split == split_name].copy()
    max_n = CONFIG.get("max_per_split")
    if max_n:
        parts = []
        rng = np.random.RandomState(CONFIG["seed"] + {"train": 1, "val": 2, "test": 3}[split_name])
        for lab, g in df.groupby("label"):
            n = min(int(max_n), len(g))
            if n <= 0:
                continue
            idx = rng.choice(g.index, size=n, replace=False)
            parts.append(g.loc[idx])
        df = pd.concat(parts) if parts else df.iloc[0:0]
    return df.reset_index(drop=True)


def _subsample(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rng = np.random.RandomState(CONFIG["seed"] + {"train": 21, "val": 22, "test": 23}[split_name])
    real = df[df.label == 0]
    gt_fake = df[(df.label == 1) & (df.method.isin(["swapping", "morphing"]))]
    extra_fake = df[(df.label == 1) & (df.method == "extra_fake")]
    other_fake = df[(df.label == 1) & (~df.method.isin(["swapping", "morphing", "extra_fake"]))]
    caps = {
        "train": (CONFIG.get("max_train_reals"), CONFIG.get("max_train_fakes"), CONFIG.get("max_train_extra_fakes")),
        "val": (CONFIG.get("max_val_reals"), CONFIG.get("max_val_fakes"), CONFIG.get("max_val_extra_fakes")),
        "test": (CONFIG.get("max_test_reals"), CONFIG.get("max_test_fakes"), CONFIG.get("max_test_extra_fakes")),
    }
    max_r, max_f, max_e = caps[split_name]
    if max_r and len(real) > int(max_r):
        real = real.sample(n=int(max_r), random_state=rng)
    if max_f and len(gt_fake) > int(max_f):
        gt_fake = gt_fake.sample(n=int(max_f), random_state=rng)
    if max_e and len(extra_fake) > int(max_e):
        extra_fake = extra_fake.sample(n=int(max_e), random_state=rng)
    return pd.concat([real, gt_fake, extra_fake, other_fake]).reset_index(drop=True)


INDEX = {s: _subsample(list_split(s), s) for s in ("train", "val", "test")}
for s, df in INDEX.items():
    n_fake = int((df.label == 1).sum()) if len(df) else 0
    n_real = len(df) - n_fake
    src = df.method.value_counts().to_dict() if len(df) and "method" in df.columns else {}
    n_gt = int((df.mask_path.fillna("") != "").sum()) if len(df) and "mask_path" in df.columns else 0
    print(f"{s}: {len(df)} unique images  fake={n_fake} real={n_real}  gt_masks={n_gt}  sources={src}")
if any(len(INDEX[s]) == 0 for s in INDEX):
    raise RuntimeError("A split has 0 images. Check swapping_img / morphing_img / reference / probe_mask.")
for s, df in INDEX.items():
    n_fake = int((df.label == 1).sum())
    n_real = int((df.label == 0).sum())
    if n_fake == 0 or n_real == 0:
        raise RuntimeError(f"{s} is missing reals ({n_real}) or fakes ({n_fake}).")
HAS_GT_MASKS = True
print("HAS_GT_MASKS:", HAS_GT_MASKS)
"""
        ),
        md("## 6. Face crop from GT mask (fakes) or Haar (reals), then 600 letterbox"),
        cd(
            r"""
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def letterbox_rgb(image_rgb: np.ndarray, size: int):
    h, w = image_rgb.shape[:2]
    scale = min(size / max(h, 1), size / max(w, 1))
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, (left, top, nw, nh)


def letterbox_mask(mask: np.ndarray, size: int, box) -> np.ndarray:
    left, top, nw, nh = box
    canvas = np.zeros((size, size), dtype=np.float32)
    if mask is None or mask.size == 0:
        return canvas
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = (m > 127).astype(np.float32)
    else:
        m = (m > 0.5).astype(np.float32)
    resized = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


_HAAR_PATH = getattr(getattr(cv2, "data", None), "haarcascades", "") + "haarcascade_frontalface_default.xml"
try:
    _HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception:
    _HAAR = cv2.CascadeClassifier(_HAAR_PATH)


def read_frame(path: Path, idx: int):
    if path.is_file() and path.suffix.lower() in EXTS_IMG:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if path.is_dir():
        frames = sorted([c for c in path.iterdir() if c.is_file() and c.suffix.lower() in EXTS_IMG])
        if not frames:
            return None
        idx = int(np.clip(idx, 0, len(frames) - 1))
        bgr = cv2.imread(str(frames[idx]), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(idx, 0)))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask_frame(path: Path, idx: int):
    rgb = read_frame(path, idx)
    if rgb is None:
        return None
    if rgb.ndim == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb


def bbox_from_mask(mask: np.ndarray, margin: float):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    pad_x, pad_y = int(round(bw * margin)), int(round(bh * margin))
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(w, x1 + pad_x), min(h, y1 + pad_y)
    return x0, y0, x1, y1


def bbox_from_haar(rgb: np.ndarray, margin: float):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    dummy = np.zeros(rgb.shape[:2], dtype=np.uint8)
    dummy[y : y + h, x : x + w] = 1
    return bbox_from_mask(dummy, margin)


def crop_pair(rgb: np.ndarray, mask: np.ndarray, box):
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def crop_ellipse_mask(size: int, box) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.float32)
    left, top, nw, nh = box
    cx, cy = left + nw / 2.0, top + nh / 2.0
    rx, ry = max(nw * 0.46, 1.0), max(nh * 0.46, 1.0)
    yy, xx = np.ogrid[:size, :size]
    m[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0] = 1.0
    return m


class DEFACTOFaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.augment = bool(augment)
        self.margin = float(CONFIG.get("face_margin") or 0.10)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rgb = read_frame(Path(row.path), int(row.frame_idx))
        if rgb is None:
            rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        is_fake = int(row.label) == 1
        mask_np = None
        mask_str = str(getattr(row, "mask_path", "") or "")
        if is_fake and mask_str and mask_str not in ("nan", "None"):
            midx = int(row.mask_frame_idx) if "mask_frame_idx" in row.index and not pd.isna(row.mask_frame_idx) else int(row.frame_idx)
            mask_np = read_mask_frame(Path(row.mask_path), midx)
        used_gt = mask_np is not None
        if used_gt:
            mask_bin = (mask_np > 127).astype(np.uint8) if mask_np.max() > 1 else (mask_np > 0).astype(np.uint8)
            box = bbox_from_mask(mask_bin, self.margin)
        else:
            mask_bin = np.zeros(rgb.shape[:2], dtype=np.uint8)
            box = bbox_from_haar(rgb, self.margin)
        if box is None:
            box = (0, 0, rgb.shape[1], rgb.shape[0])
        if not is_fake:
            mask_bin = np.zeros(rgb.shape[:2], dtype=np.uint8)
        crop, mcrop = crop_pair(rgb, mask_bin, box)
        if self.augment and random.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
            mcrop = np.ascontiguousarray(mcrop[:, ::-1])
        if self.augment and random.random() < 0.5:
            crop = np.clip(crop.astype(np.float32) * (1.0 + random.uniform(-0.15, 0.15)) + random.uniform(-12, 12), 0, 255).astype(np.uint8)
        canvas, lbox = letterbox_rgb(crop, self.size)
        if used_gt:
            mask = letterbox_mask(mcrop, self.size, lbox)
        else:
            mask = np.zeros((self.size, self.size), dtype=np.float32)
        x = canvas.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
        x = np.transpose(x, (2, 0, 1))
        method = str(getattr(row, "method", "") or "")
        wild_cls = method in {"extra_real", "extra_fake"} or "openforensics" in str(row.path).lower()
        return {
            "image": torch.from_numpy(x.copy()),
            "mask": torch.from_numpy(mask[None].copy()),
            "label": torch.tensor([float(row.label)], dtype=torch.float32),
            "has_gt_mask": torch.tensor([1.0 if used_gt else 0.0], dtype=torch.float32),
            "score_cls": torch.tensor([1.0 if wild_cls else 0.0], dtype=torch.float32),
            "path": row.path,
            "file_name": row.file_name,
        }


class EpochSeededSampler(torch.utils.data.Sampler):
    def __init__(self, n, seed):
        self.n = int(n)
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch, start_index=0):
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1000 + self.epoch)
        order = torch.randperm(self.n, generator=g).tolist()
        return iter(order[self.start_index :])

    def __len__(self):
        return max(self.n - self.start_index, 0)


class BalancedEpochSampler(torch.utils.data.Sampler):
    # Each epoch sees every image of the larger class once and oversamples the smaller class.

    def __init__(self, labels, seed):
        self.real = [i for i, y in enumerate(labels) if int(y) == 0]
        self.fake = [i for i, y in enumerate(labels) if int(y) == 1]
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0
        if self.real and self.fake:
            self.n = 2 * max(len(self.real), len(self.fake))
            self.mode = "balanced"
        else:
            self.n = len(labels)
            self.mode = "plain"

    def set_epoch(self, epoch, start_index=0):
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1000 + self.epoch)
        if self.mode != "balanced":
            order = torch.randperm(self.n, generator=g).tolist()
            return iter(order[self.start_index :])
        def take(idxs, k):
            perm = torch.randperm(len(idxs), generator=g).tolist()
            out = []
            while len(out) < k:
                out.extend(idxs[i] for i in perm)
            return out[:k]
        half = self.n // 2
        pairs = take(self.real, half) + take(self.fake, half)
        order = torch.randperm(len(pairs), generator=g).tolist()
        mixed = [pairs[i] for i in order]
        return iter(mixed[self.start_index :])

    def __len__(self):
        return max(self.n - self.start_index, 0)


def make_loader(df, shuffle, augment):
    ds = DEFACTOFaceDataset(df, CONFIG["image_size"], augment=augment)
    if shuffle and CONFIG.get("balance_batches", True):
        sampler = BalancedEpochSampler(df.label.tolist(), CONFIG["seed"])
    elif shuffle:
        sampler = EpochSeededSampler(len(ds), CONFIG["seed"])
    else:
        sampler = None
    return DataLoader(
        ds,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(shuffle and len(ds) >= int(CONFIG["batch_size"])),
    )


TRAIN_LOADER = make_loader(INDEX["train"], True, True)
VAL_LOADER = make_loader(INDEX["val"], False, False)
TEST_LOADER = make_loader(INDEX["test"], False, False)
print("train batches", len(TRAIN_LOADER), "val", len(VAL_LOADER), "test", len(TEST_LOADER))
print("est. train minutes ~", round(len(TRAIN_LOADER) * 0.8 / 60.0, 1), "(T4, 600px B7+DeepLab)")
"""
        ),
        md("## 7. Model (EfficientNet-B7 / SRM / DeepLabV3+)"),
        cd(
            r"""
def _srm_kernels() -> torch.Tensor:
    k1 = np.array(
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    k2 = np.array(
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
        dtype=np.float32,
    )
    k3 = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    ks = []
    for k in (k1, k2, k3):
        s = np.abs(k).sum()
        ks.append(k / (s if s > 0 else 1.0))
    return torch.from_numpy(np.stack(ks, 0)[:, None, :, :])


class MultiStreamExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("srm_weight", _srm_kernels())

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(x.device, x.dtype)
        std = IMAGENET_STD.to(x.device, x.dtype)
        return (x * std + mean).clamp(0, 1)

    def forward(self, x_rgb_norm: torch.Tensor) -> torch.Tensor:
        rgb = self._denorm(x_rgb_norm)
        y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        residual = F.conv2d(y, self.srm_weight.to(dtype=rgb.dtype), padding=2).tanh()
        spec = torch.fft.fftshift(torch.fft.fft2(rgb, norm="ortho"), dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        b, c, h, w = mag.shape
        mag_flat = mag.reshape(b, c, -1)
        mn = mag_flat.min(dim=-1, keepdim=True).values
        mx = mag_flat.max(dim=-1, keepdim=True).values
        mag = ((mag_flat - mn) / (mx - mn + 1e-6)).reshape(b, c, h, w)
        return torch.cat([rgb, residual, mag], dim=1)


class FeatureFusion(nn.Module):
    def __init__(self, conv3: nn.Conv2d):
        super().__init__()
        conv9 = nn.Conv2d(
            9, conv3.out_channels, kernel_size=conv3.kernel_size,
            stride=conv3.stride, padding=conv3.padding, bias=False,
        )
        with torch.no_grad():
            w = conv3.weight
            conv9.weight[:, 0:3].copy_(w)
            conv9.weight[:, 3:6].copy_(w)
            conv9.weight[:, 6:9].copy_(w)
            conv9.weight.mul_(1.0 / 3.0)
        self.conv = conv9

    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ))
        gn = 32 if out_ch % 32 == 0 else 1
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(gn, out_ch), nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), nn.Dropout(0.1),
        )

    def forward(self, x):
        res = [b(x) for b in self.branches]
        gap = F.interpolate(self.gap(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        res.append(gap)
        return self.project(torch.cat(res, dim=1))


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, high_ch: int, low_ch: int, num_classes: int = 1):
        super().__init__()
        self.aspp = ASPP(high_ch, 256)
        self.low_proj = nn.Sequential(nn.Conv2d(low_ch, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, high, low, out_size):
        high = F.interpolate(self.aspp(high), size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([high, self.low_proj(low)], dim=1)
        return F.interpolate(self.decoder(x), size=out_size, mode="bilinear", align_corners=False)


class ClassificationHead(nn.Module):
    def __init__(self, in_ch: int, dropout: float = 0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, feat):
        return self.fc(self.pool(feat).flatten(1))


class BaselineEffNet(nn.Module):
    # Baseline B7 layout is features.* + cls_head.* so resumed checkpoints keep loading.

    def __init__(self, pretrained: bool = True, backbone: str = "efficientnet_b7"):
        super().__init__()
        if backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = efficientnet_b0(weights=weights)
            ch = 1280
        else:
            weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
            net = efficientnet_b7(weights=weights)
            ch = 2560
        self.features = net.features
        self.cls_head = ClassificationHead(ch)
        self.multi_stream = False
        self.segmentation = False

    def forward(self, x):
        feat = self.features(x)
        logits = self.cls_head(feat)
        return {"classification_logits": logits, "fake_probability": torch.sigmoid(logits)}


class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, multi_stream: bool = False):
        super().__init__()
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b7(weights=weights)
        stem = net.features[0]
        self.stem_rest = nn.Sequential(*list(stem.children())[1:])
        conv0 = stem[0]
        self.multi_stream = multi_stream
        self.stream = MultiStreamExtractor() if multi_stream else None
        if multi_stream:
            self.fusion = FeatureFusion(conv0)
            self.stem_conv = self.fusion.conv
        else:
            self.fusion = None
            self.stem_conv = conv0
        self.stages = nn.ModuleList(list(net.features.children())[1:])
        self.out_channels = 2560
        self.low_stage = 1
        self.high_stage = 4

    def forward(self, x):
        x = self.stem_conv(self.stream(x) if self.multi_stream else x)
        x = self.stem_rest(x)
        low = high = None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.low_stage:
                low = x
            if i == self.high_stage:
                high = x
        return {"final": x, "low": low if low is not None else x, "high": high if high is not None else x}


class VERITASModel(nn.Module):
    def __init__(self, backbone="efficientnet_b7", multi_stream=False, segmentation=False, pretrained=True):
        super().__init__()
        if backbone != "efficientnet_b7":
            raise ValueError("This study uses EfficientNet-B7 only")
        self.multi_stream = bool(multi_stream)
        self.segmentation = bool(segmentation)
        self.encoder = EfficientNetBackbone(pretrained=pretrained, multi_stream=self.multi_stream)
        self.cls_head = ClassificationHead(self.encoder.out_channels)
        self.seg_head = DeepLabV3PlusHead(224, 48, 1) if self.segmentation else None

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.cls_head(feats["final"])
        out = {"classification_logits": logits, "fake_probability": torch.sigmoid(logits)}
        if self.segmentation:
            seg_logits = self.seg_head(feats["high"], feats["low"], out_size=x.shape[-2:])
            out["segmentation_logits"] = seg_logits
            out["segmentation_prob"] = torch.sigmoid(seg_logits)
        return out


def build_model():
    if EXP_CFG.get("legacy_baseline"):
        return BaselineEffNet(pretrained=CONFIG["use_pretrained"], backbone=CONFIG["backbone"]).to(DEVICE)
    m = VERITASModel(
        backbone=EXP_CFG.get("backbone", "efficientnet_b7"),
        multi_stream=EXP_CFG["multi_stream"],
        segmentation=EXP_CFG["segmentation"],
        pretrained=CONFIG["use_pretrained"],
    ).to(DEVICE)
    if m.segmentation:
        was_training = m.training
        m.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64, device=DEVICE)
            feats = m.encoder(dummy)
            low_ch, high_ch = feats["low"].shape[1], feats["high"].shape[1]
        if m.seg_head.low_proj[0].in_channels != low_ch or m.seg_head.aspp.branches[0][0].in_channels != high_ch:
            m.seg_head = DeepLabV3PlusHead(high_ch, low_ch, 1).to(DEVICE)
            print(f"Adapted DeepLabV3+ head to low_ch={low_ch} high_ch={high_ch}")
        if was_training:
            m.train()
    return m


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * targets).sum(dims)
        den = probs.sum(dims) + targets.sum(dims)
        return 1.0 - ((2 * inter + self.eps) / (den + self.eps)).mean()


class VERITASLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_seg=1.0, dice_weight=1.0, segmentation=False):
        super().__init__()
        self.lambda_cls = float(lambda_cls)
        self.lambda_seg = float(lambda_seg)
        self.dice_weight = float(dice_weight)
        self.segmentation = bool(segmentation)
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, outputs, labels, masks=None, has_gt_mask=None):
        l_cls = self.bce_cls(outputs["classification_logits"], labels)
        l_seg = labels.new_zeros(())
        if self.segmentation:
            if masks is None:
                raise ValueError("masks required for segmentation experiments")
            if has_gt_mask is None:
                valid = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
            else:
                valid = has_gt_mask.reshape(labels.shape[0]) > 0.5
            if valid.any():
                sl = outputs["segmentation_logits"][valid]
                sm = masks[valid]
                l_seg = self.bce_seg(sl, sm) + self.dice_weight * self.dice(sl, sm)
        total = self.lambda_cls * l_cls + (self.lambda_seg * l_seg if self.segmentation else 0.0)
        return {
            "total": total,
            "classification": l_cls.detach(),
            "segmentation": l_seg.detach() if torch.is_tensor(l_seg) else l_seg,
        }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])
print("legacy_baseline", EXP_CFG["legacy_baseline"], "multi_stream", EXP_CFG["multi_stream"], "segmentation", EXP_CFG["segmentation"])
"""
        ),
        md("## 8. Checkpoint resume (one epoch per run)"),
        cd(
            r"""
EXP = CONFIG["experiment"]


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_progress(path: Path):
    try:
        ck = _torch_load(path)
        partial = ck.get("partial") or {}
        return (int(ck.get("epoch") or 0), int(partial.get("step", 0)))
    except Exception as e:
        print("could not read", path, e)
        return (-1, -1)


def stage_checkpoints_from_input():
    if not CONFIG.get("resume_from_input", True):
        return
    ckpt_dir = WORKING / "checkpoints"
    names = {f"{EXP}_last.pt": [], f"{EXP}_best.pt": []}
    root = Path("/kaggle/input")
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn in names:
                names[fn].append(Path(dirpath) / fn)
    for fn, paths in names.items():
        if not paths:
            continue
        dst = ckpt_dir / fn
        pool = ([dst] if dst.exists() else []) + list(paths)
        best = max(pool, key=lambda p: checkpoint_progress(p))
        if best != dst:
            shutil.copy2(best, dst)
            print("staged", best, "->", dst, checkpoint_progress(best))


def save_checkpoint(path, model, optimizer, scaler, epoch, best_metric, history, epoch_seconds, partial=None):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "partial": partial,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_metric": best_metric,
            "experiment_name": EXP,
            "history": history,
            "epoch_seconds": epoch_seconds,
            "config": {k: CONFIG[k] for k in CONFIG if k != "loss_configs"},
            "exp_cfg": EXP_CFG,
        },
        tmp,
    )
    tmp.replace(path)


def load_if_any(path, model, optimizer=None):
    path = Path(path)
    if not path.exists() or not CONFIG.get("resume", True):
        return 0, -1.0, {}
    ck = _torch_load(path)
    model.load_state_dict(ck["model_state_dict"])
    if optimizer is not None and ck.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        except Exception as e:
            print("optimizer state not loaded:", e)
    start = int(ck.get("epoch") or 0)
    best = float(ck["best_metric"]) if ck.get("best_metric") is not None else -1.0
    print("resumed", path, "epochs_completed", start, "best", best)
    return start, best, ck
"""
        ),
        md("## 9. Train / eval"),
        cd(
            r"""
def mask_metrics_from_tensors(probs, targets, labels, thresh=0.5, has_gt_mask=None):
    probs = probs.numpy() if torch.is_tensor(probs) else np.asarray(probs)
    targets = targets.numpy() if torch.is_tensor(targets) else np.asarray(targets)
    labels = np.asarray(labels).reshape(-1)
    if has_gt_mask is None:
        gt = np.ones(len(labels), dtype=bool)
    else:
        gt = np.asarray(has_gt_mask).reshape(-1) > 0.5
    ious, dices, ious_fake, dices_fake = [], [], [], []
    for i in range(len(labels)):
        if not gt[i]:
            ious.append(float("nan"))
            dices.append(float("nan"))
            continue
        p = (probs[i].reshape(-1) >= thresh).astype(np.float32)
        t = (targets[i].reshape(-1) >= 0.5).astype(np.float32)
        inter = float((p * t).sum())
        union = float(((p + t) > 0).sum())
        ps, ts = float(p.sum()), float(t.sum())
        iou = 1.0 if union == 0 else inter / union
        dice = 1.0 if (ps + ts) == 0 else (2 * inter) / (ps + ts)
        ious.append(iou)
        dices.append(dice)
        if int(labels[i]) == 1:
            ious_fake.append(iou)
            dices_fake.append(dice)
    return {
        "iou_all": ious, "dice_all": dices,
        "iou_fake": ious_fake, "dice_fake": dices_fake,
    }


@torch.no_grad()
def evaluate_loader(model, loader, split_name: str, classification_wild_only: bool = False):
    model.eval()
    y_true, y_pred, y_prob, recs = [], [], [], []
    dice_all, iou_all, dice_fake, iou_fake = [], [], [], []
    use_seg = bool(getattr(model, "segmentation", False))
    n_cls = 0
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        has_gt = batch.get("has_gt_mask")
        has_gt_np = None if has_gt is None else has_gt.detach().cpu().numpy().reshape(-1)
        out = model(images)
        prob = out["fake_probability"].detach().cpu().numpy().reshape(-1)
        pred = (prob >= CONFIG["classification_threshold"]).astype(int)
        true = labels.detach().cpu().numpy().reshape(-1).astype(int)
        score_cls = batch.get("score_cls")
        if classification_wild_only and score_cls is not None:
            cls_mask = score_cls.detach().cpu().numpy().reshape(-1) > 0.5
        else:
            cls_mask = np.ones(len(true), dtype=bool)
        n_cls += int(cls_mask.sum())
        y_true.extend(true[cls_mask].tolist())
        y_pred.extend(pred[cls_mask].tolist())
        y_prob.extend(prob[cls_mask].tolist())
        mm = None
        if use_seg and "segmentation_prob" in out:
            mm = mask_metrics_from_tensors(
                out["segmentation_prob"].detach().cpu(),
                masks.detach().cpu(),
                true,
                CONFIG["mask_threshold"],
                has_gt_np,
            )
            iou_all.extend([v for v in mm["iou_all"] if v == v])
            dice_all.extend([v for v in mm["dice_all"] if v == v])
            iou_fake.extend(mm["iou_fake"])
            dice_fake.extend(mm["dice_fake"])
        for i in range(len(true)):
            rec = {
                "file_name": batch["file_name"][i],
                "path": batch["path"][i],
                "split": split_name,
                "ground_truth_label": int(true[i]),
                "predicted_label": int(pred[i]),
                "fake_probability": float(prob[i]),
            }
            if mm is not None and mm["iou_all"][i] == mm["iou_all"][i]:
                rec["iou"] = mm["iou_all"][i]
                rec["dice"] = mm["dice_all"][i]
            recs.append(rec)
        del images, labels, masks, out
    metrics = {
        "n_classification": int(n_cls),
        "classification_set": "openforensics_wild" if classification_wild_only else split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if y_true else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if y_true else float("nan"),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if y_true else float("nan"),
    }
    if y_true:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
    if use_seg and dice_all:
        metrics["dice_all"] = float(np.mean(dice_all))
        metrics["iou_all"] = float(np.mean(iou_all))
        metrics["dice"] = float(np.mean(dice_fake) if dice_fake else float("nan"))
        metrics["miou"] = float(np.mean(iou_fake) if iou_fake else float("nan"))
        metrics["miou_note"] = "dice/miou only on images with probe_mask GT (Wild fakes skipped)"
    return metrics, recs


def train_one_experiment():
    model = build_model()
    use_seg = bool(getattr(model, "segmentation", False))
    criterion = VERITASLoss(
        CONFIG["lambda_cls"], CONFIG["lambda_seg"], CONFIG["dice_weight"], use_seg
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    total_epochs = max(int(CONFIG["epochs"]), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 0.5 * (1.0 + math.cos(math.pi * min(e, total_epochs) / total_epochs))
    )
    stage_checkpoints_from_input()
    ckpt_best = WORKING / "checkpoints" / f"{EXP}_best.pt"
    ckpt_last = WORKING / "checkpoints" / f"{EXP}_last.pt"
    resume_path = ckpt_last if ckpt_last.exists() else ckpt_best
    start_epoch, best_metric, resumed = load_if_any(resume_path, model, optimizer)
    for _ in range(start_epoch):
        scheduler.step()
    history = list((resumed or {}).get("history", []))
    epoch_seconds = list((resumed or {}).get("epoch_seconds", []))
    partial = (resumed or {}).get("partial") if resume_path == ckpt_last else None
    scaler_state = (resumed or {}).get("scaler_state_dict")
    best_metric = -1.0 if best_metric is None else float(best_metric)
    amp_ok = bool(CONFIG["use_amp"] and DEVICE.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_ok)
        autocast = lambda: torch.amp.autocast("cuda", enabled=amp_ok)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_ok)
        autocast = lambda: torch.cuda.amp.autocast(enabled=amp_ok)
    if scaler_state and amp_ok:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as e:
            print("scaler not loaded:", e)

    per_run = CONFIG.get("epochs_per_run")
    stop_after = total_epochs if not per_run else min(total_epochs, start_epoch + int(per_run))
    print(f"START OF RUN | {EXP}: {start_epoch}/{total_epochs} epochs completed")
    print(f"This run trains epoch {start_epoch + 1} to epoch {stop_after}" if start_epoch < stop_after else "No new epoch this run")
    print(f"device={DEVICE} torch={torch.__version__} train_batches={len(TRAIN_LOADER)}")
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA. Full DEFACTO will not finish an epoch in 12h.")
    (WORKING / "logs" / f"{EXP}_training_progress.json").write_text(
        json.dumps({"experiment": EXP, "epochs_completed": start_epoch, "epochs_total": total_epochs}, indent=2),
        encoding="utf-8",
    )
    if CONFIG["skip_training"]:
        if ckpt_best.exists():
            load_if_any(ckpt_best, model)
        return model, start_epoch, best_metric, history

    budget_s = float(CONFIG.get("time_budget_hours") or 1e9) * 3600.0
    elapsed = lambda: time.time() - NOTEBOOK_START_TIME
    accum = max(int(CONFIG["gradient_accumulation"]), 1)
    mid_min = CONFIG.get("mid_epoch_checkpoint_minutes")
    mid_s = float(mid_min) * 60.0 if mid_min else None
    sampler = getattr(TRAIN_LOADER, "sampler", None)

    for epoch in range(start_epoch, stop_after):
        start_step = 0
        if partial is not None and int(partial.get("epoch", -1)) == epoch and hasattr(sampler, "set_epoch"):
            start_step = int(partial["step"])
            print(f"Continuing epoch {epoch + 1} from step {start_step}/{len(TRAIN_LOADER)}")
        partial = None
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch, start_step * int(CONFIG["batch_size"]))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, seen, t0, last_save = 0.0, 0, time.time(), time.time()
        it = tqdm(TRAIN_LOADER, desc=f"{EXP} epoch {epoch + 1}", leave=True)
        for i, batch in enumerate(it):
            step = start_step + i
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)
            has_gt = batch.get("has_gt_mask")
            if has_gt is not None:
                has_gt = has_gt.to(DEVICE, non_blocking=True)
            with autocast():
                out = model(images)
                losses = criterion(out, labels, masks if use_seg else None, has_gt if use_seg else None)
                loss = losses["total"] / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item()) * accum * images.size(0)
            seen += images.size(0)
            if (i + 1) % 10 == 0:
                it.set_postfix(loss=f"{float(loss.item()) * accum:.4f}", step=step + 1)
            if mid_s and (time.time() - last_save) >= mid_s:
                save_checkpoint(
                    ckpt_last, model, optimizer, scaler, epoch, best_metric, history, epoch_seconds,
                    partial={"epoch": epoch, "step": step + 1},
                )
                last_save = time.time()
                print(f"mid-epoch save epoch {epoch + 1} step {step + 1}/{len(TRAIN_LOADER)}")
            if elapsed() > budget_s:
                save_checkpoint(
                    ckpt_last, model, optimizer, scaler, epoch, best_metric, history, epoch_seconds,
                    partial={"epoch": epoch, "step": step + 1},
                )
                print("time budget hit; saved mid-epoch and stopping")
                (WORKING / "logs" / f"{EXP}_training_progress.json").write_text(
                    json.dumps({"experiment": EXP, "epochs_completed": epoch, "epochs_total": total_epochs, "partial_step": step + 1}, indent=2),
                    encoding="utf-8",
                )
                return model, epoch, best_metric, history
            del images, labels, masks, out, loss
        scheduler.step()
        train_loss = running / max(seen, 1)
        val_metrics, _ = evaluate_loader(model, VAL_LOADER, "val")
        hist = {"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(hist)
        epoch_seconds.append(time.time() - t0)
        print("END OF EPOCH", hist)
        score = float(val_metrics["f1"])
        if score >= best_metric:
            best_metric = score
            save_checkpoint(ckpt_best, model, optimizer, scaler, epoch + 1, best_metric, history, epoch_seconds)
            print("new best val f1", best_metric)
        save_checkpoint(ckpt_last, model, optimizer, scaler, epoch + 1, best_metric, history, epoch_seconds)
        (WORKING / "logs" / f"{EXP}_training_progress.json").write_text(
            json.dumps({"experiment": EXP, "epochs_completed": epoch + 1, "epochs_total": total_epochs, "best_val_f1": best_metric}, indent=2),
            encoding="utf-8",
        )
        print(f"END OF RUN | {EXP}: {epoch + 1}/{total_epochs} epochs completed")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model, stop_after, best_metric, history


MODEL, EPOCHS_DONE, BEST, HISTORY = train_one_experiment()
print("epochs done", EPOCHS_DONE, "best val f1", BEST)
"""
        ),
        md(
            """
## 10. Test (only after the last planned epoch)

**Classification** is OpenForensics Wild Test only (`extra_real` / `extra_fake`). DEFACTO reference, swap, and morph images are not in the accuracy or F1.

**Segmentation** is unchanged: dice and IoU use DEFACTO `probe_mask` only. Wild has no mask, so it does not enter those scores.
"""
        ),
        cd(
            r"""
done = EPOCHS_DONE >= int(CONFIG["epochs"])
if CONFIG.get("test_eval_when", "final") == "final" and not done:
    print(f"test_eval_when='final': {EPOCHS_DONE}/{CONFIG['epochs']} epochs, skipping official Test this run.")
else:
    ckpt_best = WORKING / "checkpoints" / f"{EXP}_best.pt"
    if ckpt_best.exists():
        load_if_any(ckpt_best, MODEL)
    metrics, recs = evaluate_loader(MODEL, TEST_LOADER, "test", classification_wild_only=bool(CONFIG.get("classification_wild_only", True)))
    cls_keys = ("n_classification", "accuracy", "f1", "precision", "recall", "tp", "tn", "fp", "fn")
    seg_keys = ("dice", "miou", "dice_all", "iou_all", "miou_note")
    print("TEST classification (OpenForensics Wild only)", {k: metrics[k] for k in cls_keys if k in metrics})
    print("TEST segmentation (DEFACTO probe_mask only)", {k: metrics[k] for k in seg_keys if k in metrics})
    if not metrics.get("n_classification"):
        print("WARNING: Wild Test is empty, so classification accuracy was not computed. Attach OpenForensics Wild.")
    pred_path = WORKING / "predictions" / f"{EXP}_predictions.csv"
    pd.DataFrame(recs).to_csv(pred_path, index=False)
    (WORKING / "logs" / f"{EXP}_test_metrics.json").write_text(
        json.dumps({"metrics": metrics, "epochs_completed": EPOCHS_DONE, "exp_cfg": EXP_CFG}, indent=2),
        encoding="utf-8",
    )
    row = {"experiment": EXP, "status": "ok", "epochs_completed": EPOCHS_DONE, **{k: v for k, v in metrics.items() if k != "miou_note"}}
    pd.DataFrame([row]).to_csv(WORKING / "experiment_results.csv", index=False)
    print("wrote", pred_path)
"""
        ),
        md(
            f"""
## Notes

- Training uses the DEFACTO hybrid (DEFACTO Face + OpenForensics Wild).
- {("**Test classification counts the mixed test.** Up to 2,000 reals, 2,000 DEFACTO fakes, and 2,000 Wild fakes." if not v.get("classification_wild_only", True) else "**Test classification is Wild only.** Accuracy and F1 ignore DEFACTO reference, swap, and morph.")}
- **Test segmentation is DEFACTO `probe_mask` only.** That restriction does not apply to localization. Wild fakes skip seg loss and dice.
- Reals: DEFACTO `reference/` plus Wild **Real**. Class-balanced batches.
- New protocol — do not resume the F1=1.0 DEFACTO-vs-Wild-Real checkpoint.
- First full run starts from ImageNet B7. Later runs attach **this notebook’s** latest output.
- All four DEFACTO notebooks must print the same `PROTOCOL_HASH`.
- `OF_01_baseline` is the hybrid baseline: same DEFACTO Face + OpenForensics Wild mix as `OF_04_full`, RGB only. Compare loads `baseline_effb7_defacto_best.pt`, not the Wild-only checkpoint.
"""
        ),
    ]
    return _nb(cells)


def build_compare_nb(openforensics_only=False):
    cells = [
        md(
            r"""
# VERITAS OpenForensics — Compare / cross-dataset Test

This notebook does **not** train. It scores checkpoints on **every attached image test set**
it can recognize (OpenForensics Wild, DEFACTO Face, FaceForensics++ frames/videos,
ForgeryNet-style `Test/0_real`+`1_fake`, or any `Test/Real`+`Test/Fake`).

**Classification test = OpenForensics Wild only.** Accuracy and F1 ignore DEFACTO and FaceForensics.
**Segmentation test = DEFACTO `probe_mask` only.** That restriction does not apply to localization. Wild has no mask, so it does not enter dice.

## Kaggle (3-epoch Full model)

1. New GPU notebook. Import `OF_05_compare.ipynb`.
2. Accelerator **GPU T4**.
3. **Add Input → Notebooks** → `OF_01_baseline` and `OF_04_full` (both trained on the DEFACTO hybrid: DEFACTO Face + OpenForensics Wild).  
   The baseline file is `baseline_effb7_defacto_best.pt`. The Full file is `full_veritas_defacto_best.pt` or `.zip`.  
   Do **not** use `baseline_effb7_wild_best.pt` here. That checkpoint was trained on Wild only.
4. **Add Input → Datasets** for each test set you want:
   - OpenForensics Wild: `ahmedgamal/openforensics-wild-dataset`
   - DEFACTO Face: `defactodataset/defactoface`
   - FaceForensics++ / ForgeryNet: attach that dump if it has images or videos
5. Save & Run All. The log prints **CLASSIFICATION** accuracy and F1 for OpenForensics Wild only, and **SEGMENTATION** dice for DEFACTO `probe_mask` only. FaceForensics is written to a predictions file and is not the classification test.

Locally, put the zip at `models/full_veritas_defacto_best.zip` and attach folders under this repo.

- `baseline_effb7_defacto` — RGB EfficientNet-B7 trained on the **same DEFACTO hybrid** as Full VERITAS (`OF_01_baseline`)
- `baseline_effb7_wild_best.pt` is ignored. It is not this baseline.
- `multistream_no_seg_defacto` (optional)
- `seg_no_multistream_defacto` (optional)
- `full_veritas_defacto` (the app / 3-epoch model)
"""
        ),
        md(architecture_md(
            "**This notebook does not train.** The baseline row is `OF_01`: RGB EfficientNet-B7 trained on the DEFACTO hybrid (DEFACTO Face plus OpenForensics Wild Real and Fake). "
            "Full VERITAS is `OF_04` on that same hybrid. The extra stem and DeepLab head are why VERITAS takes longer per image; the timing cell reports that gap."
        )),
        cd(
            r"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())

SEARCH_ROOTS = []
if Path("/kaggle/input").exists():
    SEARCH_ROOTS.append(Path("/kaggle/input"))
if Path("/kaggle/working").exists():
    SEARCH_ROOTS.append(Path("/kaggle/working"))
SEARCH_ROOTS.append(Path.cwd())
SEARCH_ROOTS.append(Path.cwd() / "kaggle_working")
if (Path.cwd() / "models").is_dir():
    SEARCH_ROOTS.append(Path.cwd() / "models")
if (Path.cwd().parent / "models").is_dir():
    SEARCH_ROOTS.append(Path.cwd().parent / "models")

EXPECTED = [
    "baseline_effb7_defacto",
    "multistream_no_seg_defacto",
    "seg_no_multistream_defacto",
    "full_veritas_defacto",
]
print("search roots:", [str(p) for p in SEARCH_ROOTS if p.exists()])
"""
        ),
        cd(
            r"""
# Image folders are huge on the Kaggle mount. Never walk them while looking for weights.
SKIP_DIRS = {
    "swapping_img", "morphing_img", "reference",
    "swapping_annotations", "morphing_annotations",
    "original_sequences", "manipulated_sequences",
    "probe_mask", "donor_mask", "inpaint_mask",
    "train", "validation", "val", "test", "testing",
    "real", "fake", "0_real", "1_fake", "authentic", "tampered",
    "img", "images", "masks", "youtube", "actors",
    "deepfakes", "face2face", "faceswap", "neuraltextures",
    "deepfakedetection", "faceshifter", "face_shifter",
    "downloaded_videos", "c0", "c23", "c40",
}

def _index_files():
    hits = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        print("indexing", root, flush=True)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
            for fn in filenames:
                low = fn.lower()
                if low.endswith((".pt", ".zip", ".csv", ".json")):
                    hits.append(Path(dirpath) / fn)
    print("indexed", len(hits), "checkpoint-like files", flush=True)
    return hits

INDEXED = _index_files()

def find_files(name_substr: str, suffix: str):
    return sorted({p for p in INDEXED if p.suffix.lower() == suffix and name_substr in p.name})


# Wild-only EfficientNet. The hybrid baseline is baseline_effb7_defacto from OF_01.
WILD_ONLY_BASELINE = "baseline_effb7_wild_best.pt"

def resolve_ckpt(exp):
    lasts = [p for p in find_files(f"{exp}_last", ".pt") if p.name == f"{exp}_last.pt" and p.name != WILD_ONLY_BASELINE]
    bests = [p for p in find_files(f"{exp}_best", ".pt") if p.name == f"{exp}_best.pt" and p.name != WILD_ONLY_BASELINE]
    zips = [p for p in find_files(f"{exp}_best", ".zip") if exp in p.name and p.suffix.lower() == ".zip" and p.name != WILD_ONLY_BASELINE]
    hit = (bests[0] if bests else None) or (lasts[0] if lasts else None) or (zips[0] if zips else None)
    if hit is not None and exp == "baseline_effb7_defacto":
        print("hybrid baseline:", hit)
    return hit

if any(p.name == WILD_ONLY_BASELINE for p in INDEXED):
    print("ignoring", WILD_ONLY_BASELINE, "(Wild-only). Baseline is baseline_effb7_defacto_best.pt from OF_01.")

pred_files = {}
ckpt_files = {}
for exp in EXPECTED:
    cands = find_files(f"{exp}_predictions", ".csv")
    pred_files[exp] = cands[0] if cands else None
    ckpt_files[exp] = resolve_ckpt(exp)
    print(exp, "predictions:", pred_files[exp])
    print(exp, "checkpoint:", ckpt_files[exp])

protocol_files = find_files("protocol", ".json")
print("protocol files", [str(p) for p in protocol_files])
NEED_INFER = [e for e in EXPECTED if pred_files[e] is None and ckpt_files[e] is not None]
print("will score Test from checkpoint:", NEED_INFER)
"""
        ),
        cd(
            r"""
protocols = []
for p in protocol_files:
    try:
        protocols.append((p, json.loads(p.read_text(encoding="utf-8"))))
    except Exception as e:
        print("skip protocol", p, e)

if protocols:
    hashes = {obj.get("hash") for _, obj in protocols}
    print("PROTOCOL hashes found:", hashes)
    if len(hashes) > 1:
        print("WARNING: protocol hashes differ. Do not treat metrics as a controlled ablation.")
        for p, obj in protocols:
            print(" ", p, obj.get("hash"), obj.get("experiment"))
    else:
        print("OK: all protocol.json files share the same hash.")
else:
    print("No protocol.json found. Comparison can still run if prediction file names match.")
"""
        ),
        md("## Score official Test from epoch-1 checkpoints (only if a CSV is missing)"),
        cd(
            r"""
import hashlib
import json as _json
import os
import time
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7
from tqdm.auto import tqdm

META = {
    "baseline_effb7_defacto": {"multi_stream": False, "segmentation": False, "legacy": True},
    "multistream_no_seg_defacto": {"multi_stream": True, "segmentation": False, "legacy": False},
    "seg_no_multistream_defacto": {"multi_stream": False, "segmentation": True, "legacy": False},
    "full_veritas_defacto": {"multi_stream": True, "segmentation": True, "legacy": False},
}
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working"
(OUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def materialize_checkpoint(path: Path) -> Path:
    path = Path(path)
    try:
        _torch_load(path)
        return path
    except Exception:
        pass
    if path.suffix.lower() != ".zip":
        raise RuntimeError(f"Cannot read checkpoint: {path}")
    import zipfile
    cache = path.with_name(path.stem.replace(" ", "_") + "_repacked.pt")
    if cache.exists() and cache.stat().st_size > 1_000_000:
        return cache
    with zipfile.ZipFile(path) as zin:
        names = zin.namelist()
        pkl = next((n for n in names if n.endswith("data.pkl")), None)
        if pkl is None:
            raise RuntimeError("Zip is not a PyTorch checkpoint (no data.pkl)")
        prefix = pkl[: -len("data.pkl")]
        with zipfile.ZipFile(cache, "w", compression=zipfile.ZIP_STORED) as zout:
            for name in names:
                if prefix and not name.startswith(prefix):
                    continue
                inner = name[len(prefix) :]
                if not inner or inner.endswith("/"):
                    continue
                zout.writestr(inner, zin.read(name))
    return cache


EXTS_IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXTS_VID = {".mp4", ".avi", ".mov", ".mkv"}
MAX_TEST_PER_CLASS = 2000


def _data_roots():
    roots = []
    if Path("/kaggle/input").exists():
        roots.append(Path("/kaggle/input"))
    for p in (
        Path.cwd() / "defacto-face",
        Path.cwd() / "openforensics-wild-dataset",
        Path.cwd() / "dataset",
        Path.cwd() / "Dataset",
        Path.cwd() / "faceforensics",
        Path.cwd() / "forgerynet",
    ):
        if p.is_dir():
            roots.append(p)
    return roots


def discover_defacto_root():
    for root in _data_roots():
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            kids = {d.lower() for d in dirnames}
            if "swapping_img" in kids or "morphing_img" in kids:
                return p
    return None


def _dir(root, name):
    if root is None or not Path(root).is_dir():
        return None
    root = Path(root)
    if (root / name).is_dir():
        return root / name
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None


def _as_img_dir(p):
    if p is None or not Path(p).is_dir():
        return None
    p = Path(p)
    return p / "img" if (p / "img").is_dir() else p


def _norm_stem(stem: str) -> str:
    s = stem.lower()
    for tok in ("_probe_mask", "_probemask", "_probe", "_mask", "_gt", "-mask"):
        s = s.replace(tok, "")
    return s


def _id_key(stem: str) -> str:
    digits = "".join(ch for ch in _norm_stem(stem) if ch.isdigit())
    return digits.lstrip("0") or digits


def index_mask_dir(mask_root):
    idx = {}
    if mask_root is None or not Path(mask_root).is_dir():
        return idx
    mask_root = Path(mask_root)
    skip_meta = {".json", ".txt", ".xml", ".csv", ".md"}
    files = [p for p in mask_root.rglob("*") if p.is_file()]
    image_files = [p for p in files if p.suffix.lower() in EXTS_IMG] or [p for p in files if p.suffix.lower() not in skip_meta]
    for p in image_files:
        idx[p.stem] = p
        idx[p.stem.lower()] = p
        idx[_norm_stem(p.stem)] = p
        kid = _id_key(p.stem)
        if kid:
            idx[kid] = p
        parent = p.parent
        if parent != mask_root and parent.name.lower() not in {"img", "probe_mask", "masks", "images"}:
            idx[parent.name] = p
            idx[parent.name.lower()] = p
    return idx


def lookup_mask(image_path: Path, img_dir, mask_idx):
    keys = [image_path.stem, image_path.stem.lower(), _norm_stem(image_path.stem), _id_key(image_path.stem)]
    try:
        rel = image_path.relative_to(img_dir).with_suffix("").as_posix().lower()
        keys.extend([rel, Path(rel).name])
    except ValueError:
        pass
    for k in keys:
        if k and k in mask_idx:
            return mask_idx[k]
    return None


def assign_split(key: str, seed=42) -> str:
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < 0.70:
        return "train"
    if x < 0.85:
        return "val"
    return "test"


def list_test(root: Path, seed=42) -> pd.DataFrame:
    rows = []
    swap_img = _as_img_dir(_dir(root, "swapping_img"))
    morph_img = _as_img_dir(_dir(root, "morphing_img"))
    ref_img = _as_img_dir(_dir(root, "reference"))
    swap_mask = _as_img_dir(_dir(_dir(root, "swapping_annotations"), "probe_mask"))
    morph_mask = _as_img_dir(_dir(_dir(root, "morphing_annotations"), "probe_mask"))
    sources = []
    if swap_img is not None:
        sources.append((swap_img, swap_mask))
    if morph_img is not None:
        sources.append((morph_img, morph_mask))
    for img_dir, mask_dir in sources:
        imgs = sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS_IMG])
        mask_idx = index_mask_dir(mask_dir)
        paired = []
        unmatched = []
        for p in imgs:
            if assign_split(p.stem, seed) != "test":
                continue
            mp = lookup_mask(p, img_dir, mask_idx)
            if mp is None:
                unmatched.append(p)
            else:
                paired.append((p, mp))
        if not paired and unmatched and mask_dir is not None:
            masks = sorted([p for p in Path(mask_dir).rglob("*") if p.is_file() and p.suffix.lower() in EXTS_IMG])
            if len(masks) == len(imgs):
                paired = [(p, m) for p, m in zip(imgs, masks) if assign_split(p.stem, seed) == "test"]
                unmatched = []
        for p, mp in paired:
            rows.append({"path": str(p), "mask_path": str(mp), "label": 1, "frame_idx": 0, "mask_frame_idx": 0, "file_name": p.name})
    if ref_img is not None:
        for p in sorted(ref_img.rglob("*")):
            if not (p.is_file() and p.suffix.lower() in EXTS_IMG):
                continue
            if assign_split("real:" + p.stem, seed) != "test":
                continue
            rows.append({"path": str(p), "mask_path": "", "label": 0, "frame_idx": 0, "mask_frame_idx": 0, "file_name": p.name})
    inp = Path("/kaggle/input")
    if inp.is_dir():
        for child in sorted(inp.iterdir()):
            if not child.is_dir() or "defacto" in child.name.lower():
                continue
            for dirpath, dirnames, _ in os.walk(child):
                p = Path(dirpath)
                if p.parent.name.lower() != "test":
                    continue
                lab = 0 if p.name.lower() == "real" else 1 if p.name.lower() == "fake" else None
                if lab is None:
                    continue
                for img in sorted([q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in EXTS_IMG]):
                    rows.append({"path": str(img), "mask_path": "", "label": lab, "frame_idx": 0, "mask_frame_idx": 0, "file_name": img.name})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    rng = np.random.RandomState(seed + 23)
    real = df[df.label == 0]
    gt_fake = df[(df.label == 1) & (df.mask_path.fillna("") != "")]
    extra_fake = df[(df.label == 1) & (df.mask_path.fillna("") == "")]
    if len(real) > 2000:
        real = real.sample(n=2000, random_state=rng)
    if len(gt_fake) > 2000:
        gt_fake = gt_fake.sample(n=2000, random_state=rng)
    if len(extra_fake) > 2000:
        extra_fake = extra_fake.sample(n=2000, random_state=rng)
    return pd.concat([real, gt_fake, extra_fake]).reset_index(drop=True)


def _media_rows(folder, label, dataset, max_n, mask_dir=None):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    imgs = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS_IMG])
    vids = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS_VID])
    files = imgs if imgs else vids
    if max_n and len(files) > max_n:
        rng = np.random.RandomState(42)
        files = [files[i] for i in rng.choice(len(files), size=max_n, replace=False)]
    rows = []
    mask_idx = index_mask_dir(mask_dir) if mask_dir is not None else {}
    for p in files:
        mp = lookup_mask(p, folder, mask_idx) if mask_idx else None
        rows.append(
            {
                "path": str(p),
                "mask_path": str(mp) if mp is not None else "",
                "label": int(label),
                "frame_idx": 0,
                "mask_frame_idx": 0,
                "file_name": p.name,
                "dataset": dataset,
            }
        )
    return rows


def discover_test_sets():
    found = []
    seen = set()

    def add(name, rows):
        df = pd.DataFrame(rows)
        if df.empty:
            return
        n_fake = int((df.label == 1).sum())
        n_real = int((df.label == 0).sum())
        if n_fake == 0 or n_real == 0:
            print("skip", name, "missing a class  fake", n_fake, "real", n_real)
            return
        found.append((name, df.reset_index(drop=True)))
        print("TEST SET", name, "n=", len(df), "fake=", n_fake, "real=", n_real, "gt_masks=", int((df.mask_path.fillna("") != "").sum()))

    droot = discover_defacto_root()
    if droot is not None:
        df = list_test(droot, seed=42)
        if not df.empty:
            df = df[~df["path"].astype(str).str.contains("openforensics", case=False, na=False)].copy()
            df["dataset"] = "defacto_face"
            add("defacto_face", df.to_dict("records"))

    for root in _data_roots():
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            kids = {d.lower(): p / d for d in dirnames}
            key = str(p.resolve())
            if key in seen:
                continue
            split_ok = p.name.lower() in {"test", "testing", "validation", "val", "test-challenge"}
            real = kids.get("real") or kids.get("0_real") or kids.get("authentic")
            fake = kids.get("fake") or kids.get("1_fake") or kids.get("tampered")
            if split_ok and real is not None and fake is not None:
                seen.add(key)
                tag = "openforensics_wild" if "openforensics" in str(p).lower() or "wild" in str(p).lower() else p.parent.name.lower() + "_" + p.name.lower()
                add(tag, _media_rows(real, 0, tag, MAX_TEST_PER_CLASS) + _media_rows(fake, 1, tag, MAX_TEST_PER_CLASS))
            if "original_sequences" in kids and "manipulated_sequences" in kids:
                seen.add(key)
                tag = "faceforensics"
                add(
                    tag,
                    _media_rows(kids["original_sequences"], 0, tag, MAX_TEST_PER_CLASS)
                    + _media_rows(kids["manipulated_sequences"], 1, tag, MAX_TEST_PER_CLASS),
                )
            if "youtube" in kids and any(k in kids for k in ("deepfakes", "face2face", "faceswap", "neuraltextures")):
                seen.add(key)
                tag = "faceforensics"
                rows = _media_rows(kids["youtube"], 0, tag, MAX_TEST_PER_CLASS)
                for fk in ("deepfakes", "face2face", "faceswap", "neuraltextures", "face_shifter", "faceshifter"):
                    if fk in kids:
                        rows.extend(_media_rows(kids[fk], 1, tag, MAX_TEST_PER_CLASS // 4))
                add(tag, rows)
    return found


IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def letterbox_rgb(image_rgb, size):
    h, w = image_rgb.shape[:2]
    scale = min(size / max(h, 1), size / max(w, 1))
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, (left, top, nw, nh)


def letterbox_mask(mask, size, box):
    left, top, nw, nh = box
    canvas = np.zeros((size, size), dtype=np.float32)
    if mask is None or mask.size == 0:
        return canvas
    m = mask.astype(np.float32)
    m = (m > 127).astype(np.float32) if m.max() > 1.0 else (m > 0.5).astype(np.float32)
    canvas[top : top + nh, left : left + nw] = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return canvas


try:
    _HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception:
    _HAAR = cv2.CascadeClassifier()


def read_frame(path: Path, idx: int):
    if path.is_file() and path.suffix.lower() in EXTS_IMG:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if path.is_dir():
        frames = sorted([c for c in path.iterdir() if c.is_file() and c.suffix.lower() in EXTS_IMG])
        if not frames:
            return None
        idx = int(np.clip(idx, 0, len(frames) - 1))
        bgr = cv2.imread(str(frames[idx]), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(idx, 0)))
    ok, bgr = cap.read()
    cap.release()
    return None if (not ok or bgr is None) else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bbox_from_mask(mask, margin):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = int(round(max(x1 - x0, 1) * margin))
    pad_y = int(round(max(y1 - y0, 1) * margin))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y)


class DEFACTOFaceDataset(Dataset):
    def __init__(self, df, size, margin=0.10):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.margin = float(margin)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = read_frame(Path(row.path), int(row.frame_idx))
        if rgb is None:
            rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        is_fake = int(row.label) == 1
        mask_np = None
        if is_fake and str(getattr(row, "mask_path", "") or ""):
            raw = read_frame(Path(row.mask_path), int(row.mask_frame_idx if "mask_frame_idx" in row.index else row.frame_idx))
            if raw is not None:
                mask_np = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY) if raw.ndim == 3 else raw
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        faces = _HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)) if not _HAAR.empty() else []
        n_faces = int(len(faces))
        if mask_np is not None:
            mask_bin = (mask_np > 127).astype(np.uint8) if mask_np.max() > 1 else (mask_np > 0).astype(np.uint8)
            box = bbox_from_mask(mask_bin, self.margin)
        else:
            mask_bin = np.zeros(rgb.shape[:2], dtype=np.uint8)
            if n_faces:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                dummy = np.zeros(rgb.shape[:2], dtype=np.uint8)
                dummy[y : y + h, x : x + w] = 1
                box = bbox_from_mask(dummy, self.margin)
            else:
                box = None
        if box is None:
            box = (0, 0, rgb.shape[1], rgb.shape[0])
        if not is_fake:
            mask_bin = np.zeros(rgb.shape[:2], dtype=np.uint8)
        x0, y0, x1, y1 = box
        crop, mcrop = rgb[y0:y1, x0:x1], mask_bin[y0:y1, x0:x1]
        canvas, lbox = letterbox_rgb(crop, self.size)
        mask = letterbox_mask(mcrop, self.size, lbox) if mask_np is not None else np.zeros((self.size, self.size), dtype=np.float32)
        x = np.transpose((canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN_NP) / IMAGENET_STD_NP, (2, 0, 1))
        return {
            "image": torch.from_numpy(x.copy()),
            "mask": torch.from_numpy(mask[None].copy()),
            "label": torch.tensor([float(row.label)], dtype=torch.float32),
            "has_gt_mask": torch.tensor([1.0 if mask_np is not None else 0.0], dtype=torch.float32),
            "path": row.path,
            "file_name": row.file_name,
            "detected_faces": n_faces,
        }


def _srm_kernels():
    k1 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    k2 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32)
    k3 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    ks = []
    for k in (k1, k2, k3):
        s = np.abs(k).sum()
        ks.append(k / (s if s > 0 else 1.0))
    return torch.from_numpy(np.stack(ks, 0)[:, None, :, :])


class MultiStreamExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("srm_weight", _srm_kernels())

    def forward(self, x_rgb_norm):
        mean, std = IMAGENET_MEAN.to(x_rgb_norm.device, x_rgb_norm.dtype), IMAGENET_STD.to(x_rgb_norm.device, x_rgb_norm.dtype)
        rgb = (x_rgb_norm * std + mean).clamp(0, 1)
        y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        residual = F.conv2d(y, self.srm_weight.to(dtype=rgb.dtype), padding=2).tanh()
        spec = torch.fft.fftshift(torch.fft.fft2(rgb, norm="ortho"), dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        b, c, h, w = mag.shape
        mag_flat = mag.reshape(b, c, -1)
        mn, mx = mag_flat.min(dim=-1, keepdim=True).values, mag_flat.max(dim=-1, keepdim=True).values
        mag = ((mag_flat - mn) / (mx - mn + 1e-6)).reshape(b, c, h, w)
        return torch.cat([rgb, residual, mag], dim=1)


class FeatureFusion(nn.Module):
    def __init__(self, conv3):
        super().__init__()
        conv9 = nn.Conv2d(9, conv3.out_channels, kernel_size=conv3.kernel_size, stride=conv3.stride, padding=conv3.padding, bias=False)
        with torch.no_grad():
            w = conv3.weight
            conv9.weight[:, 0:3].copy_(w)
            conv9.weight[:, 3:6].copy_(w)
            conv9.weight[:, 6:9].copy_(w)
            conv9.weight.mul_(1.0 / 3.0)
        self.conv = conv9

    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
        for r in rates:
            self.branches.append(nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
        gn = 32 if out_ch % 32 == 0 else 1
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.GroupNorm(gn, out_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        res = [b(x) for b in self.branches]
        gap = F.interpolate(self.gap(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        res.append(gap)
        return self.project(torch.cat(res, dim=1))


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, high_ch, low_ch, num_classes=1):
        super().__init__()
        self.aspp = ASPP(high_ch, 256)
        self.low_proj = nn.Sequential(nn.Conv2d(low_ch, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, high, low, out_size):
        high = F.interpolate(self.aspp(high), size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([high, self.low_proj(low)], dim=1)
        return F.interpolate(self.decoder(x), size=out_size, mode="bilinear", align_corners=False)


class ClassificationHead(nn.Module):
    def __init__(self, in_ch, dropout=0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, feat):
        return self.fc(self.pool(feat).flatten(1))


class BaselineEffNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = efficientnet_b7(weights=None)
        self.features = net.features
        self.cls_head = ClassificationHead(2560)
        self.multi_stream = False
        self.segmentation = False

    def forward(self, x):
        logits = self.cls_head(self.features(x))
        return {"classification_logits": logits, "fake_probability": torch.sigmoid(logits)}


class EfficientNetBackbone(nn.Module):
    def __init__(self, multi_stream=False):
        super().__init__()
        net = efficientnet_b7(weights=None)
        stem = net.features[0]
        self.stem_rest = nn.Sequential(*list(stem.children())[1:])
        conv0 = stem[0]
        self.multi_stream = multi_stream
        self.stream = MultiStreamExtractor() if multi_stream else None
        if multi_stream:
            self.fusion = FeatureFusion(conv0)
            self.stem_conv = self.fusion.conv
        else:
            self.fusion = None
            self.stem_conv = conv0
        self.stages = nn.ModuleList(list(net.features.children())[1:])
        self.out_channels = 2560
        self.low_stage, self.high_stage = 1, 4

    def forward(self, x):
        x = self.stem_conv(self.stream(x) if self.multi_stream else x)
        x = self.stem_rest(x)
        low = high = None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.low_stage:
                low = x
            if i == self.high_stage:
                high = x
        return {"final": x, "low": low if low is not None else x, "high": high if high is not None else x}


class VERITASModel(nn.Module):
    def __init__(self, multi_stream=False, segmentation=False):
        super().__init__()
        self.multi_stream = bool(multi_stream)
        self.segmentation = bool(segmentation)
        self.encoder = EfficientNetBackbone(multi_stream=self.multi_stream)
        self.cls_head = ClassificationHead(self.encoder.out_channels)
        self.seg_head = DeepLabV3PlusHead(224, 48, 1) if self.segmentation else None

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.cls_head(feats["final"])
        out = {"classification_logits": logits, "fake_probability": torch.sigmoid(logits)}
        if self.segmentation:
            seg_logits = self.seg_head(feats["high"], feats["low"], out_size=x.shape[-2:])
            out["segmentation_logits"] = seg_logits
            out["segmentation_prob"] = torch.sigmoid(seg_logits)
        return out


def build_eval_model(exp):
    cfg = META[exp]
    if cfg["legacy"]:
        return BaselineEffNet()
    return VERITASModel(multi_stream=cfg["multi_stream"], segmentation=cfg["segmentation"])


@torch.no_grad()
def score_loader(model, loader, device, thresh=0.5):
    model.eval()
    recs = []
    for batch in tqdm(loader, desc="test"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].numpy().reshape(-1).astype(int)
        detected = batch["detected_faces"].numpy().reshape(-1).astype(int) if "detected_faces" in batch else np.zeros(len(labels), dtype=int)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_ms = (time.perf_counter() - t0) * 1000.0 / max(len(labels), 1)
        prob = out["fake_probability"].detach().cpu().numpy().reshape(-1)
        pred = (prob >= thresh).astype(int)
        seg = out.get("segmentation_prob")
        gt_flag = batch["has_gt_mask"].numpy().reshape(-1) if "has_gt_mask" in batch else np.zeros(len(labels))
        gt_masks = batch["mask"].numpy() if seg is not None and "mask" in batch else None
        seg_np = seg.detach().cpu().numpy() if seg is not None else None
        for i in range(len(labels)):
            rec = {
                "file_name": batch["file_name"][i],
                "path": batch["path"][i],
                "split": "test",
                "ground_truth_label": int(labels[i]),
                "predicted_label": int(pred[i]),
                "fake_probability": float(prob[i]),
                "detected_faces": int(detected[i]),
                "infer_ms": float(per_ms),
            }
            if seg_np is not None and gt_masks is not None and gt_flag[i] > 0.5:
                p = (seg_np[i].reshape(-1) >= 0.5).astype(np.float32)
                t = (gt_masks[i].reshape(-1) >= 0.5).astype(np.float32)
                inter = float((p * t).sum())
                union = float(((p + t) > 0).sum())
                ps, ts = float(p.sum()), float(t.sum())
                rec["iou"] = 1.0 if union == 0 else inter / union
                rec["dice"] = 1.0 if (ps + ts) == 0 else (2 * inter) / (ps + ts)
            recs.append(rec)
    return recs


FULL = "full_veritas_defacto"
TEST_SETS = discover_test_sets()
SCORE_EXPS = list(NEED_INFER)
if ckpt_files.get(FULL) is not None and FULL not in SCORE_EXPS:
    SCORE_EXPS.append(FULL)

if not SCORE_EXPS:
    print("No checkpoint found. Attach OF_04_full output or models/full_veritas_defacto_best.zip")
elif not TEST_SETS:
    print("No test images found. Attach OpenForensics Wild, DEFACTO Face, FaceForensics++, or a Test/Real+Fake dump.")
else:
    if not torch.cuda.is_available():
        print("WARNING: scoring checkpoints on CPU will be slow. Settings → GPU T4.")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck0 = _torch_load(materialize_checkpoint(ckpt_files[SCORE_EXPS[0]]))
    cfg0 = ck0.get("config") or {}
    image_size = int(cfg0.get("image_size") or 600)
    print("image_size", image_size, "epoch", ck0.get("epoch"), "best", ck0.get("best_metric"))
    batch_size = 4
    for set_name, test_df in TEST_SETS:
        loader = DataLoader(
            DEFACTOFaceDataset(test_df, image_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2 if os.name != "nt" else 0,
        )
        for exp in SCORE_EXPS:
            print("SCORING", exp, "on", set_name, ckpt_files[exp])
            ck = _torch_load(materialize_checkpoint(ckpt_files[exp]))
            model = build_eval_model(exp).to(DEVICE)
            state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
            if isinstance(state, dict) and state and all(str(k).startswith("module.") for k in state):
                state = {k[len("module.") :]: v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            recs = score_loader(model, loader, DEVICE)
            dest = OUT_DIR / "predictions" / f"{exp}_{set_name}_predictions.csv"
            pd.DataFrame(recs).to_csv(dest, index=False)
            if set_name == "openforensics_wild":
                pred_files[exp] = dest
            if recs and set_name == "openforensics_wild":
                y = [r["ground_truth_label"] for r in recs]
                p = [r["predicted_label"] for r in recs]
                print(
                    "CLASSIFICATION",
                    exp,
                    set_name,
                    "acc",
                    float(accuracy_score(y, p)),
                    "f1",
                    float(f1_score(y, p, zero_division=0)),
                    "mean_infer_ms",
                    float(np.mean([r["infer_ms"] for r in recs])),
                )
            elif recs and set_name == "defacto_face":
                dices = [r["dice"] for r in recs if "dice" in r]
                print(
                    "SEGMENTATION",
                    exp,
                    set_name,
                    "n_gt_masks",
                    len(dices),
                    "dice",
                    float(np.mean(dices)) if dices else float("nan"),
                    "(classification accuracy on DEFACTO is not the test)",
                )
            if recs:
                print("wrote", dest, "epoch", ck.get("epoch") if isinstance(ck, dict) else "?")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del loader
"""
        ),
        cd(
            r"""
frames = {}
for exp, path in pred_files.items():
    if path is None:
        print("MISSING predictions for", exp)
        continue
    df = pd.read_csv(path)
    need = {"file_name", "ground_truth_label", "predicted_label", "fake_probability"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df["experiment"] = exp
    frames[exp] = df
    print(exp, "n=", len(df), "acc=", float((df.ground_truth_label == df.predicted_label).mean()))

if len(frames) < 1:
    raise RuntimeError("Need at least one prediction CSV or a checkpoint to score.")
if len(frames) < 2:
    print("Only one variant scored. Table will print; McNemar needs two.")
"""
        ),
        cd(
            r"""
def cls_metrics(df):
    y = df["ground_truth_label"].astype(int)
    p = df["predicted_label"].astype(int)
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    row = {
        "n_faces": int(len(df)),
        "accuracy": float(accuracy_score(y, p)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if "iou" in df.columns:
        fake = df[(df.ground_truth_label == 1) & df["iou"].notna()]
        row["miou_fake_only"] = float(fake["iou"].mean()) if len(fake) else float("nan")
        row["dice_fake_only"] = float(fake["dice"].mean()) if len(fake) and "dice" in fake.columns else float("nan")
    else:
        row["miou_fake_only"] = "N/A"
        row["dice_fake_only"] = "N/A"
    return row


META = {
    "baseline_effb7_defacto": {"multi_stream": False, "segmentation": False},
    "multistream_no_seg_defacto": {"multi_stream": True, "segmentation": False},
    "seg_no_multistream_defacto": {"multi_stream": False, "segmentation": True},
    "full_veritas_defacto": {"multi_stream": True, "segmentation": True},
}

out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working"
pred_dir = out_dir / "predictions"
rows = []
for exp in EXPECTED:
    parts = sorted(pred_dir.glob(f"{exp}_*_predictions.csv")) if pred_dir.is_dir() else []
    if exp in frames and not parts:
        parts = [pred_files.get(exp)] if pred_files.get(exp) else []
    if not parts:
        rows.append({"experiment": exp, "dataset": "", "status": "NOT RUN", **META[exp]})
        continue
    for path in parts:
        df = pd.read_csv(path)
        name = Path(path).name
        dataset = name[len(exp) + 1 : -len("_predictions.csv")] if name.startswith(exp + "_") else "scored"
        if dataset == "openforensics_wild":
            rows.append({"experiment": exp, "dataset": dataset, "task": "classification", "status": "ok", **META[exp], **cls_metrics(df)})
        elif dataset == "defacto_face" and "dice" in df.columns and df["dice"].notna().any():
            rows.append({
                "experiment": exp,
                "dataset": dataset,
                "task": "segmentation",
                "status": "ok",
                **META[exp],
                "n_gt_masks": int(df["dice"].notna().sum()),
                "dice_fake_only": float(df["dice"].mean()),
                "miou_fake_only": float(df["iou"].mean()) if "iou" in df.columns else float("nan"),
            })
        else:
            print("skip", dataset, "for", exp, "(classification test is OpenForensics Wild only; segmentation test is DEFACTO probe_mask)")

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

out_dir.mkdir(parents=True, exist_ok=True)
summary.to_csv(out_dir / "defacto_compare_summary.csv", index=False)
print("wrote", out_dir / "defacto_compare_summary.csv")
"""
        ),
        cd(
            r"""
def mcnemar(a: pd.DataFrame, b: pd.DataFrame):
    ka = {str(r.file_name): r for r in a.itertuples()}
    kb = {str(r.file_name): r for r in b.itertuples()}
    common = sorted(set(ka) & set(kb))
    n01 = n10 = 0
    for k in common:
        a_ok = int(ka[k].ground_truth_label) == int(ka[k].predicted_label)
        b_ok = int(kb[k].ground_truth_label) == int(kb[k].predicted_label)
        if a_ok and not b_ok:
            n01 += 1
        elif (not a_ok) and b_ok:
            n10 += 1
    n = n01 + n10
    if stats is None:
        p = None
    else:
        p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n=n, p=0.5).pvalue)
    return {"n_paired": len(common), "a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "pvalue": p}


pairs = [
    ("baseline_effb7_defacto", "multistream_no_seg_defacto"),
    ("baseline_effb7_defacto", "seg_no_multistream_defacto"),
    ("baseline_effb7_defacto", "full_veritas_defacto"),
    ("multistream_no_seg_defacto", "full_veritas_defacto"),
    ("seg_no_multistream_defacto", "full_veritas_defacto"),
]
stat_rows = []
for a, b in pairs:
    if a not in frames or b not in frames:
        stat_rows.append({"pair": f"{a} vs {b}", "status": "missing"})
        continue
    mc = mcnemar(frames[a], frames[b])
    stat_rows.append({"pair": f"{a} vs {b}", "mcnemar": mc})
    print(f"{a} vs {b}")
    print("  McNemar", mc)

(out_dir / "defacto_statistical_tests.json").write_text(json.dumps(stat_rows, indent=2, default=str), encoding="utf-8")
print("wrote", out_dir / "defacto_statistical_tests.json")
"""
        ),
        md(
            r"""
## Error analysis by number of faces

Haar counts faces on the **full frame**. The network still receives **one** crop: the `probe_mask` box when it exists, otherwise the largest face, otherwise the whole frame.

- **0 faces.** The detector missed. The whole frame is classified, so the error is a detection failure.
- **1 face.** This is the case the model was trained for.
- **2 or more faces.** Every face except the chosen crop is ignored. A real face next to a manipulated face can make the image label and the cropped face disagree. That case is the figure `examples/mixed_two_faces.png`.
"""
        ),
        cd(
            r"""
def _face_bin(n):
    n = int(n)
    if n <= 0:
        return "0 faces"
    if n == 1:
        return "1 face"
    return "2+ faces"


err_rows = []
pred_dir = out_dir / "predictions" if "out_dir" in dir() else (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working") / "predictions"
if not pred_dir.is_dir():
    pred_dir = OUT_DIR / "predictions"
for csv_path in sorted(pred_dir.glob("*_predictions.csv")):
    if "openforensics_wild" not in csv_path.name:
        continue
    df = pd.read_csv(csv_path)
    if "detected_faces" not in df.columns:
        print("no face counts in", csv_path.name, "(rescore with this notebook)")
        continue
    stem = csv_path.name[: -len("_predictions.csv")]
    df["error"] = df.ground_truth_label.astype(int) != df.predicted_label.astype(int)
    df["bin"] = df.detected_faces.map(_face_bin)
    for b, g in df.groupby("bin"):
        y = g.ground_truth_label.astype(int)
        p = g.predicted_label.astype(int)
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        err_rows.append({
            "file": stem,
            "detected_faces": b,
            "n": int(len(g)),
            "errors": int(g.error.sum()),
            "error_rate": float(g.error.mean()) if len(g) else float("nan"),
            "false_fake": fp,
            "false_real": fn,
        })
err_df = pd.DataFrame(err_rows)
if err_df.empty:
    print("No face-count table yet. Score a checkpoint first.")
else:
    print(err_df.to_string(index=False))
    err_df.to_csv(pred_dir.parent / "error_by_face_count.csv", index=False)
    print("wrote", pred_dir.parent / "error_by_face_count.csv")
    print("0 faces: whole-frame fallback after a missed detection.")
    print("1 face: the single-crop protocol.")
    print("2+ faces: only one face is scored, so a mixed real/fake photo is ambiguous.")
"""
        ),
        md(
            r"""
## Hypothesis tests and time added over the baseline

Both models score the **same images**, so the samples are paired.

- **McNemar exact test** on who is correct. Null hypothesis: the two models have the same error rate. This is the right test for two classifiers on one fixed test set. A two-sample t-test on two accuracy numbers is not, because those numbers are not independent samples.
- **Paired bootstrap** (2,000 resamples) for a 95% interval on accuracy(VERITAS) − accuracy(baseline).
- **Wilcoxon signed-rank** on per-image inference milliseconds. Times are paired and skewed, so this is a better test than a t-test. The reported gap is the median milliseconds VERITAS adds on top of the baseline.
"""
        ),
        cd(
            r"""
BASE = "baseline_effb7_defacto"
FULL_NAME = "full_veritas_defacto"


def _load_scored(exp):
    paths = sorted(pred_dir.glob(f"{exp}_*_predictions.csv"))
    if not paths:
        return None
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["dataset"] = p.name[len(exp) + 1 : -len("_predictions.csv")]
        d["experiment"] = exp
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


base_df = _load_scored(BASE)
full_df = _load_scored(FULL_NAME)
time_rows = []
for exp, d in ((BASE, base_df), (FULL_NAME, full_df)):
    if d is None or "infer_ms" not in d.columns:
        print(exp, "has no infer_ms. Rescore to measure time.")
        continue
    d = d[d.dataset == "openforensics_wild"] if "dataset" in d.columns else d
    if d.empty:
        print(exp, "has no OpenForensics Wild rows. Classification time is Wild only.")
        continue
    for ds, g in d.groupby("dataset"):
        time_rows.append({
            "experiment": exp,
            "dataset": ds,
            "n": int(len(g)),
            "median_infer_ms": float(g.infer_ms.median()),
            "mean_infer_ms": float(g.infer_ms.mean()),
        })
if base_df is not None and full_df is not None and "infer_ms" in base_df.columns and "infer_ms" in full_df.columns:
    merged = base_df.merge(full_df, on=["dataset", "file_name"], suffixes=("_base", "_full"))
    merged = merged[merged.dataset == "openforensics_wild"] if "dataset" in merged.columns else merged
    if len(merged):
        added = merged.infer_ms_full - merged.infer_ms_base
        print(
            "VERITAS adds median",
            round(float(added.median()), 2),
            "ms/image over baseline (mean",
            round(float(added.mean()), 2),
            "ms) on",
            len(merged),
            "paired OpenForensics Wild classification images",
        )
        for ds, g in merged.groupby("dataset"):
            gap = g.infer_ms_full - g.infer_ms_base
            time_rows.append({
                "experiment": "veritas_minus_baseline",
                "dataset": ds,
                "n": int(len(g)),
                "median_infer_ms": float(gap.median()),
                "mean_infer_ms": float(gap.mean()),
            })
            a_ok = g.ground_truth_label_base.astype(int) == g.predicted_label_base.astype(int)
            b_ok = g.ground_truth_label_full.astype(int) == g.predicted_label_full.astype(int)
            rng = np.random.default_rng(42)
            idx = np.arange(len(g))
            diffs = []
            for _ in range(2000):
                s = rng.choice(idx, size=len(idx), replace=True)
                diffs.append(float(b_ok.to_numpy()[s].mean() - a_ok.to_numpy()[s].mean()))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(ds, "accuracy VERITAS - baseline", round(float(b_ok.mean() - a_ok.mean()), 4), "95% bootstrap CI", round(float(lo), 4), round(float(hi), 4))
            if stats is not None and len(g) >= 10:
                try:
                    w = stats.wilcoxon(g.infer_ms_base, g.infer_ms_full, zero_method="wilcox", alternative="two-sided")
                    print(ds, "Wilcoxon time p =", float(w.pvalue))
                except ValueError as e:
                    print(ds, "Wilcoxon skipped:", e)
        print("McNemar for the same pairs is in the previous cell. H0: equal error rates on these paired images.")
time_df = pd.DataFrame(time_rows)
if not time_df.empty:
    print(time_df.to_string(index=False))
    time_df.to_csv(pred_dir.parent / "infer_time_vs_baseline.csv", index=False)
    print("wrote", pred_dir.parent / "infer_time_vs_baseline.csv")
"""
        ),
        md(
            r"""
## Limitations (for the presentation)

- **One face in, one probability out.** `examples/mixed_two_faces.png` is one frame with two faces. The right face is the DEFACTO morph `0.jpg_IN_1.jpg`. The left face is a second person from the same dataset. `examples/mixed_two_faces_mask.png` draws the `probe_mask` only on the morph. The scorer keeps a single crop, so the other face is never classified. A photo that mixes an authentic face and a manipulated face fails the same way. Face-count rows with `2+` are this case.
- **The 0.5 cutoff is not learned.** It is a step applied to the sigmoid. Moving it changes false fakes and false reals without retraining. There is no softmax because the head has one logit.
- **Localization is only supervised on `probe_mask`.** Wild fakes train the classifier and skip the segmentation loss, so a mask on those faces is not a ground-truth score.
- **Time.** Full VERITAS keeps the B7 trunk and adds the 9-channel stem plus 10 DeepLab convolutions. `infer_time_vs_baseline.csv` is the measured extra milliseconds per image, not a parameter count.
- **Few real references in DEFACTO** (about 200). A split that pairs those reals with only DEFACTO fakes can look perfect because the domains differ. Wild fakes are in the protocol to block that shortcut.
- **FaceForensics, when attached in `OF_05`, is one video frame**, and c23 and c40 can be mixed. It is not a frame-level or compression-level study.
- **The app detector is MediaPipe; this notebook counts with Haar.** A miss in either detector becomes a whole-frame classification (`0 faces`).
"""
        ),
        md(
            r"""
## How to read this

- Epoch-1 is a valid compare: attach those outputs even if Test was skipped in training.
- If `PROTOCOL_HASH` differed, stop and rerun the mismatched notebook with the agreed CONFIG (`full`, seed 42, size 600).
- Classification rows have `task=classification` and `dataset=openforensics_wild`. McNemar, the face-count error table, and inference time use that set only.
- Segmentation rows have `task=segmentation` and `dataset=defacto_face`. `dice_fake_only` / `miou_fake_only` score **probe_mask** only.
- FaceForensics predictions are written and then skipped in the summary. They are not the classification test.
"""
        ),
    ]
    nb = _nb(cells)
    nb["metadata"]["accelerator"] = "GPU"
    if openforensics_only:
        _restrict_to_openforensics(nb)
    return nb


def _restrict_to_openforensics(nb):
    """Score OpenForensics Wild only. EfficientNet is the Wild/OF baseline file. Full VERITAS is the DEFACTO checkpoint."""
    intro = """
# VERITAS — OpenForensics Test only

This notebook does **not** train. It scores two checkpoints on **OpenForensics Wild Test only** (`Test/Real` and `Test/Fake`).

| Row | Checkpoint | Where it was trained |
| --- | --- | --- |
| EfficientNet baseline | `baseline_effb7_defacto_best.pt` | DEFACTO hybrid (`OF_01_baseline`: DEFACTO Face + OpenForensics Wild), RGB only |
| Full VERITAS | `full_veritas_defacto_best.pt` or `.zip` | Same DEFACTO hybrid (`OF_04_full`) |

`baseline_effb7_wild_best.pt` is ignored. It was trained on Wild only.

## Kaggle

1. New GPU notebook. Import `OF_06_openforensics_compare.ipynb`.
2. Accelerator **GPU T4**.
3. **Add Input → Notebooks**
   - Hybrid baseline: `OF_01_baseline` output `checkpoints/baseline_effb7_defacto_best.pt`
   - Hybrid Full VERITAS: `OF_04_full` output `full_veritas_defacto_best.pt` or `.zip`
4. **Add Input → Datasets** → `ahmedgamal/openforensics-wild-dataset` only. Leave DEFACTO Face and FaceForensics off this run.
5. Save & Run All. The log must show `will score from checkpoint: ['baseline_effb7_defacto', 'full_veritas_defacto']`, then `SCORING` for both. An attached predictions CSV does not replace that score. The training CSV has no inference time.
6. This notebook is the **classification** test. Segmentation dice stays on DEFACTO `probe_mask` in `OF_05_compare` and in each training notebook's Test cell. Wild has no mask, so dice is not computed here.
"""
    expected_old = '''EXPECTED = [
    "baseline_effb7_defacto",
    "multistream_no_seg_defacto",
    "seg_no_multistream_defacto",
    "full_veritas_defacto",
]'''
    expected_new = '''EXPECTED = [
    "baseline_effb7_defacto",
    "full_veritas_defacto",
]
print("dataset: OpenForensics Wild only")
print("EfficientNet baseline file: baseline_effb7_defacto_best.pt (DEFACTO hybrid, OF_01)")
print("VERITAS file: full_veritas_defacto_best.pt or .zip")'''
    pairs_old = '''pairs = [
    ("baseline_effb7_defacto", "multistream_no_seg_defacto"),
    ("baseline_effb7_defacto", "seg_no_multistream_defacto"),
    ("baseline_effb7_defacto", "full_veritas_defacto"),
    ("multistream_no_seg_defacto", "full_veritas_defacto"),
    ("seg_no_multistream_defacto", "full_veritas_defacto"),
]'''
    pairs_new = '''pairs = [
    ("baseline_effb7_defacto", "full_veritas_defacto"),
]'''
    for cell in nb.cells:
        src = cell.source
        if "VERITAS OpenForensics — Compare / cross-dataset Test" in src:
            cell.source = intro.strip() + "\n"
        if expected_old in src:
            cell.source = src.replace(expected_old, expected_new)
        if 'split_ok = p.name.lower() in {"test", "testing", "validation", "val", "test-challenge"}' in src:
            cell.source = src.replace(
                'split_ok = p.name.lower() in {"test", "testing", "validation", "val", "test-challenge"}',
                'split_ok = p.name.lower() in {"test", "testing"}',
                1,
            )
            src = cell.source
        if "SCORE_EXPS = list(NEED_INFER)" in src:
            cell.source = src.replace(
                "SCORE_EXPS = list(NEED_INFER)\n"
                "if ckpt_files.get(FULL) is not None and FULL not in SCORE_EXPS:\n"
                "    SCORE_EXPS.append(FULL)\n",
                "# The attached training output includes a predictions CSV. That file has no infer_ms\n"
                "# and is not a fresh Wild Test score. Load both checkpoints and time them.\n"
                "SCORE_EXPS = [e for e in EXPECTED if ckpt_files.get(e) is not None]\n"
                "print(\"will score from checkpoint:\", SCORE_EXPS)\n",
                1,
            )
            src = cell.source
        if src.strip() == "## Score official Test from epoch-1 checkpoints (only if a CSV is missing)":
            cell.source = (
                "## Score Wild Test from both checkpoints\n\n"
                "Both checkpoints are loaded even when the training output already contains a predictions CSV. "
                "That CSV has no inference time.\n"
            )
            src = cell.source
        if "TEST_SETS = discover_test_sets()" in src and "openforensics_wild" not in src.split("TEST_SETS = discover_test_sets()")[1][:180]:
            cell.source = src.replace(
                "TEST_SETS = discover_test_sets()\n",
                "TEST_SETS = discover_test_sets()\n"
                "TEST_SETS = [(n, df) for n, df in TEST_SETS if n == \"openforensics_wild\"]\n"
                "if len(TEST_SETS) > 1:\n"
                "    print(\"WARNING: more than one Wild Test folder. Scoring the first only.\")\n"
                "    for _n, _df in TEST_SETS:\n"
                "        print(\" \", len(_df), _df[\"path\"].iloc[0] if len(_df) else \"\")\n"
                "    TEST_SETS = TEST_SETS[:1]\n"
                "print(\"scoring dataset: openforensics_wild only\", [n for n, _ in TEST_SETS])\n",
                1,
            )
        if pairs_old in src:
            cell.source = src.replace(pairs_old, pairs_new)
        if src.startswith("## How to read this"):
            cell.source = (
                "## How to read this\n\n"
                "- Baseline row is the DEFACTO-hybrid EfficientNet (`baseline_effb7_defacto_best.pt` from `OF_01`).\n"
                "- Full row is DEFACTO-hybrid VERITAS (`full_veritas_defacto_best` from `OF_04`).\n"
                "- Both numbers are OpenForensics Wild Test only. DEFACTO Face and FaceForensics are not scored here.\n"
                "- Use `OF_05_compare.ipynb` for the cross-dataset run.\n"
            )


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        dest = DEST_DIR / v["file"]
        nbf.write(build_training_nb(v), dest)
        print("wrote", dest, "exp", v["exp"])
    dest = DEST_DIR / "OF_05_compare.ipynb"
    nbf.write(build_compare_nb(False), dest)
    print("wrote", dest)
    dest = DEST_DIR / "OF_06_openforensics_compare.ipynb"
    nbf.write(build_compare_nb(True), dest)
    print("wrote", dest)


if __name__ == "__main__":
    main()
