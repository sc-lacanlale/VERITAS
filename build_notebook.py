"""Generate VERITAS_Kaggle.ipynb — self-contained Kaggle experimental notebook."""
from pathlib import Path

import nbformat as nbf

cells = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip() + "\n"))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip() + "\n"))


md(
    r"""
# VERITAS — Vision-based EfficientNet-Reinforced Image Tampering Authentication
## Multi-face MTL experimental framework (OpenForensics compiled)

This notebook is the complete, rerunnable Kaggle implementation of the VERITAS
ablation study. It does **not** train one monolithic model. It trains and
compares four architectural variants on the **same face-level split**:

| Experiment | EfficientNet-B7 | Multi-stream | Segmentation | Classification |
| --- | ---: | ---: | ---: | ---: |
| `baseline_effb7` | yes | no | no | yes |
| `multistream_no_seg` | yes | yes | no | yes |
| `seg_no_multistream` | yes | no | yes | yes |
| `full_veritas` | yes | yes | yes | yes |

**Inspected dataset facts** (do not assume other schemas):

- Source: compiled OpenForensics (`compiled_poly.json` / `compiled_index.parquet`)
- Categories: `0 = Real`, `1 = Fake`
- Images live in a **flat** `Images/` folder (`Images/<hash>.jpg`), not split subfolders
- Face boxes are COCO `[x, y, w, h]` in **absolute pixels**
- Polygons are COCO segmentation lists in **absolute image coordinates**
- Official splits exist (`Train`, `Val`, `Test-Dev`, `Test-Challenge`)
- Many images contain mixed real/fake faces; evaluation is **face-level**, never image-level
- Segmentation target in this study is **manipulation localization**: fake-face polygon = 1, real face = 0

Set `CONFIG["run_mode"]` to `"smoke"` (default, tiny subset) or `"full"` (research run).
"""
)

# ---------------------------------------------------------------------------
# Cell 1 — Environment
# ---------------------------------------------------------------------------
md("## 1. Environment")

code(
    r"""
import os
import sys
import platform
import subprocess
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# Install only if missing. Do not upgrade preinstalled Kaggle stacks.
_required = {
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "pyarrow": "pyarrow",
}
for import_name, pip_name in _required.items():
    try:
        __import__(import_name if import_name != "cv2" else "cv2")
    except Exception:
        print(f"Installing missing package: {pip_name}")
        _pip_install([pip_name])

try:
    import mediapipe  # noqa: F401
except Exception:
    print("Installing mediapipe (current Tasks API, not deprecated solutions)")
    try:
        _pip_install(["mediapipe"])
    except Exception as e:
        print("mediapipe install failed; detector will use a non-MediaPipe fallback:", e)

import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
from PIL import Image

print("Python:", sys.version.split()[0], platform.platform())
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("PIL:", Image.__version__ if hasattr(Image, "__version__") else getattr(Image, "PILLOW_VERSION", "?"))
try:
    import mediapipe as mp
    print("MediaPipe:", mp.__version__)
except Exception as e:
    mp = None
    print("MediaPipe: unavailable", e)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / (1024**3):.2f} GB")
else:
    print("GPU: CPU fallback")
"""
)

# ---------------------------------------------------------------------------
# Cell 2 — Imports
# ---------------------------------------------------------------------------
md("## 2. Imports")

code(
    r"""
import gc
import json
import math
import random
import shutil
import time
import hashlib
import traceback
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from scipy import stats

try:
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions as MpBaseOptions
    MP_TASKS_AVAILABLE = True
except Exception as e:
    mp_vision = None
    MpBaseOptions = None
    MP_TASKS_AVAILABLE = False
    print("MediaPipe Tasks API not importable:", e)

%matplotlib inline
plt.rcParams["figure.dpi"] = 120
"""
)

# ---------------------------------------------------------------------------
# Cell 3 — Dataset discovery
# ---------------------------------------------------------------------------
md("## 3. Dataset discovery")

code(
    r"""
def _is_kaggle() -> bool:
    return Path("/kaggle/input").exists() or Path("/kaggle/working").exists()


def discover_paths() -> Dict[str, Path]:
    # Inspect actual filesystem. Never hard-code a dataset folder name.
    search_roots: List[Path] = []
    cwd = Path.cwd()
    if Path("/kaggle/input").exists():
        search_roots.append(Path("/kaggle/input"))
    search_roots.extend(
        [
            cwd / "dataset",
            cwd / "dataset" / "compiled",
            cwd,
        ]
    )

    found_parquet = None
    found_poly = None
    found_images = None
    found_missing = None
    found_extra = None
    walked = []

    for root in search_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            walked.append(dirpath)
            names = set(filenames)
            if found_parquet is None and "compiled_index.parquet" in names:
                found_parquet = Path(dirpath) / "compiled_index.parquet"
            if found_poly is None and "compiled_poly.json" in names:
                found_poly = Path(dirpath) / "compiled_poly.json"
            if found_poly is None and "poly.json" in names:
                found_poly = Path(dirpath) / "poly.json"
            if found_missing is None and "missing_images.csv" in names:
                found_missing = Path(dirpath) / "missing_images.csv"
            if found_extra is None and "extra_files.csv" in names:
                found_extra = Path(dirpath) / "extra_files.csv"
            if found_images is None:
                if "Images" in dirnames:
                    found_images = Path(dirpath) / "Images"
                elif any(fn.lower().endswith((".jpg", ".jpeg", ".png")) for fn in filenames[:20]) and Path(dirpath).name.lower() in {
                    "images",
                    "train",
                    "val",
                    "test-dev",
                    "test-challenge",
                }:
                    found_images = Path(dirpath)
            # Do not enumerate 100k JPEGs during discovery.
            if Path(dirpath).name == "Images":
                dirnames[:] = []
            if found_parquet and found_images and found_poly:
                break
        if found_parquet and found_images:
            break

    in_kaggle = Path("/kaggle/input").exists()
    working = Path("/kaggle/working") if in_kaggle else (cwd / "kaggle_working")
    working.mkdir(parents=True, exist_ok=True)

    print("Walked roots:", [str(r) for r in search_roots if r.exists()])
    print("parquet:", found_parquet)
    print("poly.json:", found_poly)
    print("Images:", found_images)
    print("working:", working)
    if found_images is None or (found_parquet is None and found_poly is None):
        raise FileNotFoundError(
            "Could not locate compiled OpenForensics files. "
            "Attach dataset nathanielescuro/veritas-openforensics-compiled on Kaggle."
        )
    return {
        "parquet": found_parquet,
        "poly": found_poly,
        "images": found_images,
        "missing_csv": found_missing,
        "extra_csv": found_extra,
        "working": working,
        "dataset_root": found_images.parent if found_images is not None else None,
    }


PATHS = discover_paths()
for sub in ["checkpoints", "predictions", "masks", "debug", "logs"]:
    (PATHS["working"] / sub).mkdir(parents=True, exist_ok=True)
print("Artifact dirs ready under", PATHS["working"])
"""
)

# ---------------------------------------------------------------------------
# Cell 4 — Dataset structure
# ---------------------------------------------------------------------------
md("## 4. Dataset structure")

code(
    r"""
def sample_image_files(image_dir: Path, n: int = 8) -> List[Path]:
    out = []
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            out.append(p)
            if len(out) >= n:
                break
    return out


print("Image directory listing (sample):")
for p in sample_image_files(PATHS["images"]):
    print(" ", p.name, p.stat().st_size, "bytes")

subdirs = [p.name for p in PATHS["images"].iterdir() if p.is_dir()]
print("Image subdirectories:", subdirs[:20], "(empty means flat Images/ layout)")
"""
)

# ---------------------------------------------------------------------------
# Cell 5 — Central configuration
# ---------------------------------------------------------------------------
md("## 5. Central configuration")

code(
    r"""
CONFIG: Dict[str, Any] = {
    # "smoke" = tiny subset so the notebook finishes; "full" = research protocol
    "run_mode": "smoke",
    "seed": 42,
    "backbone": "efficientnet_b7",
    "use_pretrained": True,
    "image_size": 600,  # EfficientNet-B7 native; smoke mode may downscale
    "face_margin": 0.10,
    "batch_size": 4,
    "num_workers": 2 if os.name != "nt" else 0,
    "epochs": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "classification_threshold": 0.5,
    "lambda_cls": 1.0,
    "lambda_seg": 1.0,
    "dice_weight": 1.0,  # L_seg = BCE + dice_weight * Dice
    "use_amp": True,
    "gradient_accumulation": 1,
    "match_iou_threshold": 0.3,
    "face_score_threshold": 0.5,
    "split_protocol": "70_15_15",  # SOP default; "official_openforensics" also supported
    "split_pool": "train_val_testdev",  # do not mix Test-Challenge into 70/15/15 unless "all"
    "max_train_images": None,
    "max_val_images": None,
    "max_test_images": None,
    "skip_training": False,
    "resume": True,
    "experiments_to_run": [
        "baseline_effb7",
        "multistream_no_seg",
        "seg_no_multistream",
        "full_veritas",
    ],
    "loss_configs": [
        {"name": "1_1", "lambda_cls": 1.0, "lambda_seg": 1.0},
        {"name": "2_1", "lambda_cls": 2.0, "lambda_seg": 1.0},
        {"name": "1_2", "lambda_cls": 1.0, "lambda_seg": 2.0},
    ],
    "run_loss_sweeps": False,  # set True in full mode to sweep λ on full_veritas
    "latency_warmup": 5,
    "latency_repeats": 20,
}

EXPERIMENTS = {
    "baseline_effb7": {
        "multi_stream": False,
        "segmentation": False,
        "backbone": "efficientnet_b7",
    },
    "multistream_no_seg": {
        "multi_stream": True,
        "segmentation": False,
        "backbone": "efficientnet_b7",
    },
    "seg_no_multistream": {
        "multi_stream": False,
        "segmentation": True,
        "backbone": "efficientnet_b7",
    },
    "full_veritas": {
        "multi_stream": True,
        "segmentation": True,
        "backbone": "efficientnet_b7",
    },
}

if CONFIG["run_mode"] == "smoke":
    CONFIG.update(
        {
            "image_size": 320,
            "batch_size": 2,
            "epochs": 1,
            "max_train_images": 24,
            "max_val_images": 8,
            "max_test_images": 8,
            "latency_warmup": 1,
            "latency_repeats": 3,
            "run_loss_sweeps": False,
            "use_pretrained": True,
        }
    )
    print("SMOKE MODE: reduced image_size/epochs/subset. Set run_mode='full' for the thesis protocol.")
    print("SOP native EfficientNet-B7 size is 600; smoke uses", CONFIG["image_size"], "to stay runnable.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
print("CONFIG:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
"""
)

# ---------------------------------------------------------------------------
# Cell 6 — Reproducibility
# ---------------------------------------------------------------------------
md("## 6. Reproducibility")

code(
    r"""
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CONFIG["seed"])
print("Seed locked to", CONFIG["seed"])
"""
)

# ---------------------------------------------------------------------------
# Cell 7 — Annotation inspection
# ---------------------------------------------------------------------------
md("## 7. Annotation inspection")

code(
    r"""
CATEGORY_NAME = {0: "real", 1: "fake"}


def parse_segmentation(seg: Any) -> List[List[Tuple[float, float]]]:
    # Parse COCO-style segmentation from parquet string or native JSON.
    if seg is None or (isinstance(seg, float) and math.isnan(seg)):
        return []
    if isinstance(seg, str):
        seg = json.loads(seg)
    polygons: List[List[Tuple[float, float]]] = []
    if not isinstance(seg, list):
        return polygons
    for poly in seg:
        if poly is None or len(poly) == 0:
            continue
        if isinstance(poly[0], (list, tuple)):
            pts = [(float(p[0]), float(p[1])) for p in poly if len(p) >= 2]
        else:
            vals = [float(x) for x in poly]
            pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def load_annotation_index(paths: Dict[str, Path]) -> pd.DataFrame:
    if paths.get("parquet") is not None and Path(paths["parquet"]).exists():
        df = pd.read_parquet(paths["parquet"])
        print("Loaded parquet index:", df.shape)
    elif paths.get("poly") is not None:
        print("Loading poly.json (large); prefer parquet when available...")
        with open(paths["poly"], "r", encoding="utf-8") as f:
            raw = json.load(f)
        images = {im["id"]: im for im in raw.get("images", [])}
        rows = []
        for ann in raw.get("annotations", []):
            im = images.get(ann["image_id"], {})
            bbox = ann.get("bbox", [0, 0, 0, 0])
            rows.append(
                {
                    "ann_id": ann.get("id"),
                    "image_id": ann.get("image_id"),
                    "split": im.get("split", "unknown"),
                    "file_name": im.get("file_name"),
                    "category_id": ann.get("category_id"),
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_w": bbox[2],
                    "bbox_h": bbox[3],
                    "area": ann.get("area", 0),
                    "segmentation": json.dumps(ann.get("segmentation", [])),
                    "orig_ann_id": ann.get("id"),
                    "width": im.get("width"),
                    "height": im.get("height"),
                }
            )
        df = pd.DataFrame(rows)
        print("Loaded poly.json:", df.shape)
        print("categories:", raw.get("categories"))
    else:
        raise FileNotFoundError("No parquet or poly.json")
    print("columns:", list(df.columns))
    print(df.head(2).to_string())
    print("\ncategory_id counts:\n", df["category_id"].value_counts())
    print("\nsplit counts (faces):\n", df["split"].value_counts())
    print("\nunique images:", df["image_id"].nunique())
    print("faces/image mean:", df.groupby("image_id").size().mean())
    mixed = (df.groupby("image_id")["category_id"].nunique() > 1).sum()
    print("mixed real/fake images:", int(mixed))
    return df


ANN_DF = load_annotation_index(PATHS)

# Verify a representative schema row
_schema_row = ANN_DF.iloc[0]
print("\nSchema example:")
print("  image_id:", _schema_row["image_id"])
print("  file_name:", _schema_row["file_name"])
print("  category_id:", int(_schema_row["category_id"]), "=>", CATEGORY_NAME.get(int(_schema_row["category_id"])))
print("  bbox xywh:", [_schema_row["bbox_x"], _schema_row["bbox_y"], _schema_row["bbox_w"], _schema_row["bbox_h"]])
_polys = parse_segmentation(_schema_row["segmentation"])
print("  n polygons:", len(_polys), "first n pts:", 0 if not _polys else len(_polys[0]))
print("  coordinate convention: x=horizontal, y=vertical, absolute pixels, origin top-left")
"""
)

print("Notebook part 1 cells defined:", len(cells))
