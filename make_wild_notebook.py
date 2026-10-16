"""Build notebooks-wild/ — folder-labeled OpenForensics Wild, one experiment per notebook."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DEST_DIR = Path("notebooks-wild")

VARIANTS = [
    {
        "file": "WILD_01_baseline.ipynb",
        "exp": "baseline_effb7_wild",
        "title": "Baseline EfficientNet-B7",
        "who": "Member A",
        "n": "1 of 4",
        "multi_stream": False,
        "segmentation": False,
        "legacy_baseline": True,
        "backbone": "efficientnet_b7",
        "blurb": "RGB EfficientNet-B7, no multi-stream, no segmentation.",
    },
    {
        "file": "WILD_02_multistream.ipynb",
        "exp": "multistream_no_seg_wild",
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
        "file": "WILD_03_segmentation.ipynb",
        "exp": "seg_no_multistream_wild",
        "title": "Segmentation / MTL (single RGB stream)",
        "who": "Member C",
        "n": "3 of 4",
        "multi_stream": False,
        "segmentation": True,
        "legacy_baseline": False,
        "backbone": "efficientnet_b7",
        "blurb": "RGB EfficientNet-B7 + DeepLabV3+ head. Wild has no polygons: fake crops get a face-oval surrogate mask, reals get zeros.",
    },
    {
        "file": "WILD_04_full.ipynb",
        "exp": "full_veritas_wild",
        "title": "Full VERITAS",
        "who": "Member D",
        "n": "4 of 4",
        "multi_stream": True,
        "segmentation": True,
        "legacy_baseline": False,
        "backbone": "efficientnet_b7",
        "blurb": "Multi-stream + DeepLabV3+ MTL. Same surrogate masks as notebook 3.",
    },
]

B0_VARIANTS = [
    {
        "file": "WILD_B0_01_baseline.ipynb",
        "exp": "baseline_effb0_wild",
        "title": "Baseline EfficientNet-B0",
        "who": "Member A",
        "n": "1 of 1",
        "multi_stream": False,
        "segmentation": False,
        "legacy_baseline": True,
        "backbone": "efficientnet_b0",
        "out_dir": Path("notebooks-wild-b0"),
        "blurb": "RGB EfficientNet-B0 only (no multi-stream, no segmentation). Same Wild crops and 600 letterbox as the B7 notebooks. B7 baseline and Full both hit ~97% — this is the weaker-backbone check.",
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


def build_training_nb(v: dict):
    exp = v["exp"]
    ms = "True" if v["multi_stream"] else "False"
    sg = "True" if v["segmentation"] else "False"
    lg = "True" if v["legacy_baseline"] else "False"
    bb = v.get("backbone", "efficientnet_b7")
    imagenet_note = "ImageNet B0" if bb == "efficientnet_b0" else "ImageNet B7"
    merge_note = (
        "Do **not** attach B7 Wild checkpoints here. This notebook is a weaker-backbone check "
        "(B7 Wild baseline and Full both ~97%). Same 600 letterbox, RGB classification only."
        if bb == "efficientnet_b0"
        else "The other Wild variants are **not** trained here. Merge later with `WILD_05_compare.ipynb`."
    )
    protocol_extra = ', "backbone"' if bb == "efficientnet_b0" else ""
    seg_note = ""
    if v["segmentation"]:
        seg_note = """
**Masks:** this dataset has no polygons. Fake images get an ellipse on the letterboxed
face crop; real images get an all-zero mask. Dice/IoU here are **not** comparable to
compiled-poly OpenForensics.
"""
    cells = [
        md(
            f"""
# VERITAS Wild — {v['title']}

**Assigned to:** {v['who']} ({v['n']})  
**This notebook trains only:** `{exp}`

**Dataset:** [ahmedgamal/openforensics-wild-dataset](https://www.kaggle.com/datasets/ahmedgamal/openforensics-wild-dataset)

This is **not** the compiled-poly OpenForensics study in `notebooks/`. Images are already
face crops. Labels are folder names (`Real` = 0, `Fake` = 1). Splits are official
`Train` / `Validation` / `Test`.

{v['blurb']}
{seg_note}
{merge_note}

## Attach (before Run All)

1. New GPU notebook. Do **not** import this into DS18, VERITAS-finetuning, or another Wild notebook.
2. **Add Input → Datasets** → `ahmedgamal/openforensics-wild-dataset`.
3. Settings → Accelerator → **GPU T4** (not CPU / None). Internet On (first run downloads {imagenet_note}), Persistence Files only.
4. Smoke first (`run_mode: smoke`), then `full`. Full mode **aborts** if CUDA is missing — 140k crops will not finish on CPU.

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
        ),
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
def discover_wild_root() -> Path:
    roots = []
    if Path("/kaggle/input").exists():
        roots.append(Path("/kaggle/input"))
    roots.extend([Path.cwd() / "dataset", Path.cwd()])
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            kids = {d.lower() for d in dirnames}
            if {"train", "validation", "test"} <= kids or {"train", "val", "test"} <= kids:
                return p
            if p.name.lower() == "dataset" and ("train" in kids or "test" in kids):
                return p
    raise FileNotFoundError(
        "Could not find OpenForensics Wild folders. "
        "Attach dataset ahmedgamal/openforensics-wild-dataset "
        "(expect Dataset/Train|Validation|Test / Real|Fake)."
    )


def _split_dir(root: Path, name: str) -> Path:
    aliases = {
        "train": ["Train", "train"],
        "val": ["Validation", "Val", "validation", "val"],
        "test": ["Test", "test"],
    }
    for a in aliases[name]:
        p = root / a
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No {name} split under {root}")


WILD_ROOT = discover_wild_root()
IN_KAGGLE = Path("/kaggle/input").exists()
WORKING = Path("/kaggle/working") if IN_KAGGLE else Path.cwd() / "kaggle_working"
WORKING.mkdir(parents=True, exist_ok=True)
for sub in ("checkpoints", "predictions", "logs", "debug"):
    (WORKING / sub).mkdir(parents=True, exist_ok=True)
print("wild root:", WILD_ROOT)
print("working:", WORKING)
for split in ("train", "val", "test"):
    print(" ", split, _split_dir(WILD_ROOT, split))
"""
        ),
        md("## 4. Configuration"),
        cd(
            f"""
CONFIG: Dict[str, Any] = {{
    "run_mode": "smoke",
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
    "epochs_per_run": 1,
    "mid_epoch_checkpoint_minutes": 30,
    "test_eval_when": "always",
    "experiment": "{exp}",
    "max_per_split": None,
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
            "max_per_split": 64,
            "latency_repeats": 3,
        }}
    )
    print("SMOKE MODE: tiny subset, image_size=320, 1 epoch. Set run_mode='full' for the real run.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
print("PyTorch build:", torch.__version__, "CUDA:", torch.cuda.is_available())
if CONFIG["run_mode"] == "full" and DEVICE.type != "cuda":
    raise RuntimeError(
        "Full Wild training needs a GPU session. "
        "Kaggle: Settings → Accelerator → GPU T4 (or P100). "
        f"This process is {{torch.__version__}} CUDA={{torch.cuda.is_available()}}. "
        "140k crops at 600px will not finish one epoch on CPU in 12h. "
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
    "epochs", "batch_size", "learning_rate", "lambda_cls", "lambda_seg", "dice_weight"{protocol_extra}
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
        md("## 5. Index images from Real / Fake folders"),
        cd(
            r"""
LABELS = {"real": 0, "fake": 1}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _class_dir(split_dir: Path, name: str) -> Optional[Path]:
    for a in (name, name.capitalize(), name.upper(), name.lower()):
        p = split_dir / a
        if p.is_dir():
            return p
    return None


def list_split(split_name: str) -> pd.DataFrame:
    split_dir = _split_dir(WILD_ROOT, split_name)
    rows = []
    for cname, lab in LABELS.items():
        cdir = _class_dir(split_dir, cname)
        if cdir is None:
            print("WARNING: missing", split_name, cname, "under", split_dir)
            continue
        for p in sorted(cdir.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTS:
                rows.append(
                    {
                        "path": str(p),
                        "split": split_name,
                        "label": lab,
                        "class_name": cname,
                        "file_name": p.name,
                    }
                )
    df = pd.DataFrame(rows)
    if CONFIG.get("max_per_split"):
        parts = []
        rng = np.random.RandomState(CONFIG["seed"] + {"train": 1, "val": 2, "test": 3}[split_name])
        for lab, g in df.groupby("label"):
            idx = rng.choice(g.index, size=min(int(CONFIG["max_per_split"]), len(g)), replace=False)
            parts.append(g.loc[idx])
        df = pd.concat(parts).sort_index()
    return df.reset_index(drop=True)


INDEX = {s: list_split(s) for s in ("train", "val", "test")}
for s, df in INDEX.items():
    n_fake = int((df.label == 1).sum()) if len(df) else 0
    print(f"{s}: {len(df)} images  fake={n_fake} real={len(df) - n_fake}")
if any(len(INDEX[s]) == 0 for s in INDEX):
    raise RuntimeError("A split has 0 images. Check Real/Fake folder names under the dataset.")
"""
        ),
        md("## 6. Letterbox + loaders (600 / ImageNet). Seg uses a crop-oval surrogate mask."),
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


def crop_ellipse_mask(size: int, box, is_fake: bool) -> np.ndarray:
    # Wild has no polygons. Fake = ellipse on the letterboxed crop; real = zeros.
    m = np.zeros((size, size), dtype=np.float32)
    if not is_fake:
        return m
    left, top, nw, nh = box
    cx = left + nw / 2.0
    cy = top + nh / 2.0
    rx = max(nw * 0.46, 1.0)
    ry = max(nh * 0.46, 1.0)
    yy, xx = np.ogrid[:size, :size]
    m[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0] = 1.0
    return m


class WildFaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        bgr = cv2.imread(row.path, cv2.IMREAD_COLOR)
        if bgr is None:
            rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.augment and random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        if self.augment and random.random() < 0.5:
            rgb = np.clip(rgb.astype(np.float32) * (1.0 + random.uniform(-0.15, 0.15)) + random.uniform(-12, 12), 0, 255).astype(np.uint8)
        canvas, box = letterbox_rgb(rgb, self.size)
        x = canvas.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
        x = np.transpose(x, (2, 0, 1))
        mask = crop_ellipse_mask(self.size, box, bool(int(row.label) == 1))
        return {
            "image": torch.from_numpy(x.copy()),
            "mask": torch.from_numpy(mask[None].copy()),
            "label": torch.tensor([float(row.label)], dtype=torch.float32),
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


def make_loader(df, shuffle, augment):
    ds = WildFaceDataset(df, CONFIG["image_size"], augment=augment)
    sampler = EpochSeededSampler(len(ds), CONFIG["seed"]) if shuffle else None
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
"""
        ),
        md("## 7. Model (same B7 / SRM / DeepLabV3+ as compiled VERITAS)"),
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
    # WILD_01 B7 layout is still features.* + cls_head.* so those checkpoints keep loading.

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

    def forward(self, outputs, labels, masks=None):
        l_cls = self.bce_cls(outputs["classification_logits"], labels)
        l_seg = labels.new_zeros(())
        if self.segmentation:
            if masks is None:
                raise ValueError("masks required for segmentation experiments")
            l_seg = self.bce_seg(outputs["segmentation_logits"], masks) + self.dice_weight * self.dice(
                outputs["segmentation_logits"], masks
            )
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
def mask_metrics_from_tensors(probs, targets, labels, thresh=0.5):
    probs = probs.numpy() if torch.is_tensor(probs) else np.asarray(probs)
    targets = targets.numpy() if torch.is_tensor(targets) else np.asarray(targets)
    labels = np.asarray(labels).reshape(-1)
    ious, dices, ious_fake, dices_fake = [], [], [], []
    for i in range(len(labels)):
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
def evaluate_loader(model, loader, split_name: str):
    model.eval()
    y_true, y_pred, y_prob, recs = [], [], [], []
    dice_all, iou_all, dice_fake, iou_fake = [], [], [], []
    use_seg = bool(getattr(model, "segmentation", False))
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        out = model(images)
        prob = out["fake_probability"].detach().cpu().numpy().reshape(-1)
        pred = (prob >= CONFIG["classification_threshold"]).astype(int)
        true = labels.detach().cpu().numpy().reshape(-1).astype(int)
        y_true.extend(true.tolist())
        y_pred.extend(pred.tolist())
        y_prob.extend(prob.tolist())
        mm = None
        if use_seg and "segmentation_prob" in out:
            mm = mask_metrics_from_tensors(
                out["segmentation_prob"].detach().cpu(), masks.detach().cpu(), true, CONFIG["mask_threshold"]
            )
            iou_all.extend(mm["iou_all"])
            dice_all.extend(mm["dice_all"])
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
            if mm is not None:
                rec["iou"] = mm["iou_all"][i]
                rec["dice"] = mm["dice_all"][i]
            recs.append(rec)
        del images, labels, masks, out
    metrics = {
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
        metrics["miou_note"] = "surrogate crop-oval masks (Wild has no polygons); dice/miou are fake-only"
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
        print("WARNING: no CUDA. Full Wild will not finish an epoch in 12h.")
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
        if partial is not None and int(partial.get("epoch", -1)) == epoch and isinstance(sampler, EpochSeededSampler):
            start_step = int(partial["step"])
            print(f"Continuing epoch {epoch + 1} from step {start_step}/{len(TRAIN_LOADER)}")
        partial = None
        if isinstance(sampler, EpochSeededSampler):
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
            with autocast():
                out = model(images)
                losses = criterion(out, labels, masks if use_seg else None)
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
        md("## 10. Test (only after the last planned epoch)"),
        cd(
            r"""
done = EPOCHS_DONE >= int(CONFIG["epochs"])
if CONFIG.get("test_eval_when", "final") == "final" and not done:
    print(f"test_eval_when='final': {EPOCHS_DONE}/{CONFIG['epochs']} epochs, skipping official Test this run.")
else:
    ckpt_best = WORKING / "checkpoints" / f"{EXP}_best.pt"
    if ckpt_best.exists():
        load_if_any(ckpt_best, MODEL)
    metrics, recs = evaluate_loader(MODEL, TEST_LOADER, "test")
    print("TEST", metrics)
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

- Official Wild split: Train / Validation / Test folders. Do not mix compiled-poly images into this run.
- First full run starts from ImageNet B7. Later runs attach **this notebook’s** latest output.
- Smoke uses 64 images per class per split. `full` uses every image in those folders.
- `{exp}` is **not** comparable to `notebooks/` compiled-poly runs (different images, no GT boxes).
- All four Wild notebooks must print the same `PROTOCOL_HASH` (`run_mode`, seed, size, threshold, epochs, batch, lr, lambdas).
"""
        ),
    ]
    return _nb(cells)


def build_compare_nb():
    cells = [
        md(
            r"""
# VERITAS Wild — Compare / epoch-1 Test

This notebook does **not** train.

It uses existing `predictions/<exp>_predictions.csv` when present. If a variant only
has a checkpoint (typical after epoch 1, when Test was skipped), it **scores official
Test here** from `{exp}_last.pt` (else `_best.pt`).

**Notebooks 2 and 3 are optional.** Baseline + Full is enough. One variant is OK if
you only want that epoch-1 Test number.

Attach:

1. `ahmedgamal/openforensics-wild-dataset` (needed only when a CSV is missing)
2. Each training notebook’s latest output (epoch-1 is fine)

Use **GPU T4** if any checkpoint must be scored. CSV-only compare can run on CPU.

- `baseline_effb7_wild`
- `multistream_no_seg_wild` (optional)
- `seg_no_multistream_wild` (optional)
- `full_veritas_wild`
"""
        ),
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

EXPECTED = [
    "baseline_effb7_wild",
    "multistream_no_seg_wild",
    "seg_no_multistream_wild",
    "full_veritas_wild",
]
print("search roots:", [str(p) for p in SEARCH_ROOTS if p.exists()])
"""
        ),
        cd(
            r"""
def find_files(name_substr: str, suffix: str):
    hits = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() == suffix and name_substr in p.name:
                hits.append(p)
    return sorted(set(hits))


pred_files = {}
ckpt_files = {}
for exp in EXPECTED:
    cands = find_files(f"{exp}_predictions", ".csv")
    pred_files[exp] = cands[0] if cands else None
    lasts = [p for p in find_files(f"{exp}_last", ".pt") if p.name == f"{exp}_last.pt"]
    bests = [p for p in find_files(f"{exp}_best", ".pt") if p.name == f"{exp}_best.pt"]
    ckpt_files[exp] = (lasts[0] if lasts else None) or (bests[0] if bests else None)
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
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7
from tqdm.auto import tqdm

META = {
    "baseline_effb7_wild": {"multi_stream": False, "segmentation": False, "legacy": True},
    "multistream_no_seg_wild": {"multi_stream": True, "segmentation": False, "legacy": False},
    "seg_no_multistream_wild": {"multi_stream": False, "segmentation": True, "legacy": False},
    "full_veritas_wild": {"multi_stream": True, "segmentation": True, "legacy": False},
}
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working"
(OUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def discover_wild_root() -> Path:
    roots = []
    if Path("/kaggle/input").exists():
        roots.append(Path("/kaggle/input"))
    roots.extend([Path.cwd() / "dataset", Path.cwd()])
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            kids = {d.lower() for d in dirnames}
            if {"train", "validation", "test"} <= kids or {"train", "val", "test"} <= kids:
                return p
    raise FileNotFoundError("Attach ahmedgamal/openforensics-wild-dataset to score Test from checkpoints.")


def _split_dir(root: Path, name: str) -> Path:
    for a in (["Test", "test"] if name == "test" else [name]):
        p = root / a
        if p.is_dir():
            return p
    raise FileNotFoundError(name)


def list_test(root: Path) -> pd.DataFrame:
    split_dir = _split_dir(root, "test")
    rows = []
    for cname, lab in (("real", 0), ("fake", 1)):
        cdir = None
        for a in (cname, cname.capitalize(), cname.upper()):
            if (split_dir / a).is_dir():
                cdir = split_dir / a
                break
        if cdir is None:
            continue
        for p in sorted(cdir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                rows.append({"path": str(p), "label": lab, "file_name": p.name})
    return pd.DataFrame(rows)


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


def crop_ellipse_mask(size, box, is_fake):
    m = np.zeros((size, size), dtype=np.float32)
    if not is_fake:
        return m
    left, top, nw, nh = box
    cx, cy = left + nw / 2.0, top + nh / 2.0
    rx, ry = max(nw * 0.46, 1.0), max(nh * 0.46, 1.0)
    yy, xx = np.ogrid[:size, :size]
    m[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0] = 1.0
    return m


class WildFaceDataset(Dataset):
    def __init__(self, df, size):
        self.df = df.reset_index(drop=True)
        self.size = int(size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bgr = cv2.imread(row.path, cv2.IMREAD_COLOR)
        rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8) if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas, box = letterbox_rgb(rgb, self.size)
        x = np.transpose((canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN_NP) / IMAGENET_STD_NP, (2, 0, 1))
        mask = crop_ellipse_mask(self.size, box, int(row.label) == 1)
        return {
            "image": torch.from_numpy(x.copy()),
            "mask": torch.from_numpy(mask[None].copy()),
            "label": torch.tensor([float(row.label)], dtype=torch.float32),
            "path": row.path,
            "file_name": row.file_name,
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
        out = model(images)
        prob = out["fake_probability"].detach().cpu().numpy().reshape(-1)
        pred = (prob >= thresh).astype(int)
        for i in range(len(labels)):
            recs.append({
                "file_name": batch["file_name"][i],
                "path": batch["path"][i],
                "split": "test",
                "ground_truth_label": int(labels[i]),
                "predicted_label": int(pred[i]),
                "fake_probability": float(prob[i]),
            })
    return recs


if NEED_INFER:
    if not torch.cuda.is_available():
        print("WARNING: scoring checkpoints on CPU will be slow. Settings → GPU T4.")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WILD_ROOT = discover_wild_root()
    test_df = list_test(WILD_ROOT)
    print("official Test images:", len(test_df), "root", WILD_ROOT)
    image_size, batch_size = 600, 4
    ck0 = _torch_load(ckpt_files[NEED_INFER[0]])
    cfg0 = ck0.get("config") or {}
    image_size = int(cfg0.get("image_size") or 600)
    print("eval image_size", image_size, "from", NEED_INFER[0], "epoch", ck0.get("epoch"))
    loader = DataLoader(WildFaceDataset(test_df, image_size), batch_size=batch_size, shuffle=False, num_workers=2 if os.name != "nt" else 0)
    for exp in NEED_INFER:
        print("SCORING", exp, ckpt_files[exp])
        ck = _torch_load(ckpt_files[exp])
        model = build_eval_model(exp).to(DEVICE)
        model.load_state_dict(ck["model_state_dict"])
        recs = score_loader(model, loader, DEVICE)
        dest = OUT_DIR / "predictions" / f"{exp}_predictions.csv"
        pd.DataFrame(recs).to_csv(dest, index=False)
        pred_files[exp] = dest
        acc = float((pd.DataFrame(recs).ground_truth_label == pd.DataFrame(recs).predicted_label).mean()) if recs else float("nan")
        print(exp, "epoch", ck.get("epoch"), "TEST acc", acc, "wrote", dest)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
else:
    print("All variants already have prediction CSVs. Skipping checkpoint Test.")
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
        fake = df[df.ground_truth_label == 1]
        row["miou_fake_only"] = float(fake["iou"].mean()) if len(fake) else float("nan")
        row["dice_fake_only"] = float(fake["dice"].mean()) if len(fake) and "dice" in fake.columns else float("nan")
    else:
        row["miou_fake_only"] = "N/A"
        row["dice_fake_only"] = "N/A"
    return row


META = {
    "baseline_effb7_wild": {"multi_stream": False, "segmentation": False},
    "multistream_no_seg_wild": {"multi_stream": True, "segmentation": False},
    "seg_no_multistream_wild": {"multi_stream": False, "segmentation": True},
    "full_veritas_wild": {"multi_stream": True, "segmentation": True},
}

rows = []
for exp in EXPECTED:
    if exp not in frames:
        rows.append({"experiment": exp, "status": "NOT RUN", **META[exp]})
        continue
    row = {"experiment": exp, "status": "ok", **META[exp], **cls_metrics(frames[exp])}
    rows.append(row)

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working"
out_dir.mkdir(parents=True, exist_ok=True)
summary.to_csv(out_dir / "wild_compare_summary.csv", index=False)
print("wrote", out_dir / "wild_compare_summary.csv")
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
    ("baseline_effb7_wild", "multistream_no_seg_wild"),
    ("baseline_effb7_wild", "seg_no_multistream_wild"),
    ("baseline_effb7_wild", "full_veritas_wild"),
    ("multistream_no_seg_wild", "full_veritas_wild"),
    ("seg_no_multistream_wild", "full_veritas_wild"),
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

(out_dir / "wild_statistical_tests.json").write_text(json.dumps(stat_rows, indent=2, default=str), encoding="utf-8")
print("wrote", out_dir / "wild_statistical_tests.json")
"""
        ),
        md(
            r"""
## How to read this

- Epoch-1 is a valid compare: attach those outputs even if Test was skipped in training.
- If `PROTOCOL_HASH` differed, stop and rerun the mismatched notebook with the agreed CONFIG (`full`, seed 42, size 600).
- Pairing is by `file_name` on the official Wild Test folder.
- `miou_fake_only` / `dice_fake_only` exist only for the two segmentation variants, and they score **surrogate ovals**, not GT polygons.
- This table is not comparable to `notebooks/VERITAS_05_compare.ipynb`.
"""
        ),
    ]
    nb = _nb(cells)
    nb["metadata"]["accelerator"] = "GPU"
    return nb


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        dest = DEST_DIR / v["file"]
        nbf.write(build_training_nb(v), dest)
        print("wrote", dest, "exp", v["exp"])
    dest = DEST_DIR / "WILD_05_compare.ipynb"
    nbf.write(build_compare_nb(), dest)
    print("wrote", dest)
    for v in B0_VARIANTS:
        out = Path(v.get("out_dir") or DEST_DIR)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / v["file"]
        nbf.write(build_training_nb(v), dest)
        print("wrote", dest, "exp", v["exp"])


if __name__ == "__main__":
    main()
