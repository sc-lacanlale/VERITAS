"""Build notebooks-ow/ — generate Wild masks, then train on Wild plus those masks.

OW_00 paints one mask per OpenForensics Wild image with the current Full checkpoint.
OW_01–OW_04 train on those images and masks only. OW_05 reads their result files.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

DEST = Path("notebooks-ow")

VARIANTS = [
    {
        "file": "OW_01_baseline.ipynb",
        "exp": "baseline_effb7_ow",
        "title": "Baseline EfficientNet-B7",
        "who": "Member A",
        "n": "1 of 4",
        "multi_stream": False,
        "segmentation": False,
        "legacy_baseline": True,
        "blurb": "RGB EfficientNet-B7. The generated masks are not used in the loss.",
    },
    {
        "file": "OW_02_multistream.ipynb",
        "exp": "multistream_no_seg_ow",
        "title": "Multi-stream classification (no segmentation)",
        "who": "Member B",
        "n": "2 of 4",
        "multi_stream": True,
        "segmentation": False,
        "legacy_baseline": False,
        "blurb": "RGB + SRM + log-FFT. The generated masks are not used in the loss.",
    },
    {
        "file": "OW_03_segmentation.ipynb",
        "exp": "seg_no_multistream_ow",
        "title": "Segmentation / MTL (single RGB stream)",
        "who": "Member C",
        "n": "3 of 4",
        "multi_stream": False,
        "segmentation": True,
        "legacy_baseline": False,
        "blurb": "RGB EfficientNet-B7 + DeepLabV3+. Segmentation loss uses the generated mask on every image.",
    },
    {
        "file": "OW_04_full.ipynb",
        "exp": "full_veritas_ow",
        "title": "Full VERITAS",
        "who": "Member D",
        "n": "4 of 4",
        "multi_stream": True,
        "segmentation": True,
        "legacy_baseline": False,
        "blurb": "Nine-channel stem + DeepLabV3+. Segmentation loss uses the generated mask on every image.",
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


MODEL_SRC = r'''
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
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b7(weights=weights)
        self.features = net.features
        self.cls_head = ClassificationHead(2560)
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
    def __init__(self, multi_stream=False, segmentation=False, pretrained=True):
        super().__init__()
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
        return BaselineEffNet(pretrained=CONFIG["use_pretrained"]).to(DEVICE)
    m = VERITASModel(
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
'''


def build_generate_nb():
    cells = [
        md(
            """
# OpenForensics Wild — generate one mask per image

This notebook does **not** train. It runs the current Full checkpoint on every
OpenForensics Wild image and saves the predicted mask.

The mask is the model’s sigmoid output, threshold 0.5, on the same crop the
trainer will use: largest Haar face, 10% margin, otherwise the whole frame,
letterboxed to 600×600. The crop box is stored in `index.csv` so training does
not run Haar again.

## Attach

1. New GPU notebook. Accelerator **GPU T4**. Internet Off is fine.
2. Dataset `ahmedgamal/openforensics-wild-dataset`.
3. The current Full checkpoint: `full_veritas_defacto_best.pt` or `.zip` (the `OF_04_full` output).
4. On every run after the first: **Add Input → Notebooks** → this notebook’s latest successful version.

## Where the files go

`/kaggle/working/ow_masks/`

| File | What it is |
| --- | --- |
| `manifest.csv` | Every Wild image, in a stable order |
| `index.csv` | Images this shard has finished. One row per image |
| `progress.json` | Count, images per hour, whether to resume |
| `masks/{train,val,test}/{real,fake}/{image_id}.png` | 600×600 mask, 0 or 255 |

`image_id` is the first 16 hex characters of SHA-1(`train/fake/filename`).
The same file name in two splits does not collide.

## Staying under 12 hours

`time_budget_hours` is **10.5**. The run stops before that, flushes `index.csv`,
and leaves the rest for the next session. Attach this output and Run All again
with the **same** `shard_index` and `shard_count`. Finished images are skipped.

After the first 64 new images the log prints images/hour and how many more runs
the remainder needs. One T4 pass is enough when that number is 1.

To use more than one GPU, set `shard_count` **before the first full run** and
start one session per shard (`shard_index` 0, 1, …). Do not change `shard_count`
after an `index.csv` exists. Training attaches every shard output.

Set `run_mode` to `full` for the real pass. `smoke` writes 8 masks only.
"""
        ),
        md("## 1. Setup"),
        cd(
            r"""
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

NOTEBOOK_START_TIME = time.time()
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
"""
        ),
        md("## 2. Find the Wild folders and the mask output"),
        cd(
            r"""
def discover_wild_root() -> Path:
    roots = []
    if Path("/kaggle/input").exists():
        roots.append(Path("/kaggle/input"))
    roots.extend([Path.cwd() / "openforensics-wild-dataset", Path.cwd() / "Dataset", Path.cwd()])
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            if "ow_masks" in p.parts:
                continue
            kids = {d.lower() for d in dirnames}
            if {"train", "validation", "test"} <= kids or {"train", "val", "test"} <= kids:
                return p
            if p.name.lower() == "dataset" and ("train" in kids or "test" in kids):
                return p
    raise FileNotFoundError(
        "Attach ahmedgamal/openforensics-wild-dataset (Dataset/Train|Validation|Test/Real|Fake)."
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
IN_KAGGLE = Path("/kaggle/working").exists()
WORKING = Path("/kaggle/working") if IN_KAGGLE else Path.cwd() / "kaggle_working"
MASK_ROOT = WORKING / "ow_masks"
MASK_ROOT.mkdir(parents=True, exist_ok=True)
print("wild root:", WILD_ROOT)
print("mask root:", MASK_ROOT)
"""
        ),
        md("## 3. Configuration"),
        cd(
            r"""
CONFIG = {
    "run_mode": "smoke",
    "image_size": 600,
    "batch_size": 4,
    "mask_threshold": 0.5,
    "face_margin": 0.10,
    "time_budget_hours": 10.5,
    "shard_index": 0,
    "shard_count": 1,
    "max_images": None,
    "flush_every": 20,
    "use_pretrained": False,
}
EXP_CFG = {"multi_stream": True, "segmentation": True, "legacy_baseline": False}
if CONFIG["run_mode"] == "smoke":
    CONFIG["max_images"] = 8
    CONFIG["batch_size"] = 2
    print("SMOKE: 8 images. Set run_mode='full' to walk every Wild image.")
if CONFIG["run_mode"] == "full" and not torch.cuda.is_available():
    raise RuntimeError("Full mask generation needs GPU T4.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
print("device:", DEVICE)
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
"""
        ),
        md("## 4. Manifest, finished rows, and the 10.5 hour ledger"),
        cd(
            r"""
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABELS = {"real": 0, "fake": 1}
INDEX_COLS = [
    "image_id", "rel_key", "split", "label", "class_name", "file_name",
    "crop_x0", "crop_y0", "crop_x1", "crop_y1", "face_found",
    "mask_file", "positive_frac", "status", "error", "teacher",
]


def rel_key(split: str, class_name: str, file_name: str) -> str:
    return f"{split}/{class_name}/{file_name}"


def image_id_for(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _class_dir(split_dir: Path, name: str):
    for a in (name, name.capitalize(), name.upper(), name.lower()):
        p = split_dir / a
        if p.is_dir():
            return p
    return None


def build_manifest() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        split_dir = _split_dir(WILD_ROOT, split)
        for cname, lab in LABELS.items():
            cdir = _class_dir(split_dir, cname)
            if cdir is None:
                print("WARNING: missing", split, cname)
                continue
            for p in sorted(cdir.iterdir()):
                if p.is_file() and p.suffix.lower() in EXTS:
                    key = rel_key(split, cname, p.name)
                    rows.append({
                        "image_id": image_id_for(key),
                        "rel_key": key,
                        "split": split,
                        "label": lab,
                        "class_name": cname,
                        "file_name": p.name,
                        "src_path": str(p),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No Wild images found.")
    if df["image_id"].duplicated().any():
        raise RuntimeError("image_id collision; refuse to continue.")
    df = df.sort_values("rel_key").reset_index(drop=True)
    df.to_csv(MASK_ROOT / "manifest.csv", index=False)
    print("manifest", len(df), {s: int((df.split == s).sum()) for s in ("train", "val", "test")})
    return df


def _search_roots():
    roots = []
    for p in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd(), WORKING):
        if p.exists():
            roots.append(p)
    return roots


def load_prior() -> pd.DataFrame:
    frames = []
    seen = set()
    for root in _search_roots():
        for dirpath, _, filenames in os.walk(root):
            if "index.csv" not in filenames:
                continue
            path = Path(dirpath) / "index.csv"
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                df = pd.read_csv(path, dtype=str)
            except Exception as e:
                print("skip index", path, e)
                continue
            if not {"image_id", "rel_key", "mask_file", "status"} <= set(df.columns):
                continue
            df["_root"] = str(path.parent)
            frames.append(df)
            print("prior index", path, "rows", len(df))
    if not frames:
        return pd.DataFrame(columns=INDEX_COLS + ["_root"])
    return pd.concat(frames, ignore_index=True)


def png_for(row):
    p = Path(str(row["_root"])) / str(row["mask_file"])
    return p if p.is_file() else None


MANIFEST = build_manifest()
PRIOR = load_prior()
TEACHER_NAME = "pending"
"""
        ),
        md("## 5. Model (same Full VERITAS trunk as the current checkpoint)"),
        cd(MODEL_SRC),
        md("## 6. Load the current Full checkpoint"),
        cd(
            r"""
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
    cache = WORKING / (path.stem.replace(" ", "_") + "_repacked.pt")
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
                inner = name[len(prefix):]
                if not inner or inner.endswith("/"):
                    continue
                zout.writestr(inner, zin.read(name))
    return cache


def find_teacher() -> Path:
    order = [
        "full_veritas_defacto_best.pt",
        "full_veritas_defacto_best.zip",
        "full_veritas_defacto_last.pt",
    ]
    found = {name: [] for name in order}
    for root in _search_roots():
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn in found:
                    found[fn].append(Path(dirpath) / fn)
    for name in order:
        if found[name]:
            return found[name][0]
    raise FileNotFoundError(
        "Attach full_veritas_defacto_best.pt or .zip from OF_04_full. "
        "The baseline checkpoint has no mask head."
    )


teacher_path = find_teacher()
print("teacher file:", teacher_path)
ck = _torch_load(materialize_checkpoint(teacher_path))
state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
exp_cfg = (ck.get("exp_cfg") if isinstance(ck, dict) else None) or {}
if exp_cfg and (not exp_cfg.get("segmentation", True) or not exp_cfg.get("multi_stream", True)):
    raise RuntimeError(f"Checkpoint is not Full VERITAS: {exp_cfg}")
model = build_model()
model.load_state_dict(state, strict=True)
model.eval()
TEACHER_NAME = teacher_path.name
print("loaded", TEACHER_NAME, "epoch", None if not isinstance(ck, dict) else ck.get("epoch"))
"""
        ),
        md("## 7. Paint masks until the time budget"),
        cd(
            r"""
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def letterbox_rgb(image_rgb, size: int):
    h, w = image_rgb.shape[:2]
    scale = min(size / max(h, 1), size / max(w, 1))
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top: top + nh, left: left + nw] = resized
    return canvas


def bbox_from_mask(mask, margin: float):
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


def bbox_from_haar(rgb, margin: float):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    dummy = np.zeros(rgb.shape[:2], dtype=np.uint8)
    dummy[y: y + h, x: x + w] = 1
    return bbox_from_mask(dummy, margin)


def shard_of(image_id: str) -> int:
    return int(image_id[:8], 16) % int(CONFIG["shard_count"])


def prior_done_for_shard() -> dict:
    done = {}
    if PRIOR.empty:
        return done
    teachers = set()
    for rec in PRIOR.to_dict("records"):
        if str(rec.get("status")) != "ok":
            continue
        iid = str(rec["image_id"])
        if shard_of(iid) != int(CONFIG["shard_index"]):
            continue
        png = png_for(rec)
        if png is None:
            continue
        teacher = str(rec.get("teacher") or "")
        if teacher and teacher not in ("nan", "None"):
            teachers.add(teacher)
        done[iid] = rec
    if len(teachers) > 1:
        raise RuntimeError(f"Prior masks come from more than one checkpoint: {teachers}")
    if teachers and TEACHER_NAME not in teachers and "pending" not in teachers:
        # The file name of a resumed zip and a .pt can differ. Compare the stem family.
        def family(name: str) -> str:
            return name.replace("_repacked", "").replace(".zip", "").replace(".pt", "")
        if {family(t) for t in teachers} != {family(TEACHER_NAME)}:
            raise RuntimeError(
                f"Prior masks were painted by {teachers}, this run loaded {TEACHER_NAME}. "
                "Do not mix checkpoints in one mask set."
            )
    return done


DONE = prior_done_for_shard()
SEEN = set(DONE)
if not PRIOR.empty:
    for rec in PRIOR.to_dict("records"):
        iid = str(rec.get("image_id") or "")
        if not iid or shard_of(iid) != int(CONFIG["shard_index"]):
            continue
        if str(rec.get("status")) == "error":
            SEEN.add(iid)
print("already finished in this shard:", len(DONE), "| rows that will not be retried:", len(SEEN))

copied = 0
for iid, rec in DONE.items():
    src = png_for(rec)
    dst = MASK_ROOT / str(rec["mask_file"])
    if src is None or dst.exists() or src.resolve() == dst.resolve():
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied += 1
print("copied earlier masks into this output:", copied)


def write_index(rows: list):
    df = pd.DataFrame(rows)
    for col in INDEX_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[INDEX_COLS].drop_duplicates("image_id", keep="last")
    tmp = MASK_ROOT / "index.csv.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(MASK_ROOT / "index.csv")
    return df


def write_progress(payload: dict):
    tmp = MASK_ROOT / "progress.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(MASK_ROOT / "progress.json")


shard_df = MANIFEST[MANIFEST.image_id.map(shard_of) == int(CONFIG["shard_index"])].reset_index(drop=True)
if CONFIG.get("max_images"):
    shard_df = shard_df.iloc[: int(CONFIG["max_images"])].reset_index(drop=True)
todo = shard_df[~shard_df.image_id.isin(SEEN)].reset_index(drop=True)
print(f"this shard {CONFIG['shard_index']}/{CONFIG['shard_count']}: {len(shard_df)} images, {len(todo)} still to paint")

budget_s = float(CONFIG["time_budget_hours"]) * 3600.0
size = int(CONFIG["image_size"])
thresh = float(CONFIG["mask_threshold"])
margin = float(CONFIG["face_margin"])
rows = []
for rec in DONE.values():
    item = {k: rec.get(k, "") for k in INDEX_COLS}
    item["teacher"] = TEACHER_NAME
    rows.append(item)
if not PRIOR.empty:
    for rec in PRIOR.to_dict("records"):
        iid = str(rec.get("image_id") or "")
        if iid in DONE or shard_of(iid) != int(CONFIG["shard_index"]):
            continue
        if str(rec.get("status")) == "error":
            item = {k: rec.get(k, "") for k in INDEX_COLS}
            item["teacher"] = TEACHER_NAME
            rows.append(item)

pending = []
n_new = 0
stopped = "finished"
t_paint = time.time()


def flush_batch():
    global n_new
    if not pending:
        return
    batch = torch.stack([p["tensor"] for p in pending]).to(DEVICE)
    with torch.no_grad():
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(batch)
        else:
            out = model(batch)
    if "segmentation_prob" not in out:
        raise RuntimeError("Loaded model has no segmentation head.")
    probs = out["segmentation_prob"].detach().float().cpu().numpy()
    for i, item in enumerate(pending):
        pred = (probs[i, 0] >= thresh).astype(np.uint8) * 255
        dst = MASK_ROOT / item["mask_file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), pred):
            rows.append({**item, "positive_frac": "", "status": "error", "error": "imwrite failed", "teacher": TEACHER_NAME})
        else:
            rows.append({
                **item,
                "positive_frac": f"{float((pred > 0).mean()):.6f}",
                "status": "ok",
                "error": "",
                "teacher": TEACHER_NAME,
            })
        n_new += 1
    pending.clear()
    del batch, out, probs


def record_error(row, message):
    rows.append({
        "image_id": row.image_id,
        "rel_key": row.rel_key,
        "split": row.split,
        "label": int(row.label),
        "class_name": row.class_name,
        "file_name": row.file_name,
        "crop_x0": "", "crop_y0": "", "crop_x1": "", "crop_y1": "",
        "face_found": 0,
        "mask_file": "",
        "positive_frac": "",
        "status": "error",
        "error": message,
        "teacher": TEACHER_NAME,
    })


for n_seen, row in enumerate(todo.itertuples(index=False), start=1):
    if time.time() - NOTEBOOK_START_TIME > budget_s:
        stopped = "time_budget"
        break
    bgr = cv2.imread(row.src_path, cv2.IMREAD_COLOR)
    if bgr is None:
        record_error(row, "unreadable")
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    box = bbox_from_haar(rgb, margin)
    face_found = 1 if box is not None else 0
    if box is None:
        box = (0, 0, rgb.shape[1], rgb.shape[0])
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        record_error(row, "empty crop")
        continue
    canvas = letterbox_rgb(crop, size)
    x = canvas.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
    tensor = torch.from_numpy(np.transpose(x, (2, 0, 1)).copy())
    pending.append({
        "tensor": tensor,
        "image_id": row.image_id,
        "rel_key": row.rel_key,
        "split": row.split,
        "label": int(row.label),
        "class_name": row.class_name,
        "file_name": row.file_name,
        "crop_x0": int(x0), "crop_y0": int(y0), "crop_x1": int(x1), "crop_y1": int(y1),
        "face_found": face_found,
        "mask_file": f"masks/{row.split}/{row.class_name}/{row.image_id}.png",
    })
    if len(pending) >= int(CONFIG["batch_size"]):
        flush_batch()
    if n_new and n_new % int(CONFIG["flush_every"]) == 0 and not pending:
        write_index(rows)
        rate = n_new / max(time.time() - t_paint, 1e-6)
        remain = len(todo) - n_seen
        print(
            f"painted {n_new} this run | shard done {len(DONE) + n_new}/{len(shard_df)} | "
            f"{rate * 3600:.0f} images/hour | remaining hours {remain / rate / 3600:.2f}"
        )
        if n_new == int(CONFIG["flush_every"]) or n_new == 64:
            hours_needed = (len(todo) / rate) / 3600.0
            runs = int(np.ceil(hours_needed / float(CONFIG["time_budget_hours"])))
            print(f"estimate for the images that were still queued at the start: {hours_needed:.2f} h, about {runs} run(s) at this rate")
            if runs > 1 and int(CONFIG["shard_count"]) == 1 and len(DONE) == 0:
                print(
                    "This shard will not finish in one session. Let this run hit the time budget, "
                    "then attach this output and Run All again with the same shard_index and shard_count. "
                    "Do not change shard_count after index.csv exists."
                )

flush_batch()
index_df = write_index(rows)
elapsed_h = (time.time() - NOTEBOOK_START_TIME) / 3600.0
n_ok = int((index_df.status == "ok").sum()) if len(index_df) else 0
rate = n_new / max(time.time() - t_paint, 1e-6)
remain = max(len(shard_df) - n_ok, 0)
payload = {
    "teacher": TEACHER_NAME,
    "shard_index": int(CONFIG["shard_index"]),
    "shard_count": int(CONFIG["shard_count"]),
    "n_manifest": int(len(MANIFEST)),
    "n_shard": int(len(shard_df)),
    "n_ok_this_shard": n_ok,
    "n_painted_this_run": int(n_new),
    "images_per_hour": None if n_new == 0 else round(rate * 3600.0, 1),
    "hours_used": round(elapsed_h, 3),
    "remaining_this_shard": int(remain),
    "stopped_reason": stopped if remain else "finished",
    "resume": remain > 0,
}
write_progress(payload)
print(json.dumps(payload, indent=2))
if CONFIG["run_mode"] == "smoke":
    print("SMOKE only painted a handful of images. progress.json does not mean the full Wild set is done.")
elif payload["resume"]:
    print("RESUME: attach this notebook output and Run All again. Keep shard_index and shard_count.")
else:
    print("SHARD COMPLETE. Training attaches this output (and the other shard outputs, if shard_count > 1).")
"""
        ),
    ]
    return _nb(cells)


def build_training_nb(v: dict):
    exp = v["exp"]
    ms = "True" if v["multi_stream"] else "False"
    sg = "True" if v["segmentation"] else "False"
    lg = "True" if v["legacy_baseline"] else "False"
    seg_line = (
        "Segmentation loss uses the generated mask on every image. Dice on Test is agreement with that mask, not a hand-labeled polygon."
        if v["segmentation"]
        else "This notebook does not use the mask in the loss. The mask files still have to be present so the four notebooks share one image list."
    )
    cells = [
        md(
            f"""
# OpenForensics Wild + generated masks — {v['title']}

**Assigned to:** {v['who']} ({v['n']})
**This notebook trains only:** `{exp}`

{v['blurb']}
{seg_line}

Images and labels come from OpenForensics Wild `Real` / `Fake` folders.
The mask for each image is the PNG written by `OW_00_generate_masks.ipynb`.
The crop is the box stored in `index.csv` (the box the mask was painted on).

## Attach

1. New GPU notebook. Do not import this into an OF or WILD training session.
2. Dataset `ahmedgamal/openforensics-wild-dataset`.
3. The finished `OW_00` output. Kaggle shows that output as `_output_.zip`. This notebook reads `ow_masks/index.csv` and the mask PNGs inside that zip. A loose `ow_masks/` folder works too.
4. On later epochs: this notebook’s own latest output (`{exp}_last.pt`).
5. GPU T4, Internet On for the first run (ImageNet B7 download).

Do not attach the Full checkpoint as this run’s weights. Training starts from ImageNet.
Do not attach a different study’s `{exp}` file; the experiment name is `{exp}`.

## One epoch per run

Each Save & Run All trains one more epoch, then scores Test.
`time_budget_hours` is 11. A mid-epoch checkpoint is written every 30 minutes.
The next run must say `START OF RUN | {exp}: N/3`.

Set `run_mode` to `full` after `OW_00` has finished every image.
"""
        ),
        md("## 1. Environment"),
        cd(
            r"""
import importlib
import subprocess
import sys

def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

missing = []
for mod, pip_name in {
    "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "tqdm": "tqdm",
}.items():
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pip_name)
if missing:
    _pip_install(missing)

import time
import torch
NOTEBOOK_START_TIME = time.time()
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
        ),
        md("## 2. Imports"),
        cd(
            r"""
import gc
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7
from tqdm.auto import tqdm
"""
        ),
        md("## 3. Wild images and the generated-mask folder"),
        cd(
            r"""
def discover_wild_root() -> Path:
    roots = []
    if Path("/kaggle/input").exists():
        roots.append(Path("/kaggle/input"))
    roots.extend([Path.cwd() / "openforensics-wild-dataset", Path.cwd() / "Dataset", Path.cwd()])
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            if "ow_masks" in p.parts:
                continue
            kids = {d.lower() for d in dirnames}
            if {"train", "validation", "test"} <= kids or {"train", "val", "test"} <= kids:
                return p
            if p.name.lower() == "dataset" and ("train" in kids or "test" in kids):
                return p
    raise FileNotFoundError("Attach ahmedgamal/openforensics-wild-dataset.")


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
IN_KAGGLE = Path("/kaggle/working").exists()
WORKING = Path("/kaggle/working") if IN_KAGGLE else Path.cwd() / "kaggle_working"
WORKING.mkdir(parents=True, exist_ok=True)
for sub in ("checkpoints", "predictions", "logs"):
    (WORKING / sub).mkdir(parents=True, exist_ok=True)
print("wild root:", WILD_ROOT)
print("working:", WORKING)
"""
        ),
        md("## 4. Configuration"),
        cd(
            f"""
CONFIG = {{
    "run_mode": "smoke",
    "seed": 42,
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
}}
if CONFIG["run_mode"] == "smoke":
    CONFIG["max_per_split"] = 64
    CONFIG["epochs"] = 1
    print("SMOKE: up to 64 images per class per split, 1 epoch. Set run_mode='full' after the mask set is complete.")
if CONFIG["run_mode"] == "full" and not torch.cuda.is_available():
    raise RuntimeError("Full training needs GPU T4.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("THIS NOTEBOOK TRAINS:", CONFIG["experiment"], EXP_CFG)
_protocol = {{k: CONFIG[k] for k in [
    "run_mode", "seed", "image_size", "classification_threshold",
    "epochs", "batch_size", "learning_rate", "lambda_cls", "lambda_seg", "dice_weight",
]}}
PROTOCOL_HASH = hashlib.sha256(json.dumps(_protocol, sort_keys=True, default=str).encode()).hexdigest()[:16]
print("PROTOCOL_HASH:", PROTOCOL_HASH)
(WORKING / "logs" / "protocol.json").write_text(
    json.dumps({{"hash": PROTOCOL_HASH, "protocol": _protocol, "experiment": CONFIG["experiment"]}}, indent=2),
    encoding="utf-8",
)
"""
        ),
        md("## 5. Join each Wild image to its generated mask"),
        cd(
            r"""
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABELS = {"real": 0, "fake": 1}


def rel_key(split: str, class_name: str, file_name: str) -> str:
    return f"{split}/{class_name}/{file_name}"


def _class_dir(split_dir: Path, name: str):
    for a in (name, name.capitalize(), name.upper(), name.lower()):
        p = split_dir / a
        if p.is_dir():
            return p
    return None


def list_images(split_name: str) -> pd.DataFrame:
    split_dir = _split_dir(WILD_ROOT, split_name)
    rows = []
    for cname, lab in LABELS.items():
        cdir = _class_dir(split_dir, cname)
        if cdir is None:
            print("WARNING: missing", split_name, cname)
            continue
        for p in sorted(cdir.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTS:
                rows.append({
                    "path": str(p),
                    "split": split_name,
                    "label": lab,
                    "class_name": cname,
                    "file_name": p.name,
                    "rel_key": rel_key(split_name, cname, p.name),
                })
    return pd.DataFrame(rows)


_MASK_COLS = {"image_id", "rel_key", "mask_file", "status", "crop_x0"}
_ZIP_MEMBERS = {}


def _take_index(df, label):
    if not _MASK_COLS <= set(df.columns):
        return None
    print("mask index", label, "rows", len(df))
    return df


def _frames_from_zip(path: Path):
    try:
        zin = zipfile.ZipFile(path)
    except Exception as e:
        print("skip zip", path, e)
        return []
    names = zin.namelist()
    indexes = [n for n in names if n.replace("\\", "/").rstrip("/").endswith("index.csv")]
    frames = []
    members = set(names)
    for index_name in indexes:
        try:
            df = pd.read_csv(io.BytesIO(zin.read(index_name)), dtype=str)
        except Exception as e:
            print("skip", path, index_name, e)
            continue
        got = _take_index(df, f"{path} :: {index_name}")
        if got is None:
            continue
        norm = index_name.replace("\\", "/")
        prefix = norm[: norm.rfind("/") + 1] if "/" in norm else ""
        got["_zip"] = str(path)
        got["_zip_prefix"] = prefix
        frames.append(got)
        _ZIP_MEMBERS[str(path)] = members
    zin.close()
    return frames


def load_mask_table() -> pd.DataFrame:
    frames = []
    roots = []
    for p in (Path("/kaggle/input"), Path("/kaggle/working"), WORKING, Path.cwd()):
        if p.exists():
            roots.append(p)
    seen = set()
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if "index.csv" in filenames:
                path = Path(dirpath) / "index.csv"
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    try:
                        df = pd.read_csv(path, dtype=str)
                    except Exception as e:
                        print("skip", path, e)
                        df = None
                    if df is not None:
                        got = _take_index(df, path)
                        if got is not None:
                            got["_root"] = str(path.parent)
                            frames.append(got)
            for fn in filenames:
                if not fn.lower().endswith(".zip"):
                    continue
                path = Path(dirpath) / fn
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                frames.extend(_frames_from_zip(path))
    if not frames:
        raise FileNotFoundError(
            "No ow_masks/index.csv found. Attach the finished OW_00 notebook. "
            "Its Kaggle output is _output_.zip, and that zip must contain ow_masks/index.csv."
        )
    all_df = pd.concat(frames, ignore_index=True)
    kept = []
    used = set()
    for rec in all_df.to_dict("records"):
        iid = str(rec["image_id"])
        if iid in used or str(rec.get("status")) != "ok":
            continue
        zip_path = rec.get("_zip")
        if isinstance(zip_path, str) and zip_path and zip_path.lower() != "nan":
            prefix = rec.get("_zip_prefix")
            prefix = "" if not isinstance(prefix, str) or prefix.lower() == "nan" else prefix
            member = prefix + str(rec["mask_file"]).replace("\\", "/")
            members = _ZIP_MEMBERS.get(zip_path)
            if members is not None and member not in members:
                continue
            rec["mask_path"] = "zip:" + zip_path + "::" + member
        else:
            png = Path(str(rec["_root"])) / str(rec["mask_file"])
            if not png.is_file():
                continue
            rec["mask_path"] = str(png)
        kept.append(rec)
        used.add(iid)
    if not kept:
        raise FileNotFoundError("index.csv rows were found, but none of the PNG files are present.")
    out = pd.DataFrame(kept).drop_duplicates("rel_key", keep="last")
    print("masks ready", len(out), "teachers", sorted(set(out.teacher.dropna().astype(str))) if "teacher" in out.columns else [])
    return out


MASKS = load_mask_table()
INDEX = {}
for split in ("train", "val", "test"):
    images = list_images(split)
    if images.empty:
        raise RuntimeError(f"{split} has no images")
    merged = images.merge(
        MASKS[["rel_key", "mask_path", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "status", "positive_frac"]],
        on="rel_key",
        how="left",
    )
    have = merged["mask_path"].notna()
    print(f"{split}: images {len(merged)}  with mask {int(have.sum())}  missing {int((~have).sum())}")
    if CONFIG["run_mode"] == "full" and (~have).any():
        missing = merged.loc[~have, "rel_key"].head(8).tolist()
        raise RuntimeError(
            f"{split} is missing {int((~have).sum())} generated masks. Finish OW_00 before full training. Example: {missing}"
        )
    ready = merged.loc[have].copy()
    if CONFIG.get("max_per_split"):
        parts = []
        rng = np.random.RandomState(CONFIG["seed"] + {"train": 1, "val": 2, "test": 3}[split])
        for _, g in ready.groupby("label"):
            n = min(int(CONFIG["max_per_split"]), len(g))
            parts.append(g.sample(n=n, random_state=rng) if n < len(g) else g)
        ready = pd.concat(parts)
    if ready.empty:
        raise RuntimeError(f"{split} has no generated masks. Run OW_00 first.")
    INDEX[split] = ready.reset_index(drop=True)
    print(f"  using {len(INDEX[split])}  fake={int((INDEX[split].label == 1).sum())} real={int((INDEX[split].label == 0).sum())}")
"""
        ),
        md("## 6. Dataset (saved crop + saved 600 mask)"),
        cd(
            r"""
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def letterbox_rgb(image_rgb, size: int):
    h, w = image_rgb.shape[:2]
    scale = min(size / max(h, 1), size / max(w, 1))
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top: top + nh, left: left + nw] = resized
    return canvas


_ZIP_HANDLES = {}


def read_gray_mask(spec: str):
    spec = str(spec)
    if spec.startswith("zip:"):
        zpath, member = spec[4:].split("::", 1)
        zin = _ZIP_HANDLES.get(zpath)
        if zin is None:
            zin = zipfile.ZipFile(zpath)
            _ZIP_HANDLES[zpath] = zin
        data = np.frombuffer(zin.read(member), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(spec, cv2.IMREAD_GRAYSCALE)


class GeneratedMaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        bgr = cv2.imread(row.path, cv2.IMREAD_COLOR)
        if bgr is None:
            rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            x0 = y0 = 0
            x1 = y1 = self.size
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            x0 = int(float(row.crop_x0)); y0 = int(float(row.crop_y0))
            x1 = int(float(row.crop_x1)); y1 = int(float(row.crop_y1))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
        crop = rgb[y0:y1, x0:x1]
        mask = read_gray_mask(row.mask_path)
        if mask is None:
            mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if mask.shape != (self.size, self.size):
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment and random.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if self.augment and random.random() < 0.5:
            crop = np.clip(crop.astype(np.float32) * (1.0 + random.uniform(-0.15, 0.15)) + random.uniform(-12, 12), 0, 255).astype(np.uint8)
        canvas = letterbox_rgb(crop, self.size)
        x = canvas.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
        x = np.transpose(x, (2, 0, 1))
        target = (mask.astype(np.float32) > 127).astype(np.float32)
        return {
            "image": torch.from_numpy(x.copy()),
            "mask": torch.from_numpy(target[None].copy()),
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
        return iter(order[self.start_index:])

    def __len__(self):
        return max(self.n - self.start_index, 0)


def make_loader(df, shuffle, augment):
    ds = GeneratedMaskDataset(df, CONFIG["image_size"], augment=augment)
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


if int(CONFIG["image_size"]) != 600:
    raise RuntimeError("Generated masks are 600×600. Leave image_size at 600.")
TRAIN_LOADER = make_loader(INDEX["train"], True, True)
VAL_LOADER = make_loader(INDEX["val"], False, False)
TEST_LOADER = make_loader(INDEX["test"], False, False)
print("train batches", len(TRAIN_LOADER), "val", len(VAL_LOADER), "test", len(TEST_LOADER))
"""
        ),
        md("## 7. Model"),
        cd(MODEL_SRC),
        cd(
            r"""
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
        md("## 8. Loss"),
        cd(
            r"""
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
                raise ValueError("masks required")
            l_seg = self.bce_seg(outputs["segmentation_logits"], masks) + self.dice_weight * self.dice(
                outputs["segmentation_logits"], masks
            )
        total = self.lambda_cls * l_cls + (self.lambda_seg * l_seg if self.segmentation else 0.0)
        return {"total": total, "classification": l_cls.detach(), "segmentation": l_seg.detach() if torch.is_tensor(l_seg) else l_seg}
"""
        ),
        md("## 9. Checkpoint resume (one epoch per run)"),
        cd(
            f"""
EXP = CONFIG["experiment"]


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_progress(path: Path):
    try:
        ck = _torch_load(path)
        partial = ck.get("partial") or {{}}
        return (int(ck.get("epoch") or 0), int(partial.get("step", 0)))
    except Exception as e:
        print("could not read", path, e)
        return (-1, -1)


def stage_checkpoints_from_input():
    if not CONFIG.get("resume_from_input", True):
        return
    ckpt_dir = WORKING / "checkpoints"
    names = {{f"{{EXP}}_last.pt": [], f"{{EXP}}_best.pt": []}}
    root = Path("/kaggle/input")
    if not root.exists():
        return
    for dirpath, _, filenames in os.walk(root):
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
        {{
            "epoch": epoch,
            "partial": partial,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_metric": best_metric,
            "experiment_name": EXP,
            "history": history,
            "epoch_seconds": epoch_seconds,
            "config": CONFIG,
            "exp_cfg": EXP_CFG,
        }},
        tmp,
    )
    tmp.replace(path)


def load_if_any(path, model, optimizer=None):
    path = Path(path)
    if not path.exists() or not CONFIG.get("resume", True):
        return 0, -1.0, {{}}
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
        md("## 10. Train and test"),
        cd(
            r"""
def mask_metrics_from_tensors(probs, targets, thresh=0.5):
    probs = probs.numpy() if torch.is_tensor(probs) else np.asarray(probs)
    targets = targets.numpy() if torch.is_tensor(targets) else np.asarray(targets)
    ious, dices = [], []
    for i in range(len(targets)):
        p = (probs[i].reshape(-1) >= thresh).astype(np.float32)
        t = (targets[i].reshape(-1) >= 0.5).astype(np.float32)
        inter = float((p * t).sum())
        union = float(((p + t) > 0).sum())
        ps, ts = float(p.sum()), float(t.sum())
        ious.append(1.0 if union == 0 else inter / union)
        dices.append(1.0 if (ps + ts) == 0 else (2 * inter) / (ps + ts))
    return ious, dices


@torch.no_grad()
def evaluate_loader(model, loader, split_name: str):
    model.eval()
    y_true, y_pred, y_prob, recs = [], [], [], []
    dice_all, iou_all = [], []
    use_seg = bool(getattr(model, "segmentation", False))
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        out = model(images)
        prob = out["fake_probability"].detach().cpu().numpy().reshape(-1)
        pred = (prob >= CONFIG["classification_threshold"]).astype(int)
        true = labels.detach().cpu().numpy().reshape(-1).astype(int)
        y_true.extend(true.tolist()); y_pred.extend(pred.tolist()); y_prob.extend(prob.tolist())
        mm = None
        if use_seg and "segmentation_prob" in out:
            mm = mask_metrics_from_tensors(out["segmentation_prob"].detach().cpu(), masks.detach().cpu(), CONFIG["mask_threshold"])
            iou_all.extend(mm[0]); dice_all.extend(mm[1])
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
                rec["iou"] = mm[0][i]
                rec["dice"] = mm[1][i]
            recs.append(rec)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if y_true else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if y_true else float("nan"),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if y_true else float("nan"),
        "n": int(len(y_true)),
    }
    if y_true:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
    if use_seg and dice_all:
        metrics["dice"] = float(np.mean(dice_all))
        metrics["miou"] = float(np.mean(iou_all))
        metrics["dice_note"] = "agreement with the generated mask, not a hand-labeled polygon"
    return metrics, recs


def train_one_experiment():
    model = build_model()
    use_seg = bool(getattr(model, "segmentation", False))
    criterion = VERITASLoss(CONFIG["lambda_cls"], CONFIG["lambda_seg"], CONFIG["dice_weight"], use_seg)
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
    best_metric = -1.0 if best_metric is None else float(best_metric)
    amp_ok = bool(CONFIG["use_amp"] and DEVICE.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_ok)
        autocast = lambda: torch.amp.autocast("cuda", enabled=amp_ok)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_ok)
        autocast = lambda: torch.cuda.amp.autocast(enabled=amp_ok)
    scaler_state = (resumed or {}).get("scaler_state_dict")
    if scaler_state and amp_ok:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as e:
            print("scaler not loaded:", e)
    per_run = CONFIG.get("epochs_per_run")
    stop_after = total_epochs if not per_run else min(total_epochs, start_epoch + int(per_run))
    print(f"START OF RUN | {EXP}: {start_epoch}/{total_epochs} epochs completed")
    if CONFIG["skip_training"]:
        if ckpt_best.exists():
            load_if_any(ckpt_best, model)
        return model, start_epoch, best_metric, history
    budget_s = float(CONFIG.get("time_budget_hours") or 11) * 3600.0
    elapsed = lambda: time.time() - NOTEBOOK_START_TIME
    accum = max(int(CONFIG["gradient_accumulation"]), 1)
    mid_s = float(CONFIG.get("mid_epoch_checkpoint_minutes") or 30) * 60.0
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
            if (time.time() - last_save) >= mid_s:
                save_checkpoint(ckpt_last, model, optimizer, scaler, epoch, best_metric, history, epoch_seconds, partial={"epoch": epoch, "step": step + 1})
                last_save = time.time()
                print(f"mid-epoch save epoch {epoch + 1} step {step + 1}/{len(TRAIN_LOADER)}")
            if elapsed() > budget_s:
                save_checkpoint(ckpt_last, model, optimizer, scaler, epoch, best_metric, history, epoch_seconds, partial={"epoch": epoch, "step": step + 1})
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
over_budget = (time.time() - NOTEBOOK_START_TIME) > float(CONFIG["time_budget_hours"]) * 3600.0
if over_budget:
    print("time budget already used; skipping Test this run")
elif EPOCHS_DONE >= int(CONFIG["epochs"]) or CONFIG.get("test_eval_when") == "always":
    ckpt_best = WORKING / "checkpoints" / f"{EXP}_best.pt"
    if ckpt_best.exists():
        load_if_any(ckpt_best, MODEL)
    metrics, recs = evaluate_loader(MODEL, TEST_LOADER, "test")
    print("TEST", metrics)
    pred_path = WORKING / "predictions" / f"{EXP}_predictions.csv"
    pd.DataFrame(recs).to_csv(pred_path, index=False)
    row = {"experiment": EXP, "status": "ok", "epochs_completed": EPOCHS_DONE, **{k: v for k, v in metrics.items() if not str(k).endswith("note")}}
    pd.DataFrame([row]).to_csv(WORKING / "experiment_results.csv", index=False)
    print("wrote", pred_path)
"""
        ),
        md(
            f"""
## Notes

- `{exp}` starts from ImageNet. It does not resume `full_veritas_defacto`.
- Classification labels are the Wild folder names. Real = 0, fake = 1.
- The mask PNG is the current Full model’s prediction on that crop. Dice measures agreement with that prediction.
- All four `OW_01`–`OW_04` notebooks must print the same `PROTOCOL_HASH` when `run_mode` matches.
- `OW_05_compare.ipynb` reads the `experiment_results.csv` files. It does not train.
"""
        ),
    ]
    return _nb(cells)


def build_compare_nb():
    cells = [
        md(
            """
# OpenForensics Wild + generated masks — compare

This notebook does not train and does not load images.

Attach the latest output of each `OW_01`–`OW_04` notebook that has finished a test cell.
Each output contains `experiment_results.csv`.

Classification numbers are Wild folder labels.
Dice and mIoU, when present, are agreement with the generated masks.
"""
        ),
        cd(
            r"""
import json
from pathlib import Path

import pandas as pd

EXPECTED = [
    "baseline_effb7_ow",
    "multistream_no_seg_ow",
    "seg_no_multistream_ow",
    "full_veritas_ow",
]
roots = [p for p in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd(), Path.cwd() / "kaggle_working") if p.exists()]
frames = []
for root in roots:
    for path in root.rglob("experiment_results.csv"):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print("skip", path, e)
            continue
        if "experiment" not in df.columns:
            continue
        hit = df[df.experiment.astype(str).isin(EXPECTED)].copy()
        if hit.empty:
            continue
        hit["source"] = str(path)
        frames.append(hit)
        print("read", path, hit.experiment.tolist())
if not frames:
    raise FileNotFoundError("No OW experiment_results.csv found. Attach the OW_01–OW_04 outputs after their test cells have run.")
all_df = pd.concat(frames, ignore_index=True)
all_df = all_df.drop_duplicates("experiment", keep="last")
cols = [c for c in ["experiment", "epochs_completed", "n", "accuracy", "f1", "precision", "recall", "tp", "tn", "fp", "fn", "dice", "miou"] if c in all_df.columns]
print(all_df[cols].sort_values("experiment").to_string(index=False))
missing = [e for e in EXPECTED if e not in set(all_df.experiment.astype(str))]
if missing:
    print("still missing:", missing)
out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working"
out.mkdir(parents=True, exist_ok=True)
all_df.to_csv(out / "ow_compare.csv", index=False)
print("wrote", out / "ow_compare.csv")
print("Dice and mIoU compare each segmentation model with the generated mask. They are not a hand-labeled localization score.")
"""
        ),
    ]
    return _nb(cells)


README = """# OpenForensics Wild with generated masks

This folder is separate from `notebooks-wild/` and `notebooks-of/`.

`OW_00` runs the current Full checkpoint on every OpenForensics Wild image and saves a mask.
`OW_01` to `OW_04` then train on those Wild images and those masks. No other image set is attached.

The mask is the Full model’s prediction on one face crop, threshold 0.5. It is not a hand-drawn polygon. Dice in the training notebooks is agreement with that prediction. The classification label is still the Wild `Real` or `Fake` folder.

## What to attach

| Notebook | Attach | Writes |
| --- | --- | --- |
| `OW_00_generate_masks.ipynb` | Wild dataset, `full_veritas_defacto_best.pt` or `.zip`, and this notebook’s previous output when resuming | `/kaggle/working/ow_masks/` |
| `OW_01` … `OW_04` | Wild dataset, the finished `OW_00` notebook (Kaggle output `_output_.zip` or a loose `ow_masks/` folder), and that notebook’s own checkpoint on later epochs | `checkpoints/{experiment}_best.pt` and `_last.pt` |
| `OW_05_compare.ipynb` | The `OW_01`–`OW_04` outputs | `ow_compare.csv` |

Wild dataset: `ahmedgamal/openforensics-wild-dataset`.

## How a mask run stays under 12 hours

The Kaggle session limit is 12 hours. `OW_00` stops at **10.5 hours**, writes `index.csv` and `progress.json`, and exits.

1. Save the version.
2. New session. Attach the Wild dataset, the same Full checkpoint, and this notebook’s latest output.
3. Leave `shard_index` and `shard_count` unchanged. Run All.
4. Stop when `progress.json` says `"resume": false`.

Finished images are skipped. The new output copies the earlier PNGs forward, so the latest version of a shard is a complete mask set for that shard.

`manifest.csv` is the full image list. `index.csv` is the finished rows for this shard. Each PNG lives at `masks/{split}/{real or fake}/{image_id}.png`. `image_id` is a hash of `train/fake/filename`, so the same file name in another split does not collide. The crop box in `index.csv` is the box the mask was painted on. Training reads that box and does not detect the face again.

After 64 new images the log prints images per hour. If that estimate needs more than one session, resume as above. To split the work across GPUs instead, set `shard_count` before the first full run and start one notebook per `shard_index`. Training attaches every shard. Do not change `shard_count` after the first `index.csv` exists.

`run_mode: smoke` paints 8 images. `full` walks the shard.

## How training reads it

`OW_01`–`OW_04` start from ImageNet, learning rate 1e-4, both loss weights 1.0, 3 epochs, one epoch per run. They do not load the Full checkpoint’s weights.

A training run stops itself at 11 hours and writes a mid-epoch checkpoint every 30 minutes. Attach that output to continue. The log must show the epoch number going up by one when a full epoch finishes.

`full` refuses to train if any Wild image is still missing a mask. Finish `OW_00` first.
"""


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    nbf.write(build_generate_nb(), DEST / "OW_00_generate_masks.ipynb")
    for v in VARIANTS:
        nbf.write(build_training_nb(v), DEST / v["file"])
    nbf.write(build_compare_nb(), DEST / "OW_05_compare.ipynb")
    (DEST / "README.md").write_text(README, encoding="utf-8")
    print("wrote", DEST)


if __name__ == "__main__":
    main()
