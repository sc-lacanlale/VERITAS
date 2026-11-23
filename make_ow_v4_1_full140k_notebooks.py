"""Generate matched full-140k Kaggle T4x2 DDP notebooks for:
  1. OWV41_140k_01_baseline_b7.ipynb  (Matched Baseline EfficientNet-B7 + Forensic Pipeline)
  2. OWV41_140k_02_veritas_v4_1.ipynb (VERITAS v4.1: Pruned SRM + Active Seg + Decoupled Dual-Expert Fusion)

Both notebooks share the exact same data pipeline, seed=42, forensic augmentations,
2-stage training schedule (1 warmup + 2 finetune), 8,000 stratified Val split,
full 10,905 Test evaluation, and 5-condition Robustness Stress Test.
Mask archives are unpacked strictly to /tmp/ow_masks_cache so /kaggle/working stays
clean and compact (~275 MB total output).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCH_V4_PATH = ROOT / "veritas-app" / "backend" / "inference" / "architecture_v4.py"
OUT_DIR = ROOT / "notebooks-ow-v4"


def build_standalone_arch_code() -> str:
    src = ARCH_V4_PATH.read_text(encoding="utf-8")
    replacement = '''import numpy as np

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ClassificationHead(nn.Module):
    def __init__(self, in_ch: int = 2560, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))
'''
    target = "from .architecture import ClassificationHead, IMAGENET_MEAN, IMAGENET_STD"
    assert target in src, "Expected import line not found in architecture_v4.py"
    return src.replace("from __future__ import annotations\n", "").replace(target, replacement)


TRAIN_SCRIPT_BODY = r'''from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

# === Embedded VERITAS v4 / v4.1 architecture ===
__ARCH_V4_CODE__

IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABELS = {"real": 0, "fake": 1}


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, dev


def cleanup_ddp(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tmp_cache_dir(working: Path) -> Path:
    if Path("/kaggle/working").exists() and Path("/tmp").exists():
        d = Path("/tmp/ow_masks_cache")
    else:
        d = working / "_tmp_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    aliases = {"train": ["Train", "train"], "val": ["Validation", "Val", "validation", "val"], "test": ["Test", "test"]}
    for a in aliases[name]:
        p = root / a
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No {name} split under {root}")


def _class_dir(split_dir: Path, name: str):
    for a in (name, name.capitalize(), name.upper(), name.lower()):
        p = split_dir / a
        if p.is_dir():
            return p
    return None


def list_images(wild_root: Path, split_name: str) -> pd.DataFrame:
    split_dir = _split_dir(wild_root, split_name)
    rows = []
    for cname, lab in LABELS.items():
        cdir = _class_dir(split_dir, cname)
        if cdir is None:
            continue
        for p in sorted(cdir.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTS:
                rows.append({
                    "path": str(p),
                    "split": split_name,
                    "label": lab,
                    "class_name": cname,
                    "file_name": p.name,
                    "rel_key": f"{split_name}/{cname}/{p.name}",
                })
    return pd.DataFrame(rows)


def unpack_mask_outputs(input_root: Path, cache_dir: Path) -> list:
    roots = []
    if not input_root.exists():
        return roots
    for archive in sorted(input_root.rglob("_output_.zip")):
        key = hashlib.sha256(str(archive).encode()).hexdigest()[:12]
        target = cache_dir / f"unpacked_masks_{key}"
        marker = target / ".done"
        if marker.exists():
            roots.append(target)
            continue
        t0 = time.time()
        with zipfile.ZipFile(archive) as z:
            members = [
                m for m in z.infolist()
                if not m.is_dir() and "ow_masks" in PurePosixPath(m.filename.replace("\\", "/")).parts
            ]
            if not members:
                continue
            print(f"[CACHE] Unpacking {len(members)} mask files to {target} ...", flush=True)
            target.mkdir(parents=True, exist_ok=True)
            for member in members:
                rel = PurePosixPath(member.filename.replace("\\", "/"))
                if rel.is_absolute() or ".." in rel.parts:
                    continue
                rel = Path(*rel.parts[rel.parts.index("ow_masks"):])
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out)
            marker.write_text("ok", encoding="utf-8")
            print(f"[CACHE] Unpacked masks to /tmp in {time.time() - t0:.1f}s", flush=True)
            roots.append(target)
    return roots


def load_mask_table(tmp_cache: Path, working: Path) -> pd.DataFrame:
    frames = []
    roots = [p for p in (Path("/kaggle/input"), tmp_cache, Path.cwd()) if p.exists()]
    roots.extend(unpack_mask_outputs(Path("/kaggle/input"), tmp_cache))
    seen = set()
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if "index.csv" not in filenames:
                continue
            path = Path(dirpath) / "index.csv"
            if str(path) in seen:
                continue
            seen.add(str(path))
            try:
                df = pd.read_csv(path, dtype=str)
            except Exception:
                continue
            if not {"image_id", "rel_key", "mask_file", "status", "crop_x0"} <= set(df.columns):
                continue
            df["_root"] = str(path.parent)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No ow_masks/index.csv found. Attach clarenceshanley/veritas-ow-00-generate-masks.")
    all_df = pd.concat(frames, ignore_index=True)
    kept, used = [], set()
    for rec in all_df.to_dict("records"):
        iid = str(rec["image_id"])
        if iid in used or str(rec.get("status")) != "ok":
            continue
        png = Path(str(rec["_root"])) / str(rec["mask_file"])
        if not png.is_file():
            continue
        rec["mask_path"] = str(png)
        kept.append(rec)
        used.add(iid)
    return pd.DataFrame(kept).drop_duplicates("rel_key", keep="last")


def build_index(cfg: dict, tmp_cache: Path, working: Path, rank: int) -> dict:
    cache_path = tmp_cache / "index_v41_140k.json"
    if rank == 0:
        wild_root = discover_wild_root()
        masks = load_mask_table(tmp_cache, working)
        index = {}
        limits = {"train": cfg["train_per_class"], "val": cfg["val_per_class"], "test": cfg["test_per_class"]}
        for split in ("train", "val", "test"):
            imgs = list_images(wild_root, split)
            merged = imgs.merge(
                masks[["rel_key", "mask_path", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "positive_frac"]],
                on="rel_key",
                how="left",
            )
            ready = merged.loc[merged["mask_path"].notna()].copy()
            lim = limits[split]
            if lim is not None:
                parts = []
                rng = np.random.RandomState(cfg["seed"] + {"train": 11, "val": 22, "test": 33}[split])
                for _, g in ready.groupby("label"):
                    n = min(int(lim), len(g))
                    parts.append(g.sample(n=n, random_state=rng) if n < len(g) else g)
                ready = pd.concat(parts)
            ready = ready.reset_index(drop=True)
            index[split] = ready.to_dict("records")
            print(
                f"[INDEX] {split}: {len(ready)} images "
                f"(real={(ready.label==0).sum()}, fake={(ready.label==1).sum()})",
                flush=True,
            )
        cache_path.write_text(json.dumps(index), encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
    loaded = json.loads(cache_path.read_text(encoding="utf-8"))
    return {k: pd.DataFrame(v) for k, v in loaded.items()}


def letterbox_rgb(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_rgb.shape[:2]
    scale = min(size / max(h, 1), size / max(w, 1))
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    valid = np.zeros((size, size), dtype=np.float32)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top : top + nh, left : left + nw] = resized
    valid[top : top + nh, left : left + nw] = 1.0
    return canvas, valid


def apply_forensic_augmentations(crop: np.ndarray, rng: random.Random, forensic_aug: bool) -> np.ndarray:
    if rng.random() < 0.45:
        alpha = 1.0 + rng.uniform(-0.15, 0.15)
        beta = rng.uniform(-12.0, 12.0)
        crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if not forensic_aug:
        return crop
    if rng.random() < 0.25:
        h, w = crop.shape[:2]
        sc = rng.uniform(0.65, 0.95)
        small = cv2.resize(crop, (max(16, int(w * sc)), max(16, int(h * sc))), interpolation=cv2.INTER_AREA)
        crop = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.20:
        k = rng.choice([3, 5])
        crop = cv2.GaussianBlur(crop, (k, k), 0)
    if rng.random() < 0.45:
        q = rng.randint(45, 95)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                crop = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return crop


def apply_corruption(crop: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return crop
    if condition == "jpeg_q50":
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return crop
    if condition == "jpeg_q30":
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return crop
    if condition == "blur_sigma2":
        return cv2.GaussianBlur(crop, (0, 0), sigmaX=2.0, sigmaY=2.0)
    if condition == "resize_0.5x":
        h, w = crop.shape[:2]
        small = cv2.resize(crop, (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"Unknown condition: {condition}")


class OpenForensicsMaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, augment: bool, forensic_aug: bool, seed: int, condition: str = "clean"):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.augment = bool(augment)
        self.forensic_aug = bool(forensic_aug)
        self.seed = int(seed)
        self.condition = str(condition)
        self.epoch = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + int(idx))
        bgr = cv2.imread(row.path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Unreadable image: {row.path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        x0, y0 = max(0, int(float(row.crop_x0))), max(0, int(float(row.crop_y0)))
        x1, y1 = min(w, max(x0 + 1, int(float(row.crop_x1)))), min(h, max(y0 + 1, int(float(row.crop_y1))))
        crop = rgb[y0:y1, x0:x1]

        mask = cv2.imread(str(row.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unreadable mask: {row.mask_path}")
        if mask.shape != (self.size, self.size):
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            crop = apply_forensic_augmentations(crop, rng, self.forensic_aug)
        elif self.condition != "clean":
            crop = apply_corruption(crop, self.condition)

        canvas, valid = letterbox_rgb(crop, self.size)
        if self.augment and rng.random() < 0.5:
            canvas = np.ascontiguousarray(canvas[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            valid = np.ascontiguousarray(valid[:, ::-1])

        x = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
        x = np.transpose(x, (2, 0, 1))
        target = (mask.astype(np.float32) > 127.0).astype(np.float32)
        return {
            "image": torch.from_numpy(x.copy()),
            "mask": torch.from_numpy(target[None].copy()),
            "valid_mask": torch.from_numpy(valid[None].copy()),
            "label": torch.tensor([float(row.label)], dtype=torch.float32),
            "rel_key": str(row.rel_key),
            "file_name": str(row.file_name),
        }


class VERITASLossV41(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_rgb=0.50, lambda_aux=0.50, lambda_seg=0.50, dice_weight=1.0, variant="veritas_v4_1"):
        super().__init__()
        self.lambda_cls = float(lambda_cls)
        self.lambda_rgb = float(lambda_rgb)
        self.lambda_aux = float(lambda_aux)
        self.lambda_seg = float(lambda_seg)
        self.dice_weight = float(dice_weight)
        self.variant = str(variant)

    def forward(self, out: dict, labels: torch.Tensor, masks: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        labels_f = labels.float()
        cls_loss = F.binary_cross_entropy_with_logits(out["classification_logits"].float(), labels_f)
        rgb_loss = cls_loss.new_zeros(())
        aux_loss = cls_loss.new_zeros(())
        seg_loss = cls_loss.new_zeros(())

        if self.variant != "baseline":
            if "rgb_logits" in out:
                rgb_loss = F.binary_cross_entropy_with_logits(out["rgb_logits"].float(), labels_f)
            aux_loss = F.binary_cross_entropy_with_logits(out["aux_forensic_logits"].float(), labels_f)
            seg_logits = out["segmentation_logits"].float()
            targets = masks.float()
            real = labels_f[:, 0] == 0
            targets = torch.where(real[:, None, None, None], torch.zeros_like(targets), targets)
            valid = valid_mask.float()
            dims = (1, 2, 3)
            px = valid.sum(dims).clamp_min(1.0)
            nonempty = (targets * valid).sum(dims) > 0
            trusted = real | nonempty
            positive = (~real) & nonempty

            per_bce = (F.binary_cross_entropy_with_logits(seg_logits, targets, reduction="none") * valid).sum(dims) / px
            probs = seg_logits.sigmoid() * valid
            fg = targets * valid
            per_dice = 1.0 - (2.0 * (probs * fg).sum(dims) + 1e-6) / (probs.sum(dims) + fg.sum(dims) + 1e-6)
            per_img = torch.where(positive, (per_bce + self.dice_weight * per_dice) / (1.0 + self.dice_weight), per_bce)
            tw = trusted.float()
            seg_loss = (per_img * tw).sum() / tw.sum().clamp_min(1.0)

        total = (
            self.lambda_cls * cls_loss
            + self.lambda_rgb * rgb_loss
            + self.lambda_aux * aux_loss
            + self.lambda_seg * seg_loss
        )
        return {
            "total": total,
            "cls": cls_loss.detach(),
            "rgb": rgb_loss.detach(),
            "aux": aux_loss.detach(),
            "seg": seg_loss.detach(),
        }


def freeze_backbone_bn(model: nn.Module):
    enc = model.module.encoder if isinstance(model, DDP) else model.encoder
    for m in enc.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, backbone_mult: float):
    raw = model.module if isinstance(model, DDP) else model
    bb = [p for p in raw.encoder.parameters() if p.requires_grad]
    bb_ids = {id(p) for p in bb}
    heads = [p for p in raw.parameters() if p.requires_grad and id(p) not in bb_ids]
    groups = []
    if bb:
        groups.append({"params": bb, "lr": lr * backbone_mult})
    if heads:
        groups.append({"params": heads, "lr": lr})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, is_ddp: bool, variant: str, use_amp: bool):
    model.eval()
    is_veritas = variant != "baseline"
    has_rgb_head = variant == "veritas_v4_1"
    local_recs = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        vm = batch["valid_mask"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            out = model(x, vm)
        probs = out["fake_probability"].float().squeeze(1).cpu().numpy()
        y = labels.squeeze(1).cpu().numpy().astype(int)
        rgb_p = out["rgb_logits"].float().sigmoid().squeeze(1).cpu().numpy() if has_rgb_head else None
        aux_p = out["aux_forensic_logits"].float().sigmoid().squeeze(1).cpu().numpy() if is_veritas else None
        gates = out["stream_gates"].float().cpu().numpy() if is_veritas else None

        real_pos_frac = [None] * len(y)
        fake_dice = [None] * len(y)
        fake_iou = [None] * len(y)
        if is_veritas:
            pred_m = (out["segmentation_prob"].float() >= 0.5).float() * vm
            gt_m = masks.float() * vm
            v_sum = vm.sum(dim=(1, 2, 3)).clamp_min(1.0)
            p_sum = pred_m.sum(dim=(1, 2, 3))
            g_sum = gt_m.sum(dim=(1, 2, 3))
            inter = (pred_m * gt_m).sum(dim=(1, 2, 3))
            union = (pred_m + gt_m - pred_m * gt_m).sum(dim=(1, 2, 3))
            rpf = (p_sum / v_sum).cpu().numpy()
            dice_arr = ((2.0 * inter + 1e-6) / (p_sum + g_sum + 1e-6)).cpu().numpy()
            iou_arr = ((inter + 1e-6) / (union + 1e-6)).cpu().numpy()
            for i in range(len(y)):
                if y[i] == 0:
                    real_pos_frac[i] = float(rpf[i])
                else:
                    fake_dice[i] = float(dice_arr[i])
                    fake_iou[i] = float(iou_arr[i])

        for i in range(len(y)):
            rec = {
                "rel_key": batch["rel_key"][i],
                "file_name": batch["file_name"][i],
                "ground_truth_label": int(y[i]),
                "fake_probability": float(probs[i]),
                "predicted_label": int(probs[i] >= 0.5),
            }
            if is_veritas:
                if has_rgb_head:
                    rec["rgb_probability"] = float(rgb_p[i])
                rec["aux_forensic_probability"] = float(aux_p[i])
                rec["real_positive_fraction"] = real_pos_frac[i]
                rec["dice_teacher_fake"] = fake_dice[i]
                rec["iou_teacher_fake"] = fake_iou[i]
                rec["gate_rgb"] = float(gates[i, 0])
                rec["gate_srm"] = float(gates[i, 1])
            local_recs.append(rec)

    if is_ddp:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_recs)
        merged = [r for sub in gathered for r in sub]
    else:
        merged = local_recs

    dedup = {r["rel_key"]: r for r in merged}
    recs = [dedup[k] for k in sorted(dedup.keys())]

    y_true = np.array([r["ground_truth_label"] for r in recs], dtype=int)
    p = np.array([r["fake_probability"] for r in recs], dtype=float)
    y_pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "n": int(len(recs)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    if is_veritas:
        aux_p = np.array([r["aux_forensic_probability"] for r in recs], dtype=float)
        metrics["aux_forensic_accuracy"] = float(accuracy_score(y_true, (aux_p >= 0.5).astype(int)))
        metrics["aux_forensic_roc_auc"] = float(roc_auc_score(y_true, aux_p))
        if has_rgb_head:
            rgb_p = np.array([r["rgb_probability"] for r in recs], dtype=float)
            metrics["rgb_accuracy"] = float(accuracy_score(y_true, (rgb_p >= 0.5).astype(int)))
            metrics["rgb_roc_auc"] = float(roc_auc_score(y_true, rgb_p))
            eps = 1e-6
            lr = np.log(np.clip(rgb_p, eps, 1 - eps) / (1 - np.clip(rgb_p, eps, 1 - eps)))
            la = np.log(np.clip(aux_p, eps, 1 - eps) / (1 - np.clip(aux_p, eps, 1 - eps)))
            metrics["logit_corr_rgb_aux"] = float(np.corrcoef(lr, la)[0, 1]) if len(lr) > 2 else float("nan")
        rpf_vals = [r["real_positive_fraction"] for r in recs if r["real_positive_fraction"] is not None]
        d_vals = [r["dice_teacher_fake"] for r in recs if r["dice_teacher_fake"] is not None]
        i_vals = [r["iou_teacher_fake"] for r in recs if r["iou_teacher_fake"] is not None]
        metrics["real_mean_positive_fraction"] = float(np.mean(rpf_vals)) if rpf_vals else float("nan")
        metrics["real_mask_nonempty_rate"] = float(np.mean([v > 0 for v in rpf_vals])) if rpf_vals else float("nan")
        metrics["dice_teacher_fake_only"] = float(np.mean(d_vals)) if d_vals else float("nan")
        metrics["miou_teacher_fake_only"] = float(np.mean(i_vals)) if i_vals else float("nan")
        metrics["mean_gate_rgb"] = float(np.mean([r["gate_rgb"] for r in recs]))
        metrics["mean_gate_srm"] = float(np.mean([r["gate_srm"] for r in recs]))
    return metrics, recs


@torch.no_grad()
def benchmark_inference(model: nn.Module, device: torch.device, batch_size: int = 4, size: int = 600):
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()
    x = torch.randn(batch_size, 3, size, size, device=device)
    vm = torch.ones(batch_size, 1, size, size, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        _ = raw(x, vm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    repeats = 5
    for _ in range(repeats):
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            _ = raw(x, vm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    batch_ms = ((time.perf_counter() - t0) / repeats) * 1000.0
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        f"per_image_ms_bs{batch_size}": batch_ms / batch_size,
        f"batch_latency_ms_bs{batch_size}": batch_ms,
        f"peak_vram_mib_bs{batch_size}": peak_bytes / (1024 * 1024),
    }


def train_one_stage(
    model: nn.Module,
    loader: DataLoader,
    sampler,
    criterion: VERITASLossV41,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    epoch: int,
    grad_accum: int,
    use_amp: bool,
    rank: int,
    stage_name: str,
):
    model.train()
    freeze_backbone_bn(model)
    loader.dataset.epoch = epoch
    if sampler is not None:
        sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    n_steps = len(loader)
    tot_loss = tot_cls = tot_rgb = tot_aux = tot_seg = 0.0
    log_every = max(1, n_steps // 10)
    for step, batch in enumerate(loader, start=1):
        x = batch["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        vm = batch["valid_mask"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            out = model(x, vm)
            losses = criterion(out, labels, masks, vm)
            scaled_loss = losses["total"] / grad_accum
        scaler.scale(scaled_loss).backward()
        if step % grad_accum == 0 or step == n_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        tot_loss += float(losses["total"].detach())
        tot_cls += float(losses["cls"])
        tot_rgb += float(losses["rgb"])
        tot_aux += float(losses["aux"])
        tot_seg += float(losses["seg"])
        if rank == 0 and (step % log_every == 0 or step == n_steps):
            dt = time.time() - t0
            eta = (dt / step) * (n_steps - step)
            print(
                f"  [{stage_name} ep {epoch}] step {step}/{n_steps} "
                f"loss={tot_loss/step:.4f} cls={tot_cls/step:.4f} rgb={tot_rgb/step:.4f} "
                f"aux={tot_aux/step:.4f} seg={tot_seg/step:.4f} ({dt:.0f}s, eta {eta/60:.1f}m)",
                flush=True,
            )
    return time.time() - t0


def run_single_variant(variant: str, cfg: dict, index: dict, working: Path, is_ddp: bool, rank: int, world_size: int, local_rank: int, device: torch.device):
    set_seed(cfg["seed"])
    is_veritas = variant != "baseline"
    t_start = time.time()
    if rank == 0:
        print(f"\n==================== RUNNING FULL-140K {variant.upper()} ====================", flush=True)

    train_ds = OpenForensicsMaskDataset(index["train"], cfg["image_size"], augment=True, forensic_aug=cfg["use_forensic_aug"], seed=cfg["seed"])
    val_ds = OpenForensicsMaskDataset(index["val"], cfg["image_size"], augment=False, forensic_aug=False, seed=cfg["seed"])
    test_ds = OpenForensicsMaskDataset(index["test"], cfg["image_size"], augment=False, forensic_aug=False, seed=cfg["seed"])

    tr_samp = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg["seed"], drop_last=True) if is_ddp else None
    va_samp = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None
    te_samp = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None

    nw = cfg["num_workers"]
    eval_bs = cfg.get("eval_batch_size_per_gpu", cfg["batch_size_per_gpu"])
    kw = dict(num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=tr_samp, shuffle=(tr_samp is None), drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, sampler=va_samp, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, sampler=te_samp, shuffle=False, **kw)

    grad_accum = max(1, cfg["target_effective_batch"] // (cfg["batch_size_per_gpu"] * world_size))

    if is_ddp:
        if rank == 0:
            _ = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        dist.barrier()
    raw_model = VERITASV4(variant=variant, pretrained=True).to(device).to(memory_format=torch.channels_last)
    criterion = VERITASLossV41(
        lambda_cls=cfg["lambda_cls"],
        lambda_rgb=cfg.get("lambda_rgb", 0.50),
        lambda_aux=cfg["lambda_aux"],
        lambda_seg=cfg["lambda_seg"],
        dice_weight=cfg["dice_weight"],
        variant=variant,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["use_amp"] and device.type == "cuda")) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=(cfg["use_amp"] and device.type == "cuda"))

    best_val_acc = -1.0
    best_path = working / "checkpoints" / f"{variant}_140k_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    pred_dir = working / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    history = []

    def log_val(stage_label: str, ep_num: int, sec: float, val_m: dict, val_recs: list):
        nonlocal best_val_acc
        extra = ""
        if is_veritas:
            extra = (
                f" rgb={val_m.get('rgb_accuracy', 0):.4f} aux={val_m['aux_forensic_accuracy']:.4f} "
                f"r={val_m.get('logit_corr_rgb_aux', 0):.3f} alpha={val_m['mean_gate_rgb']:.3f} "
                f"dice={val_m['dice_teacher_fake_only']:.4f}"
            )
        print(
            f"  [{stage_label} ep {ep_num}] VAL acc={val_m['accuracy']:.4f} f1={val_m['f1']:.4f} "
            f"auc={val_m['roc_auc']:.4f}{extra} ({sec/60:.1f}m)",
            flush=True,
        )
        history.append({"stage": stage_label, "epoch": ep_num, "seconds": sec, **val_m})
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            torch.save(raw_model.state_dict(), best_path)
            pd.DataFrame(val_recs).to_csv(pred_dir / f"{variant}_140k_val_predictions.csv", index=False)
            (working / f"{variant}_140k_progress.json").write_text(
                json.dumps({"best_val_acc": best_val_acc, "history": history}, indent=2), encoding="utf-8"
            )

    # Stage 1: Frozen-backbone head warmup
    if cfg["warmup_epochs"] > 0:
        raw_model.freeze_backbone(True)
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False) if is_ddp else raw_model
        opt = build_optimizer(model, lr=cfg["warmup_lr"], weight_decay=cfg["weight_decay"], backbone_mult=1.0)
        updates = max(1, math.ceil(len(train_loader) / grad_accum) * cfg["warmup_epochs"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=cfg["warmup_lr"] * 0.1)
        for ep in range(1, cfg["warmup_epochs"] + 1):
            sec = train_one_stage(model, train_loader, tr_samp, criterion, opt, sched, scaler, device, ep, grad_accum, cfg["use_amp"], rank, "Stage1-Warmup")
            val_m, val_recs = evaluate(raw_model, val_loader, device, is_ddp, variant, cfg["use_amp"])
            if rank == 0:
                log_val("warmup", ep, sec, val_m, val_recs)
        del model, opt, sched
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stage 2: Full end-to-end fine-tuning
    if cfg["finetune_epochs"] > 0:
        raw_model.freeze_backbone(False)
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False) if is_ddp else raw_model
        opt = build_optimizer(model, lr=cfg["finetune_lr"], weight_decay=cfg["weight_decay"], backbone_mult=cfg["backbone_lr_multiplier"])
        updates = max(1, math.ceil(len(train_loader) / grad_accum) * cfg["finetune_epochs"])
        warmup_u = max(1, int(round(updates * 0.05)))
        def lr_fn(u):
            if u < warmup_u:
                return (u + 1) / warmup_u
            prog = min(1.0, (u - warmup_u) / max(1, updates - warmup_u))
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
        for ep in range(1, cfg["finetune_epochs"] + 1):
            sec = train_one_stage(model, train_loader, tr_samp, criterion, opt, sched, scaler, device, cfg["warmup_epochs"] + ep, grad_accum, cfg["use_amp"], rank, "Stage2-Finetune")
            val_m, val_recs = evaluate(raw_model, val_loader, device, is_ddp, variant, cfg["use_amp"])
            if rank == 0:
                log_val("finetune", ep, sec, val_m, val_recs)
        del model, opt, sched
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_ddp:
        dist.barrier()

    # Clean evaluation on ALL 10,905 OpenForensics Test images
    raw_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_m, test_recs = evaluate(raw_model, test_loader, device, is_ddp, variant, cfg["use_amp"])
    train_eval_sec = time.time() - t_start

    if rank == 0:
        bench1 = benchmark_inference(raw_model, device, batch_size=1, size=cfg["image_size"])
        bench4 = benchmark_inference(raw_model, device, batch_size=4, size=cfg["image_size"])
        test_m.update(bench1)
        test_m.update(bench4)
        test_m["total_train_eval_seconds"] = train_eval_sec
        test_m["variant"] = variant
        if variant == "veritas_v4_1":
            test_m["learned_alpha"] = float((0.35 + 0.30 * torch.sigmoid(raw_model.fusion_head.raw_alpha)).item())
            test_m["learned_res_scale"] = float((0.25 * torch.tanh(raw_model.fusion_head.raw_res_scale)).item())
        pd.DataFrame(test_recs).to_csv(pred_dir / f"{variant}_140k_test_predictions.csv", index=False)
        (working / f"{variant}_140k_summary.json").write_text(
            json.dumps({"metrics": test_m, "history": history, "config": cfg}, indent=2), encoding="utf-8"
        )
        print(f"=== FINAL CLEAN TEST ({variant}) ===\n" + json.dumps(test_m, indent=2), flush=True)

    del train_loader, val_loader, test_loader
    gc.collect()

    # Built-in 5-Condition Robustness Stress Test on ALL 10,905 Test images (guarded by wall-clock budget)
    robustness = {"clean": test_m}
    budget_sec = float(cfg.get("robustness_time_budget_sec", 37800))
    eval_kw = dict(num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=False)
    for cond in cfg.get("robustness_conditions", ["jpeg_q50", "jpeg_q30", "blur_sigma2", "resize_0.5x"]):
        elapsed = time.time() - t_start
        # Broadcast budget decision from rank 0 so all DDP ranks stay synchronized
        go = torch.tensor([1 if elapsed < budget_sec else 0], device=device)
        if is_ddp:
            dist.broadcast(go, src=0)
        if int(go.item()) == 0:
            if rank == 0:
                print(f"[ROBUSTNESS] Skipping {cond}: elapsed {elapsed/3600:.2f}h >= budget {budget_sec/3600:.2f}h", flush=True)
            break
        tc = time.time()
        cond_ds = OpenForensicsMaskDataset(index["test"], cfg["image_size"], augment=False, forensic_aug=False, seed=cfg["seed"], condition=cond)
        cond_samp = DistributedSampler(cond_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None
        cond_loader = DataLoader(cond_ds, batch_size=eval_bs, sampler=cond_samp, shuffle=False, **eval_kw)
        cm, crecs = evaluate(raw_model, cond_loader, device, is_ddp, variant, cfg["use_amp"])
        if rank == 0:
            cm["eval_seconds"] = time.time() - tc
            robustness[cond] = cm
            pd.DataFrame(crecs).to_csv(pred_dir / f"{variant}_140k_robust_{cond}.csv", index=False)
            (working / f"{variant}_140k_robustness_summary.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")
            print(
                f"  [ROBUSTNESS {cond:12s}] Acc={cm['accuracy']*100:.2f}% F1={cm['f1']:.4f} "
                f"AUC={cm['roc_auc']:.4f} ({cm['eval_seconds']:.0f}s)",
                flush=True,
            )

    del raw_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    working = Path(cfg["working_dir"])
    working.mkdir(parents=True, exist_ok=True)
    tmp_cache = get_tmp_cache_dir(working)

    is_ddp, rank, world_size, local_rank, device = setup_ddp()
    try:
        if rank == 0:
            print(
                f"DDP={is_ddp} world_size={world_size} device={device} "
                f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
                f"tmp_cache={tmp_cache}",
                flush=True,
            )
        index = build_index(cfg, tmp_cache, working, rank)
        run_single_variant(cfg["variant"], cfg, index, working, is_ddp, rank, world_size, local_rank, device)
    finally:
        cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
'''


def make_cell(cell_type: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    cell = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook(variant: str, title: str, subtitle: str, out_name: str) -> Path:
    arch_code = build_standalone_arch_code()
    train_script = TRAIN_SCRIPT_BODY.replace("__ARCH_V4_CODE__", arch_code)
    work_subdir = f"ow_v41_140k_{variant}"

    config_code = f'''import json
from pathlib import Path
import torch

WORKING = Path("/kaggle/working/{work_subdir}") if Path("/kaggle/working").exists() else Path.cwd() / "{work_subdir}"
WORKING.mkdir(parents=True, exist_ok=True)

CONFIG = {{
    "variant": {variant!r},
    "working_dir": str(WORKING),
    "seed": 42,
    "image_size": 600,
    "train_per_class": None,   # FULL 140,000 OpenForensics Train images
    "val_per_class": 4000,     # 8,000 stratified Val images (fast epoch validation)
    "test_per_class": None,    # ALL 10,905 OpenForensics Test images
    "warmup_epochs": 1,        # Stage 1: frozen EfficientNet-B7, train heads/forensic/seg
    "finetune_epochs": 2,      # Stage 2: full end-to-end fine-tuning
    "batch_size_per_gpu": 4,   # 4 x 2 GPUs x 2 grad_accum = 16 effective batch
    "eval_batch_size_per_gpu": 8,
    "target_effective_batch": 16,
    "num_workers": 2,
    "use_amp": True,
    "use_forensic_aug": True,  # Identical forensic pipeline for both models
    "warmup_lr": 3e-4,
    "finetune_lr": 1.5e-4,
    "backbone_lr_multiplier": 0.1,
    "weight_decay": 1e-4,
    "lambda_cls": 1.0,
    "lambda_rgb": 0.50,
    "lambda_aux": 0.50,
    "lambda_seg": 0.50,
    "dice_weight": 1.0,
    "robustness_conditions": ["jpeg_q50", "jpeg_q30", "blur_sigma2", "resize_0.5x"],
    "robustness_time_budget_sec": 37800,  # 10.5 hrs safety guard under Kaggle 12h limit
}}

cfg_path = WORKING / "config.json"
cfg_path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Variant={{CONFIG['variant']}} | GPUs={{n_gpus}}", [torch.cuda.get_device_name(i) for i in range(n_gpus)])
print("Saved config to:", cfg_path)
'''

    summary_code = f'''import json
import pandas as pd

summary = json.loads((WORKING / "{variant}_140k_summary.json").read_text())
print("=== CLEAN TEST SUMMARY ({variant}) ===")
print(json.dumps(summary["metrics"], indent=2))

rob_path = WORKING / "{variant}_140k_robustness_summary.json"
if rob_path.exists():
    rob = json.loads(rob_path.read_text())
    rows = []
    for cond, m in rob.items():
        rows.append({{
            "condition": cond,
            "acc_pct": round(m["accuracy"] * 100, 2),
            "f1": round(m["f1"], 4),
            "prec": round(m["precision"], 4),
            "rec": round(m["recall"], 4),
            "auc": round(m["roc_auc"], 4),
            "fp": m["fp"],
            "fn": m["fn"],
        }})
    print("\\n=== 5-CONDITION ROBUSTNESS TABLE (10,905 TEST IMAGES) ===")
    print(pd.DataFrame(rows).to_string(index=False))
'''

    cells = [
        make_cell("markdown", f"# {title}\n\n{subtitle}\n\n- **Accelerator:** `GPU T4 x2` DDP\n- **Data:** `ahmedgamal/openforensics-wild-dataset` + `clarenceshanley/veritas-ow-00-generate-masks`\n- **Clean output:** Unpacks masks strictly to `/tmp/ow_masks_cache` so `/kaggle/working/` stays clean (~275 MB).\n"),
        make_cell("code", config_code),
        make_cell(
            "code",
            "# Write standalone DDP-safe full-140k training script\n"
            f"SCRIPT_CODE = {train_script!r}\n"
            "script_path = WORKING / 'train_v41_140k.py'\n"
            "script_path.write_text(SCRIPT_CODE, encoding='utf-8')\n"
            "print('Wrote training script:', script_path, 'bytes:', len(SCRIPT_CODE))\n",
        ),
        make_cell(
            "code",
            """import subprocess
import sys

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
if n_gpus >= 2:
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={n_gpus}", str(script_path), "--config", str(cfg_path)]
else:
    cmd = [sys.executable, str(script_path), "--config", str(cfg_path)]

print("Launching:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
if rc != 0:
    raise RuntimeError(f"Training script failed with exit code {rc}")
print("Full-140k training, clean test, and robustness evaluation finished cleanly!")
""",
        ),
        make_cell("code", summary_code),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    out_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def main():
    build_notebook(
        variant="baseline",
        title="VERITAS OW v4.1 (140k Full) — Matched Baseline EfficientNet-B7",
        subtitle="Full 140,000-image OpenForensics Train run of EfficientNet-B7 with the identical forensic augmentation pipeline and 2-stage schedule used by VERITAS v4.1, evaluated on all 10,905 Test images across 5 robustness conditions.",
        out_name="OWV41_140k_01_baseline_b7.ipynb",
    )
    build_notebook(
        variant="veritas_v4_1",
        title="VERITAS OW v4.1 (140k Full) — Proposed Decoupled Multi-Stream & Active Segmentation Model",
        subtitle="Full 140,000-image OpenForensics Train run of VERITAS v4.1 (pruned dead Freq/MaskContrast, retained 9-ch color SRM + SpatialForensicEncoderV4 + Multi-Scale ActiveSegmentationHeadV4 + DecoupledForensicFusionV41 with bounded dual-expert logit fusion), evaluated on all 10,905 Test images across 5 robustness conditions.",
        out_name="OWV41_140k_02_veritas_v4_1.ipynb",
    )


if __name__ == "__main__":
    main()
