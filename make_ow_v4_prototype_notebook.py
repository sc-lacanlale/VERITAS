"""Generate self-contained Kaggle T4x2 DDP head-to-head prototype notebook for VERITAS OW v4."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCH_V4_PATH = ROOT / "veritas-app" / "backend" / "inference" / "architecture_v4.py"
OUT_DIR = ROOT / "notebooks-ow-v4"
OUT_NB = OUT_DIR / "OWV4_prototype_head_to_head.ipynb"


def build_standalone_arch_code() -> str:
    src = ARCH_V4_PATH.read_text(encoding="utf-8")
    # Replace relative import from .architecture with self-contained helpers
    replacement = '''import numpy as np

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
    ks = [k / (np.abs(k).sum() or 1.0) for k in (k1, k2, k3)]
    return torch.from_numpy(np.stack(ks, 0)[:, None, :, :])


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
import tempfile
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

# === Embedded VERITASV4 architecture ===
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
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if not m.is_dir() and "ow_masks" in PurePosixPath(m.filename.replace("\\", "/")).parts]
            if not members:
                continue
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
            roots.append(target)
    return roots


def load_mask_table(working: Path) -> pd.DataFrame:
    frames = []
    roots = [p for p in (Path("/kaggle/input"), Path("/kaggle/working"), working, Path.cwd()) if p.exists()]
    roots.extend(unpack_mask_outputs(Path("/kaggle/input"), working / "cache"))
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


def build_index(cfg: dict, working: Path, rank: int) -> dict:
    cache_path = working / "cache" / "index_v4proto.json"
    if rank == 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        wild_root = discover_wild_root()
        masks = load_mask_table(working)
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
    # Mild brightness/contrast (same as OW v3)
    if rng.random() < 0.45:
        alpha = 1.0 + rng.uniform(-0.15, 0.15)
        beta = rng.uniform(-12.0, 12.0)
        crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if not forensic_aug:
        return crop
    # Random downscale-upscale resize jitter (simulates social-media/messaging rescaling)
    if rng.random() < 0.25:
        h, w = crop.shape[:2]
        sc = rng.uniform(0.65, 0.95)
        small = cv2.resize(crop, (max(16, int(w * sc)), max(16, int(h * sc))), interpolation=cv2.INTER_AREA)
        crop = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # Mild Gaussian blur
    if rng.random() < 0.20:
        k = rng.choice([3, 5])
        crop = cv2.GaussianBlur(crop, (k, k), 0)
    # Random JPEG recompression (critical for preventing RGB/SRM/FFT overfitting to train compression)
    if rng.random() < 0.45:
        q = rng.randint(45, 95)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                crop = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return crop


class ProtoMaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, augment: bool, forensic_aug: bool, seed: int):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.augment = bool(augment)
        self.forensic_aug = bool(forensic_aug)
        self.seed = int(seed)
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


class VERITASLossV4(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_aux=0.30, lambda_seg=0.50, dice_weight=1.0, is_v4=True):
        super().__init__()
        self.lambda_cls = float(lambda_cls)
        self.lambda_aux = float(lambda_aux)
        self.lambda_seg = float(lambda_seg)
        self.dice_weight = float(dice_weight)
        self.is_v4 = bool(is_v4)

    def forward(self, out: dict, labels: torch.Tensor, masks: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        labels_f = labels.float()
        cls_loss = F.binary_cross_entropy_with_logits(out["classification_logits"].float(), labels_f)
        aux_loss = cls_loss.new_zeros(())
        seg_loss = cls_loss.new_zeros(())

        if self.is_v4:
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

        total = self.lambda_cls * cls_loss + self.lambda_aux * aux_loss + self.lambda_seg * seg_loss
        return {
            "total": total,
            "cls": cls_loss.detach(),
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, is_ddp: bool, is_v4: bool, use_amp: bool):
    model.eval()
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
        aux_p = out["aux_forensic_logits"].float().sigmoid().squeeze(1).cpu().numpy() if is_v4 else None
        gates = out["stream_gates"].float().cpu().numpy() if is_v4 else None

        real_pos_frac = [None] * len(y)
        fake_dice = [None] * len(y)
        fake_iou = [None] * len(y)
        if is_v4:
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
            if is_v4:
                rec["aux_forensic_probability"] = float(aux_p[i])
                rec["real_positive_fraction"] = real_pos_frac[i]
                rec["dice_teacher_fake"] = fake_dice[i]
                rec["iou_teacher_fake"] = fake_iou[i]
                rec["gate_rgb"] = float(gates[i, 0])
                rec["gate_srm"] = float(gates[i, 1])
                rec["gate_freq"] = float(gates[i, 2])
                rec["gate_contrast"] = float(gates[i, 3])
            local_recs.append(rec)

    if is_ddp:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_recs)
        merged = [r for sub in gathered for r in sub]
    else:
        merged = local_recs

    # Deduplicate DistributedSampler padding cleanly by rel_key
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
    if is_v4:
        aux_p = np.array([r["aux_forensic_probability"] for r in recs], dtype=float)
        metrics["aux_forensic_accuracy"] = float(accuracy_score(y_true, (aux_p >= 0.5).astype(int)))
        metrics["aux_forensic_roc_auc"] = float(roc_auc_score(y_true, aux_p))
        rpf_vals = [r["real_positive_fraction"] for r in recs if r["real_positive_fraction"] is not None]
        d_vals = [r["dice_teacher_fake"] for r in recs if r["dice_teacher_fake"] is not None]
        i_vals = [r["iou_teacher_fake"] for r in recs if r["iou_teacher_fake"] is not None]
        metrics["real_mean_positive_fraction"] = float(np.mean(rpf_vals)) if rpf_vals else float("nan")
        metrics["real_mask_nonempty_rate"] = float(np.mean([v > 0 for v in rpf_vals])) if rpf_vals else float("nan")
        metrics["dice_teacher_fake_only"] = float(np.mean(d_vals)) if d_vals else float("nan")
        metrics["miou_teacher_fake_only"] = float(np.mean(i_vals)) if i_vals else float("nan")
        for g in ("gate_rgb", "gate_srm", "gate_freq", "gate_contrast"):
            metrics[f"mean_{g}"] = float(np.mean([r[g] for r in recs]))
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
    _ = raw(x, vm)  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    repeats = 5
    for _ in range(repeats):
        _ = raw(x, vm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    batch_ms = ((time.perf_counter() - t0) / repeats) * 1000.0
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "batch_size": batch_size,
        "per_image_ms_at_batch_size": batch_ms / batch_size,
        "batch_latency_ms": batch_ms,
        "peak_vram_mib": peak_bytes / (1024 * 1024),
    }


def train_one_stage(
    model: nn.Module,
    loader: DataLoader,
    sampler,
    criterion: VERITASLossV4,
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
    tot_loss = tot_cls = tot_aux = tot_seg = 0.0
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
        tot_aux += float(losses["aux"])
        tot_seg += float(losses["seg"])
        if rank == 0 and (step % max(1, n_steps // 4) == 0 or step == n_steps):
            dt = time.time() - t0
            print(
                f"  [{stage_name} ep {epoch}] step {step}/{n_steps} "
                f"loss={tot_loss/step:.4f} cls={tot_cls/step:.4f} aux={tot_aux/step:.4f} seg={tot_seg/step:.4f} ({dt:.0f}s)",
                flush=True,
            )
    return time.time() - t0


def run_variant(variant: str, cfg: dict, index: dict, working: Path, is_ddp: bool, rank: int, world_size: int, local_rank: int, device: torch.device):
    set_seed(cfg["seed"])
    is_v4 = variant == "veritas_v4"
    if rank == 0:
        print(f"\n==================== RUNNING {variant.upper()} ====================", flush=True)

    train_ds = ProtoMaskDataset(index["train"], cfg["image_size"], augment=True, forensic_aug=cfg["use_forensic_aug"], seed=cfg["seed"])
    val_ds = ProtoMaskDataset(index["val"], cfg["image_size"], augment=False, forensic_aug=False, seed=cfg["seed"])
    test_ds = ProtoMaskDataset(index["test"], cfg["image_size"], augment=False, forensic_aug=False, seed=cfg["seed"])

    tr_samp = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg["seed"], drop_last=True) if is_ddp else None
    va_samp = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None
    te_samp = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None

    nw = cfg["num_workers"]
    kw = dict(num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=(nw > 0))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=tr_samp, shuffle=(tr_samp is None), drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size_per_gpu"], sampler=va_samp, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size_per_gpu"], sampler=te_samp, shuffle=False, **kw)

    grad_accum = max(1, cfg["target_effective_batch"] // (cfg["batch_size_per_gpu"] * world_size))

    if is_ddp:
        if rank == 0:
            _ = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        dist.barrier()
    raw_model = VERITASV4(variant=variant, pretrained=True).to(device).to(memory_format=torch.channels_last)
    criterion = VERITASLossV4(
        lambda_cls=cfg["lambda_cls"],
        lambda_aux=cfg["lambda_aux"],
        lambda_seg=cfg["lambda_seg"],
        dice_weight=cfg["dice_weight"],
        is_v4=is_v4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["use_amp"] and device.type == "cuda"))

    best_val_acc = -1.0
    best_path = working / "checkpoints" / f"{variant}_v4proto_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    t_start = time.time()

    # Stage 1: Frozen-backbone head warmup (identical schedule for both Baseline & VERITAS v4)
    if cfg["warmup_epochs"] > 0:
        raw_model.freeze_backbone(True)
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False) if is_ddp else raw_model
        opt = build_optimizer(model, lr=cfg["warmup_lr"], weight_decay=cfg["weight_decay"], backbone_mult=1.0)
        updates = max(1, math.ceil(len(train_loader) / grad_accum) * cfg["warmup_epochs"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=cfg["warmup_lr"] * 0.1)
        for ep in range(1, cfg["warmup_epochs"] + 1):
            sec = train_one_stage(model, train_loader, tr_samp, criterion, opt, sched, scaler, device, ep, grad_accum, cfg["use_amp"], rank, "Stage1-Warmup")
            val_m, _ = evaluate(raw_model, val_loader, device, is_ddp, is_v4, cfg["use_amp"])
            if rank == 0:
                print(f"  [Stage1 ep {ep}] VAL acc={val_m['accuracy']:.4f} f1={val_m['f1']:.4f} auc={val_m['roc_auc']:.4f} ({sec:.0f}s)", flush=True)
                history.append({"stage": "warmup", "epoch": ep, "seconds": sec, **val_m})
                if val_m["accuracy"] > best_val_acc:
                    best_val_acc = val_m["accuracy"]
                    torch.save(raw_model.state_dict(), best_path)
        del model, opt, sched

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
            val_m, val_recs = evaluate(raw_model, val_loader, device, is_ddp, is_v4, cfg["use_amp"])
            if rank == 0:
                extra = f" aux_acc={val_m['aux_forensic_accuracy']:.4f} gates=[{val_m['mean_gate_rgb']:.2f},{val_m['mean_gate_srm']:.2f},{val_m['mean_gate_freq']:.2f},{val_m['mean_gate_contrast']:.2f}]" if is_v4 else ""
                print(f"  [Stage2 ep {ep}] VAL acc={val_m['accuracy']:.4f} f1={val_m['f1']:.4f} auc={val_m['roc_auc']:.4f}{extra} ({sec:.0f}s)", flush=True)
                history.append({"stage": "finetune", "epoch": ep, "seconds": sec, **val_m})
                if val_m["accuracy"] > best_val_acc:
                    best_val_acc = val_m["accuracy"]
                    torch.save(raw_model.state_dict(), best_path)
                    pd.DataFrame(val_recs).to_csv(working / "predictions" / f"{variant}_v4proto_val_predictions.csv", index=False)
        del model, opt, sched

    if is_ddp:
        dist.barrier()

    # Load best checkpoint and evaluate on FULL 10,905 OpenForensics Test set
    raw_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_m, test_recs = evaluate(raw_model, test_loader, device, is_ddp, is_v4, cfg["use_amp"])
    total_sec = time.time() - t_start

    if rank == 0:
        bench = benchmark_inference(raw_model, device, batch_size=4, size=cfg["image_size"])
        test_m.update(bench)
        test_m["total_train_eval_seconds"] = total_sec
        test_m["variant"] = variant
        pred_dir = working / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_recs).to_csv(pred_dir / f"{variant}_v4proto_test_predictions.csv", index=False)
        (working / f"{variant}_v4proto_summary.json").write_text(json.dumps({"metrics": test_m, "history": history, "config": cfg}, indent=2), encoding="utf-8")
        print(f"=== FINAL TEST ({variant}) ===\n" + json.dumps(test_m, indent=2), flush=True)

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
    (working / "predictions").mkdir(parents=True, exist_ok=True)

    is_ddp, rank, world_size, local_rank, device = setup_ddp()
    try:
        if rank == 0:
            print(f"DDP={is_ddp} world_size={world_size} device={device} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
        index = build_index(cfg, working, rank)
        for variant in cfg["run_variants"]:
            run_variant(variant, cfg, index, working, is_ddp, rank, world_size, local_rank, device)
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


def build_notebook():
    arch_code = build_standalone_arch_code()
    train_script = TRAIN_SCRIPT_BODY.replace("__ARCH_V4_CODE__", arch_code)

    cells = [
        make_cell(
            "markdown",
            """# VERITAS OW v4 — Head-to-Head Prototype (`T4 × 2` DDP)

Tests the redesigned **VERITAS v4** architecture against **Baseline EfficientNet-B7** under the **exact same data subset, forensic augmentations, and 2-stage schedule**, evaluated on **all 10,905 OpenForensics Test images**.

### Kaggle Setup (3 steps)
1. **Accelerator:** Select **`GPU T4 x2`** (automatically falls back to single GPU if `T4 x1`).
2. **Add Data:** Attach:
   - Dataset: `ahmedgamal/openforensics-wild-dataset`
   - Notebook output: `clarenceshanley/veritas-ow-00-generate-masks`
3. **Run All:** Default `28,000` stratified train images (14k Real / 14k Fake) + full `10,905` test set runs **both** Baseline B7 and VERITAS v4 in **~1.3–1.6 hours** on `T4 × 2`.

### What changed in VERITAS v4 vs OW v3
- **Zero-sync `ForensicStreamsV4`:** Eliminates per-image Python/`content_bounds` GPU→CPU syncs; uses vectorized content-mean fill + batched `rfft2/irfft2` directional frequency bands + 2-px eroded valid-mask gating.
- **Deeper multi-scale segmentation:** ASPP taps B7 **Stage 7 (`19×19`) + Stage 5 (`38×38`) + Stage 2 (`150×150`) + spatial SRM/Freq (`38×38`)**.
- **Active Segmentation → Classification feedback:** Mask-guided Foreground-vs-Background contrastive pooling (`global`, `fg`, `bg`, `|fg - bg|`) across RGB, SRM, and Freq.
- **Anti-collapse multi-modal fusion:** Independent per-stream `LayerNorm` + balanced 256-d tokens + learnable softmax stream gates + auxiliary forensic-only loss (`lambda_aux=0.30`).
""",
        ),
        make_cell(
            "code",
            """import json
from pathlib import Path
import torch

WORKING = Path("/kaggle/working/ow_v4_proto") if Path("/kaggle/working").exists() else Path.cwd() / "ow_v4_proto"
WORKING.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "working_dir": str(WORKING),
    "seed": 42,
    "image_size": 600,
    # 14,000 per class = 28,000 train images (20% of OpenForensics Train).
    # Set to None to run the full 140,000 train set.
    "train_per_class": 12000,
    "val_per_class": 4000,
    "test_per_class": None,  # Always evaluate on ALL 10,905 OpenForensics Test images
    "warmup_epochs": 1,      # Stage 1: frozen backbone, train heads only (both models!)
    "finetune_epochs": 2,    # Stage 2: full end-to-end fine-tune (both models!)
    "batch_size_per_gpu": 4, # 4 x 2 GPUs x 2 grad_accum = 16 effective batch (safe for 15GB T4 at 600px)
    "target_effective_batch": 16,
    "num_workers": 2, # 2 workers x 2 DDP ranks = 4 total Kaggle CPU cores
    "use_amp": True,
    "use_forensic_aug": True,  # Applied identically to BOTH Baseline B7 and VERITAS v4
    "warmup_lr": 3e-4,
    "finetune_lr": 1.5e-4,
    "backbone_lr_multiplier": 0.1,
    "weight_decay": 1e-4,
    "lambda_cls": 1.0,
    "lambda_aux": 0.30,
    "lambda_seg": 0.50,
    "dice_weight": 1.0,
    "run_variants": ["baseline", "veritas_v4"],
}

cfg_path = WORKING / "config.json"
cfg_path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"GPUs detected: {n_gpus}", [torch.cuda.get_device_name(i) for i in range(n_gpus)])
print("Saved config to:", cfg_path)
""",
        ),
        make_cell(
            "code",
            "# Write standalone DDP-safe training script\n"
            f"SCRIPT_CODE = {train_script!r}\n"
            "script_path = WORKING / 'train_v4_proto.py'\n"
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
print("Head-to-head training & test evaluation finished cleanly!")
""",
        ),
        make_cell(
            "code",
            """import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def mcnemar_exact_p(b_wins: int, v_wins: int) -> float:
    n = b_wins + v_wins
    if n == 0:
        return 1.0
    k = min(b_wins, v_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


base_sum = json.loads((WORKING / "baseline_v4proto_summary.json").read_text())["metrics"]
v4_sum = json.loads((WORKING / "veritas_v4_v4proto_summary.json").read_text())["metrics"]

df_b = pd.read_csv(WORKING / "predictions/baseline_v4proto_test_predictions.csv").sort_values("rel_key").reset_index(drop=True)
df_v = pd.read_csv(WORKING / "predictions/veritas_v4_v4proto_test_predictions.csv").sort_values("rel_key").reset_index(drop=True)
assert (df_b["rel_key"] == df_v["rel_key"]).all(), "Test set mismatch!"

y = df_b["ground_truth_label"].values
pb = df_b["fake_probability"].values
pv = df_v["fake_probability"].values

b_ok = (pb >= 0.5).astype(int) == y
v_ok = (pv >= 0.5).astype(int) == y
v_only = int((v_ok & ~b_ok).sum())
b_only = int((b_ok & ~v_ok).sum())
p_val = mcnemar_exact_p(b_only, v_only)

eps = 1e-6
lb = np.log(np.clip(pb, eps, 1 - eps) / (1 - np.clip(pb, eps, 1 - eps)))
lf = np.log(np.clip(pv, eps, 1 - eps) / (1 - np.clip(pv, eps, 1 - eps)))
logit_r = float(np.corrcoef(lb, lf)[0, 1])

# Calibrate threshold strictly on Validation (no Test leakage!)
for name, val_file, test_p in [
    ("baseline", WORKING / "predictions/baseline_v4proto_val_predictions.csv", pb),
    ("veritas_v4", WORKING / "predictions/veritas_v4_v4proto_val_predictions.csv", pv),
]:
    if val_file.exists():
        dv = pd.read_csv(val_file)
        vy, vp = dv["ground_truth_label"].values, dv["fake_probability"].values
        best_thr, best_acc = 0.5, -1.0
        for thr in np.linspace(0.1, 0.9, 81):
            a = accuracy_score(vy, (vp >= thr).astype(int))
            if a > best_acc:
                best_acc, best_thr = a, float(thr)
        cal_test_acc = accuracy_score(y, (test_p >= best_thr).astype(int))
        print(f"[Val-Calibrated] {name}: best_val_thr={best_thr:.2f} -> Test Acc={cal_test_acc*100:.2f}%")

summary_table = pd.DataFrame([
    {
        "Model": "Baseline EffNet-B7 (proto)",
        "Acc (%)": round(base_sum["accuracy"] * 100, 2),
        "F1": round(base_sum["f1"], 4),
        "Prec": round(base_sum["precision"], 4),
        "Rec": round(base_sum["recall"], 4),
        "ROC-AUC": round(base_sum["roc_auc"], 4),
        "FP": base_sum["fp"],
        "FN": base_sum["fn"],
        "ms/img (b4)": round(base_sum["per_image_ms_at_batch_size"], 2),
        "VRAM (MiB)": round(base_sum["peak_vram_mib"], 1),
    },
    {
        "Model": "VERITAS v4 (proto)",
        "Acc (%)": round(v4_sum["accuracy"] * 100, 2),
        "F1": round(v4_sum["f1"], 4),
        "Prec": round(v4_sum["precision"], 4),
        "Rec": round(v4_sum["recall"], 4),
        "ROC-AUC": round(v4_sum["roc_auc"], 4),
        "FP": v4_sum["fp"],
        "FN": v4_sum["fn"],
        "ms/img (b4)": round(v4_sum["per_image_ms_at_batch_size"], 2),
        "VRAM (MiB)": round(v4_sum["peak_vram_mib"], 1),
    },
])
display(summary_table)

print(f"Delta Accuracy: {(v4_sum['accuracy'] - base_sum['accuracy'])*100:+.2f} pp")
print(f"Paired McNemar (n={len(y)}): VERITAS-only wins={v_only}, B7-only wins={b_only}, exact p={p_val:.3e}")
print(f"Logit correlation r(B7, v4) = {logit_r:.4f} (OW v3 was 0.9884; lower means v4 learned genuinely distinct cues)")
print(f"Aux Forensic-Only (SRM+Freq+Mask, NO RGB) Test Acc={v4_sum['aux_forensic_accuracy']*100:.2f}%, AUC={v4_sum['aux_forensic_roc_auc']:.4f}")
print(
    f"Learned Stream Gates -> RGB={v4_sum['mean_gate_rgb']:.3f}, SRM={v4_sum['mean_gate_srm']:.3f}, "
    f"Freq={v4_sum['mean_gate_freq']:.3f}, MaskContrast={v4_sum['mean_gate_contrast']:.3f}"
)
print(
    f"Segmentation -> Fake Dice={v4_sum['dice_teacher_fake_only']:.4f}, mIoU={v4_sum['miou_teacher_fake_only']:.4f}, "
    f"Real Pos Frac={v4_sum['real_mean_positive_fraction']*100:.2f}%, Real Nonempty={v4_sum['real_mask_nonempty_rate']*100:.2f}%"
)
""",
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Generated {OUT_NB} ({OUT_NB.stat().st_size} bytes)")


if __name__ == "__main__":
    build_notebook()
