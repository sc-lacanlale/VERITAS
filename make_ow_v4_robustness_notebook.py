"""Generate Kaggle T4x2 DDP robustness stress-test notebook for saved VERITAS OW v4 vs Baseline B7 checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCH_V4_PATH = ROOT / "veritas-app" / "backend" / "inference" / "architecture_v4.py"
OUT_DIR = ROOT / "notebooks-ow-v4"
OUT_NB = OUT_DIR / "OWV4_robustness_stress_test.ipynb"


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


EVAL_SCRIPT_BODY = r'''from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
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
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
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


def discover_wild_root() -> Path:
    roots = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else []
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


def unpack_from_zips(input_root: Path, cache_dir: Path, wanted_substrings: tuple[str, ...]) -> list[Path]:
    roots = []
    if not input_root.exists():
        return roots
    for archive in sorted(input_root.rglob("_output_.zip")):
        key = hashlib.sha256((str(archive) + "|".join(wanted_substrings)).encode()).hexdigest()[:12]
        target = cache_dir / f"unpacked_{key}"
        marker = target / ".done"
        if marker.exists():
            roots.append(target)
            continue
        with zipfile.ZipFile(archive) as z:
            members = [
                m for m in z.infolist()
                if not m.is_dir() and any(s in m.filename.replace("\\", "/") for s in wanted_substrings)
            ]
            if not members:
                continue
            target.mkdir(parents=True, exist_ok=True)
            for member in members:
                rel = PurePosixPath(member.filename.replace("\\", "/"))
                if rel.is_absolute() or ".." in rel.parts:
                    continue
                dst = target / Path(*rel.parts)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out)
            marker.write_text("ok", encoding="utf-8")
            roots.append(target)
    return roots


def find_file_in_roots(roots: list[Path], filename: str) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob(filename):
            if p.is_file():
                return p
    return None


def load_mask_table(working: Path) -> pd.DataFrame:
    frames = []
    roots = [p for p in (Path("/kaggle/input"), Path("/kaggle/working"), working, Path.cwd()) if p.exists()]
    roots.extend(unpack_from_zips(Path("/kaggle/input"), working / "cache", ("ow_masks/",)))
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


def build_test_index(working: Path, rank: int, max_per_class: int | None = None) -> pd.DataFrame:
    cache_path = working / "cache" / "test_index_robustness.json"
    if rank == 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        wild_root = discover_wild_root()
        masks = load_mask_table(working)
        imgs = list_images(wild_root, "test")
        merged = imgs.merge(
            masks[["rel_key", "mask_path", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "positive_frac"]],
            on="rel_key",
            how="left",
        )
        ready = merged.loc[merged["mask_path"].notna()].copy()
        if max_per_class is not None:
            parts = []
            rng = np.random.RandomState(77)
            for _, g in ready.groupby("label"):
                n = min(int(max_per_class), len(g))
                parts.append(g.sample(n=n, random_state=rng) if n < len(g) else g)
            ready = pd.concat(parts)
        ready = ready.reset_index(drop=True)
        print(f"[TEST INDEX] {len(ready)} images (real={(ready.label==0).sum()}, fake={(ready.label==1).sum()})", flush=True)
        cache_path.write_text(json.dumps(ready.to_dict("records")), encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
    return pd.DataFrame(json.loads(cache_path.read_text(encoding="utf-8")))


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


def apply_degradation(crop: np.ndarray, condition: str) -> np.ndarray:
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
        return cv2.GaussianBlur(crop, (5, 5), 2.0)
    if condition == "resize_0.5x":
        h, w = crop.shape[:2]
        small = cv2.resize(crop, (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"Unknown condition: {condition}")


class StressTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, condition: str):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.condition = condition

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        bgr = cv2.imread(row.path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Unreadable image: {row.path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        x0, y0 = max(0, int(float(row.crop_x0))), max(0, int(float(row.crop_y0)))
        x1, y1 = min(w, max(x0 + 1, int(float(row.crop_x1)))), min(h, max(y0 + 1, int(float(row.crop_y1))))
        crop = apply_degradation(rgb[y0:y1, x0:x1], self.condition)

        mask = cv2.imread(str(row.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unreadable mask: {row.mask_path}")
        if mask.shape != (self.size, self.size):
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        canvas, valid = letterbox_rgb(crop, self.size)
        x = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN_NP) / IMAGENET_STD_NP
        x = np.transpose(x, (2, 0, 1))
        target = (mask.astype(np.float32) > 127.0).astype(np.float32)
        return {
            "image": torch.from_numpy(x.copy()),
            "mask": torch.from_numpy(target[None].copy()),
            "valid_mask": torch.from_numpy(valid[None].copy()),
            "label": torch.tensor([float(row.label)], dtype=torch.float32),
            "rel_key": str(row.rel_key),
        }


@torch.inference_mode()
def evaluate_condition(model: nn.Module, loader: DataLoader, device: torch.device, is_ddp: bool, is_v4: bool):
    model.eval()
    local_recs = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        vm = batch["valid_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x, valid_mask=vm)
        logits = out["classification_logits"].float().squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        labels = batch["label"].squeeze(-1).numpy().astype(int)

        aux_probs = np.full_like(probs, np.nan)
        gates_np = np.full((len(probs), 4), np.nan, dtype=np.float32)
        seg_pos_frac = np.full_like(probs, np.nan)
        seg_dice = np.full_like(probs, np.nan)
        seg_iou = np.full_like(probs, np.nan)

        if is_v4:
            aux_logits = out["aux_forensic_logits"].float().squeeze(-1).cpu().numpy()
            aux_probs = 1.0 / (1.0 + np.exp(-np.clip(aux_logits, -40, 40)))
            gates_np = out["stream_gates"].float().cpu().numpy()
            pred_bin = (out["segmentation_logits"].float().sigmoid() >= 0.5) & (vm > 0.5)
            gt_bin = (batch["mask"].to(device, non_blocking=True) > 0.5) & (vm > 0.5)
            vpx = (vm > 0.5).flatten(1).sum(1).clamp_min(1).float()
            p_sum = pred_bin.flatten(1).sum(1).float()
            g_sum = gt_bin.flatten(1).sum(1).float()
            inter = (pred_bin & gt_bin).flatten(1).sum(1).float()
            union = (pred_bin | gt_bin).flatten(1).sum(1).float()
            seg_pos_frac = (p_sum / vpx).cpu().numpy()
            seg_dice = torch.where(g_sum > 0, (2.0 * inter + 1e-6) / (p_sum + g_sum + 1e-6), torch.full_like(inter, float("nan"))).cpu().numpy()
            seg_iou = torch.where(g_sum > 0, (inter + 1e-6) / (union + 1e-6), torch.full_like(inter, float("nan"))).cpu().numpy()

        for i in range(len(labels)):
            local_recs.append({
                "rel_key": batch["rel_key"][i],
                "label": int(labels[i]),
                "logit": float(logits[i]),
                "prob": float(probs[i]),
                "aux_prob": float(aux_probs[i]),
                "seg_pos_frac": float(seg_pos_frac[i]),
                "seg_dice": float(seg_dice[i]),
                "seg_iou": float(seg_iou[i]),
                "g_rgb": float(gates_np[i, 0]),
                "g_srm": float(gates_np[i, 1]),
                "g_freq": float(gates_np[i, 2]),
                "g_con": float(gates_np[i, 3]),
            })

    if is_ddp:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_recs)
        all_recs = [r for sub in gathered for r in sub]
    else:
        all_recs = local_recs

    dedup = {r["rel_key"]: r for r in all_recs}
    df = pd.DataFrame(dedup.values()).sort_values("rel_key").reset_index(drop=True)
    return df


def summarize_pair(b_df: pd.DataFrame, v_df: pd.DataFrame, condition: str) -> dict:
    m = b_df.merge(v_df, on=["rel_key", "label"], suffixes=("_b", "_v"))
    y = m["label"].values.astype(int)
    pb = m["prob_b"].values
    pv = m["prob_v"].values
    paux = m["aux_prob_v"].values
    lb = m["logit_b"].values
    lv = m["logit_v"].values

    def cls_metrics(p):
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            "acc": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "prec": float(precision_score(y, pred, zero_division=0)),
            "rec": float(recall_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }

    bm = cls_metrics(pb)
    vm = cls_metrics(pv)
    am = cls_metrics(paux)
    blend_va = cls_metrics(0.5 * pv + 0.5 * paux)
    blend_ba = cls_metrics(0.5 * pb + 0.5 * paux)

    b_ok = ((pb >= 0.5).astype(int) == y)
    v_ok = ((pv >= 0.5).astype(int) == y)
    a_ok = ((paux >= 0.5).astype(int) == y)

    v_only = int((v_ok & ~b_ok).sum())
    b_only = int((~v_ok & b_ok).sum())
    p_v = float(binomtest(min(v_only, b_only), v_only + b_only, 0.5).pvalue) if (v_only + b_only) > 0 else 1.0

    a_only = int((a_ok & ~b_ok).sum())
    b_only_a = int((~a_ok & b_ok).sum())
    p_a = float(binomtest(min(a_only, b_only_a), a_only + b_only_a, 0.5).pvalue) if (a_only + b_only_a) > 0 else 1.0

    b_err = ~b_ok
    v_err = ~v_ok
    shared_err = int((b_err & v_err).sum())

    reals = m[m["label"] == 0]
    fakes = m[m["label"] == 1]
    real_corr = float(np.corrcoef(reals["prob_v"], reals["seg_pos_frac_v"])[0, 1]) if len(reals) > 2 and reals["seg_pos_frac_v"].std() > 0 else float("nan")

    return {
        "condition": condition,
        "n": int(len(m)),
        "baseline": bm,
        "veritas_v4": vm,
        "aux_forensic_only": am,
        "blend_v4_aux": blend_va,
        "blend_b7_aux": blend_ba,
        "delta_v4_vs_b7_pp": round((vm["acc"] - bm["acc"]) * 100.0, 3),
        "delta_aux_vs_b7_pp": round((am["acc"] - bm["acc"]) * 100.0, 3),
        "mcnemar_v4_vs_b7": {"v4_only": v_only, "b7_only": b_only, "p": p_v},
        "mcnemar_aux_vs_b7": {"aux_only": a_only, "b7_only": b_only_a, "p": p_a},
        "decision_agreement": float(((pb >= 0.5) == (pv >= 0.5)).mean()),
        "shared_error_frac_of_v4": float(shared_err / max(1, int(v_err.sum()))),
        "logit_corr_b7_v4": float(np.corrcoef(lb, lv)[0, 1]) if len(lb) > 2 else float("nan"),
        "seg_fake_dice": float(np.nanmean(fakes["seg_dice_v"])),
        "seg_fake_miou": float(np.nanmean(fakes["seg_iou_v"])),
        "seg_real_pos_frac": float(np.nanmean(reals["seg_pos_frac_v"])),
        "seg_real_nonempty": float((reals["seg_pos_frac_v"] > 0).mean()),
        "real_mask_prob_corr": real_corr,
        "gates": {
            "rgb": float(np.nanmean(m["g_rgb_v"])),
            "srm": float(np.nanmean(m["g_srm_v"])),
            "freq": float(np.nanmean(m["g_freq_v"])),
            "contrast": float(np.nanmean(m["g_con_v"])),
        },
    }


def locate_checkpoints(working: Path, rank: int) -> dict[str, str]:
    info_path = working / "cache" / "ckpt_paths.json"
    if rank == 0:
        info_path.parent.mkdir(parents=True, exist_ok=True)
        roots = [p for p in (Path("/kaggle/input"), Path("/kaggle/working"), working, Path.cwd()) if p.exists()]
        b_ckpt = find_file_in_roots(roots, "baseline_v4proto_best.pt")
        v_ckpt = find_file_in_roots(roots, "veritas_v4_v4proto_best.pt")
        b_csv = find_file_in_roots(roots, "baseline_v4proto_test_predictions.csv")
        v_csv = find_file_in_roots(roots, "veritas_v4_v4proto_test_predictions.csv")
        if b_ckpt is None or v_ckpt is None:
            extra = unpack_from_zips(
                Path("/kaggle/input"),
                working / "cache",
                ("baseline_v4proto_best.pt", "veritas_v4_v4proto_best.pt", "_test_predictions.csv"),
            )
            roots.extend(extra)
            b_ckpt = b_ckpt or find_file_in_roots(roots, "baseline_v4proto_best.pt")
            v_ckpt = v_ckpt or find_file_in_roots(roots, "veritas_v4_v4proto_best.pt")
            b_csv = b_csv or find_file_in_roots(roots, "baseline_v4proto_test_predictions.csv")
            v_csv = v_csv or find_file_in_roots(roots, "veritas_v4_v4proto_test_predictions.csv")
        if b_ckpt is None or v_ckpt is None:
            raise FileNotFoundError(f"Could not find saved checkpoints under /kaggle/input (b={b_ckpt}, v={v_ckpt})")
        info = {
            "baseline_ckpt": str(b_ckpt),
            "veritas_v4_ckpt": str(v_ckpt),
            "baseline_clean_csv": str(b_csv) if b_csv else "",
            "veritas_v4_clean_csv": str(v_csv) if v_csv else "",
        }
        print("[CHECKPOINTS]", json.dumps(info, indent=2), flush=True)
        info_path.write_text(json.dumps(info), encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
    return json.loads(info_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    working = Path(cfg["working_dir"])
    working.mkdir(parents=True, exist_ok=True)

    is_ddp, rank, world_size, local_rank, device = setup_ddp()
    try:
        if rank == 0:
            print(f"DDP={is_ddp} world_size={world_size} device={device} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
        ckpts = locate_checkpoints(working, rank)
        test_df = build_test_index(working, rank, max_per_class=cfg.get("test_per_class"))

        # Load models once (pretrained=False because we immediately load saved state_dicts -> zero download delay!)
        models = {}
        for variant, ckpt_key in [("baseline", "baseline_ckpt"), ("veritas_v4", "veritas_v4_ckpt")]:
            m = VERITASV4(variant=variant, pretrained=False).to(device).to(memory_format=torch.channels_last)
            state = torch.load(ckpts[ckpt_key], map_location=device, weights_only=True)
            m.load_state_dict(state)
            m.eval()
            models[variant] = m

        all_results = []
        for cond in cfg["conditions"]:
            t0 = time.time()
            if rank == 0:
                print(f"\n==================== CONDITION: {cond} (n={len(test_df)}) ====================", flush=True)
            ds = StressTestDataset(test_df, cfg["image_size"], cond)
            samp = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp else None
            nw = cfg["num_workers"]
            loader = DataLoader(ds, batch_size=cfg["batch_size_per_gpu"], sampler=samp, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"), persistent_workers=False)

            b_df = evaluate_condition(models["baseline"], loader, device, is_ddp, is_v4=False)
            v_df = evaluate_condition(models["veritas_v4"], loader, device, is_ddp, is_v4=True)

            if rank == 0:
                summary = summarize_pair(b_df, v_df, cond)
                summary["eval_seconds"] = round(time.time() - t0, 1)
                all_results.append(summary)
                print(
                    f"[{cond}] B7={summary['baseline']['acc']*100:.2f}% (AUC {summary['baseline']['auc']:.4f}) | "
                    f"v4={summary['veritas_v4']['acc']*100:.2f}% (AUC {summary['veritas_v4']['auc']:.4f}, delta {summary['delta_v4_vs_b7_pp']:+.2f} pp, p={summary['mcnemar_v4_vs_b7']['p']:.2e}) | "
                    f"Aux={summary['aux_forensic_only']['acc']*100:.2f}% (delta {summary['delta_aux_vs_b7_pp']:+.2f} pp, p={summary['mcnemar_aux_vs_b7']['p']:.2e}) | "
                    f"Dice={summary['seg_fake_dice']:.4f} | r={summary['logit_corr_b7_v4']:.4f} | {summary['eval_seconds']}s",
                    flush=True,
                )
            if is_ddp:
                dist.barrier()

        if rank == 0:
            out_json = working / "robustness_summary.json"
            out_json.write_text(json.dumps({"config": cfg, "results": all_results}, indent=2), encoding="utf-8")
            print("\n=== ROBUSTNESS SUMMARY JSON ===", flush=True)
            print(json.dumps(all_results, indent=2), flush=True)
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
    eval_script = EVAL_SCRIPT_BODY.replace("__ARCH_V4_CODE__", arch_code)

    cells = [
        make_cell(
            "markdown",
            """# VERITAS OW v4 vs Baseline B7 — Robustness Stress Test (All 10,905 OpenForensics Test Images)

Evaluates the **already-trained saved checkpoints** (`baseline_v4proto_best.pt` and `veritas_v4_v4proto_best.pt`, zero retraining) on **all 10,905 OpenForensics Test images** across 5 conditions:
1. `clean` (original OpenForensics Test + error overlap, Real-mask coupling, and blend checks)
2. `jpeg_q50` (moderate social-media JPEG recompression)
3. `jpeg_q30` (heavy Messenger/WhatsApp JPEG recompression)
4. `blur_sigma2` (Gaussian blur, `k=5, sigma=2.0`)
5. `resize_0.5x` (downscale 50% -> upscale back)
""",
        ),
        make_cell(
            "code",
            """import json
from pathlib import Path
import torch

WORKING = Path("/kaggle/working/ow_v4_robust") if Path("/kaggle/working").exists() else Path.cwd() / "ow_v4_robust"
WORKING.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "working_dir": str(WORKING),
    "image_size": 600,
    "test_per_class": None,  # ALL 10,905 OpenForensics Test images
    "batch_size_per_gpu": 12,
    "num_workers": 2,
    "conditions": ["clean", "jpeg_q50", "jpeg_q30", "blur_sigma2", "resize_0.5x"],
}

cfg_path = WORKING / "config.json"
cfg_path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"GPUs detected: {n_gpus}", [torch.cuda.get_device_name(i) for i in range(n_gpus)])
""",
        ),
        make_cell(
            "code",
            f"SCRIPT_CODE = {eval_script!r}\n"
            "script_path = WORKING / 'eval_v4_robustness.py'\n"
            "script_path.write_text(SCRIPT_CODE, encoding='utf-8')\n"
            "print('Wrote eval script:', script_path, 'bytes:', len(SCRIPT_CODE))\n",
        ),
        make_cell(
            "code",
            """import subprocess, sys

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
    raise RuntimeError(f"Robustness eval failed with exit code {rc}")
print("Robustness stress test finished cleanly!")
""",
        ),
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
    OUT_NB.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Generated {OUT_NB} ({OUT_NB.stat().st_size:,} bytes, {len(cells)} cells)")


if __name__ == "__main__":
    build_notebook()
