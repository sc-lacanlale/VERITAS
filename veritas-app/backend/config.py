"""Inference settings. Values match the trained Full VERITAS notebook protocol."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]


def _default_checkpoint() -> Path:
    env = os.environ.get("MODEL_CHECKPOINT")
    if env:
        return Path(env).expanduser()
    models = REPO_ROOT / "models"
    for candidate in (
        models / "full_veritas_defacto_best.zip",
        models / "full_veritas_defacto_best.pt",
        *sorted(models.glob("full_veritas_defacto_best*.zip"), key=lambda p: p.stat().st_mtime, reverse=True),
        models / "full_veritas_wild_best.zip",
        models / "full_veritas_wild_best.pt",
        models / "full_veritas_best.zip",
        models / "full_veritas_best.pt",
        APP_ROOT / "backend" / "models" / "checkpoint.pt",
    ):
        if candidate.exists():
            return candidate
    return models / "full_veritas_defacto_best.zip"


def _checkpoint_kind(path: Path) -> str:
    name = path.name.lower()
    if "wild" in name and "defacto" not in name:
        return "wild"
    return "openforensics"


_CHECKPOINT = _default_checkpoint()
_KIND = _checkpoint_kind(_CHECKPOINT)

MODEL_CONFIG = {
    "checkpoint": _CHECKPOINT,
    "device": os.environ.get("MODEL_DEVICE", "cuda"),
    "image_size": 600,
    # OpenForensics Full: GT box + 10% then 600 letterbox. Wild-only weights: already-cropped faces.
    "face_margin": 0.0 if _KIND == "wild" else 0.10,
    "classification_threshold": 0.5,
    "mask_threshold": 0.5,
    "face_score_threshold": 0.5,
    "multi_stream": True,
    "segmentation": True,
    # Wild-only DeepLab was an oval surrogate. OpenForensics Full used GT masks — show them.
    "emit_masks": _KIND != "wild",
    "checkpoint_kind": _KIND,
    "backbone": "efficientnet_b7",
    "max_upload_bytes": int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)),
    "allowed_suffixes": {".jpg", ".jpeg", ".png", ".webp"},
    "allowed_mimes": {"image/jpeg", "image/png", "image/webp"},
}
