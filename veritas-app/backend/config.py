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
    for candidate in (
        REPO_ROOT / "models" / "full_veritas_best.zip",
        REPO_ROOT / "models" / "full_veritas_best.pt",
        APP_ROOT / "backend" / "models" / "checkpoint.pt",
    ):
        if candidate.exists():
            return candidate
    return REPO_ROOT / "models" / "full_veritas_best.zip"


MODEL_CONFIG = {
    "checkpoint": _default_checkpoint(),
    "device": os.environ.get("MODEL_DEVICE", "cuda"),
    "image_size": 600,
    "face_margin": 0.10,
    "classification_threshold": 0.5,
    "mask_threshold": 0.5,
    "face_score_threshold": 0.5,
    "multi_stream": True,
    "segmentation": True,
    "backbone": "efficientnet_b7",
    "max_upload_bytes": int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)),
    "allowed_suffixes": {".jpg", ".jpeg", ".png", ".webp"},
    "allowed_mimes": {"image/jpeg", "image/png", "image/webp"},
}
