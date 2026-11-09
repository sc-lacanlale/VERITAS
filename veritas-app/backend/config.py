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
        # OW_04 output (OpenForensics Wild + generated masks) — the final Full model.
        models / "full_veritas_ow_best.zip",
        models / "full_veritas_ow_best.pt",
        *sorted(models.glob("full_veritas_ow_best*.zip"), key=lambda p: p.stat().st_mtime, reverse=True),
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
    # Face-quality gate: a face must clear these to be analyzable. These are real,
    # measurable signals from the detector output (confidence + bbox geometry).
    # - quality_min_confidence: detector score below this = "not clear enough".
    # - quality_min_face_frac: face's larger side must be >= this fraction of the
    #   image's larger side, else the face is too small/obscured to analyze.
    "quality_min_confidence": float(os.environ.get("FACE_QUALITY_MIN_CONF", 0.60)),
    "quality_min_face_frac": float(os.environ.get("FACE_QUALITY_MIN_FRAC", 0.08)),
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


# --------------------------------------------------------------------------- #
# Baseline (RGB EfficientNet-B7, classification only) — optional, for the     #
# app's model selector. Looked up on every call so a checkpoint dropped into  #
# models/ is picked up without a restart. Full VERITAS settings above are     #
# unchanged.                                                                  #
# --------------------------------------------------------------------------- #
def baseline_checkpoint() -> Path | None:
    env = os.environ.get("BASELINE_CHECKPOINT")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None
    models = REPO_ROOT / "models"
    for name in (
        "baseline_effb7_ow_best.pt",
        "baseline_effb7_ow_best.zip",
        "baseline_effb7_defacto_best.pt",
        "baseline_effb7_defacto_best.zip",
    ):
        if (models / name).exists():
            return models / name
    # Any other baseline_effb7*_best file (e.g. a Chrome "(1)" download). The Wild-only
    # baseline is skipped on purpose: the notebooks say it is not the comparison baseline.
    others = [
        p
        for p in models.glob("baseline_effb7*_best*")
        if p.suffix.lower() in (".pt", ".zip") and "wild" not in p.name.lower() and "_repacked" not in p.name
    ]
    if others:
        return max(others, key=lambda p: p.stat().st_mtime)
    return None
