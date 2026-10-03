"""Optional annotated overlay and PNG encoding for API responses."""
from __future__ import annotations

import base64

import cv2
import numpy as np


def encode_png(mask_or_rgb: np.ndarray) -> str:
    if mask_or_rgb.ndim == 2:
        img = np.clip(mask_or_rgb * 255.0 if mask_or_rgb.dtype != np.uint8 else mask_or_rgb, 0, 255).astype(np.uint8)
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
    else:
        img = cv2.cvtColor(mask_or_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 40, 40), alpha: float = 0.45) -> np.ndarray:
    vis = image_rgb.copy()
    if mask is None or mask.size == 0:
        return vis
    m = mask
    if m.shape[:2] != vis.shape[:2]:
        m = cv2.resize(m, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_LINEAR)
    binm = m > 0.5
    tint = np.zeros_like(vis)
    tint[binm] = color
    vis[binm] = (vis[binm].astype(np.float32) * (1 - alpha) + tint[binm].astype(np.float32) * alpha).astype(np.uint8)
    return vis


def annotate_full_image(image_rgb: np.ndarray, faces: list) -> np.ndarray:
    vis = image_rgb.copy()
    for f in faces:
        x1, y1, x2, y2 = [int(round(v)) for v in f["bbox"]]
        fake = f["classification"]["label"] == "fake"
        color = (220, 40, 40) if fake else (40, 180, 80)
        if fake and f.get("_mask_full") is not None:
            vis = overlay_mask(vis, f["_mask_full"], color=color)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        p = f["classification"]["fake_probability"]
        text = f"Face {f['face_id']} — {f['classification']['label'].upper()} — {p:.2f}"
        cv2.putText(vis, text, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return vis
