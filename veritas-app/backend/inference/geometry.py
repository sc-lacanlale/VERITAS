"""Same crop / letterbox / mask projection as the training notebook GeometryTransform."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def clip_bbox(x1, y1, x2, y2, width, height):
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, x1 + 1.0), width)
    y2 = min(max(y2, y1 + 1.0), height)
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, width, height, margin: float):
    bw, bh = x2 - x1, y2 - y1
    return clip_bbox(
        x1 - bw * margin,
        y1 - bh * margin,
        x2 + bw * margin,
        y2 + bh * margin,
        width,
        height,
    )


def stable_face_id_order(boxes: Sequence[Tuple[float, float, float, float]]) -> List[int]:
    keyed = [((y1 + y2) * 0.5, (x1 + x2) * 0.5, i) for i, (x1, y1, x2, y2) in enumerate(boxes)]
    keyed.sort()
    return [i for _, _, i in keyed]


class GeometryTransform:
    def __init__(self, size: int, face_margin: float = 0.1):
        self.size = int(size)
        self.face_margin = float(face_margin)

    def crop_window(self, bbox_xyxy, image_w, image_h):
        x1, y1, x2, y2 = bbox_xyxy
        return expand_bbox(x1, y1, x2, y2, image_w, image_h, self.face_margin)

    def letterbox_params(self, crop_w: float, crop_h: float) -> Dict[str, float]:
        scale = min(self.size / max(crop_h, 1e-6), self.size / max(crop_w, 1e-6))
        nw = crop_w * scale
        nh = crop_h * scale
        return {
            "scale": scale,
            "left": (self.size - nw) / 2.0,
            "top": (self.size - nh) / 2.0,
            "nw": nw,
            "nh": nh,
            "crop_w": crop_w,
            "crop_h": crop_h,
        }

    def apply_image(self, image_rgb: np.ndarray, bbox_xyxy):
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = self.crop_window(bbox_xyxy, w, h)
        x1i, y1i, x2i, y2i = [int(round(v)) for v in (x1, y1, x2, y2)]
        x1i, y1i, x2i, y2i = [int(v) for v in clip_bbox(x1i, y1i, x2i, y2i, w, h)]
        crop = image_rgb[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            meta = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **self.letterbox_params(1, 1)}
            return canvas, meta
        ch, cw = crop.shape[:2]
        params = self.letterbox_params(cw, ch)
        nw, nh = max(int(round(params["nw"])), 1), max(int(round(params["nh"])), 1)
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        left = min(max(int(round(params["left"])), 0), self.size - nw)
        top = min(max(int(round(params["top"])), 0), self.size - nh)
        canvas[top : top + nh, left : left + nw] = resized
        meta = {
            "x1": float(x1i),
            "y1": float(y1i),
            "x2": float(x2i),
            "y2": float(y2i),
            "scale": params["scale"],
            "left": float(left),
            "top": float(top),
            "nw": float(nw),
            "nh": float(nh),
            "crop_w": float(cw),
            "crop_h": float(ch),
            "orig_w": float(w),
            "orig_h": float(h),
        }
        return canvas, meta

    def mask_to_original(self, mask_hw: np.ndarray, meta, orig_h: int, orig_w: int) -> np.ndarray:
        full = np.zeros((orig_h, orig_w), dtype=np.float32)
        left, top = int(round(meta["left"])), int(round(meta["top"]))
        nw, nh = int(round(meta["nw"])), int(round(meta["nh"]))
        x1, y1 = int(round(meta["x1"])), int(round(meta["y1"]))
        x2, y2 = int(round(meta["x2"])), int(round(meta["y2"]))
        crop_w, crop_h = max(x2 - x1, 1), max(y2 - y1, 1)
        region = mask_hw[top : top + nh, left : left + nw]
        if region.size == 0:
            return full
        restored = cv2.resize(region.astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        y2e = min(y1 + crop_h, orig_h)
        x2e = min(x1 + crop_w, orig_w)
        full[y1:y2e, x1:x2e] = restored[: y2e - y1, : x2e - x1]
        return full
