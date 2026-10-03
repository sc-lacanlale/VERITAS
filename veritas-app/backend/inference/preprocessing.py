"""ImageNet-normalized tensors — identical to FaceCropDataset.__getitem__ (no augment)."""
from __future__ import annotations

import numpy as np
import torch

from .geometry import GeometryTransform

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_to_tensor(canvas_rgb: np.ndarray) -> torch.Tensor:
    x = canvas_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1)).copy())


def prepare_face(image_rgb: np.ndarray, bbox, geo: GeometryTransform):
    canvas, meta = geo.apply_image(image_rgb, bbox)
    return letterbox_to_tensor(canvas), canvas, meta
