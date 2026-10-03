"""Project a model-space face mask back onto the original image."""
from __future__ import annotations

import numpy as np

from .geometry import GeometryTransform


def project_mask_to_original_image(face_mask: np.ndarray, meta: dict, orig_h: int, orig_w: int, geo: GeometryTransform):
    return geo.mask_to_original(face_mask, meta, orig_h, orig_w)
