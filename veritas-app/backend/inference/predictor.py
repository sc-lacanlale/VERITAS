"""Face-level VERITAS inference. Never collapses an image to a single label."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .geometry import GeometryTransform
from .mask_projection import project_mask_to_original_image
from .preprocessing import prepare_face
from .visualization import annotate_full_image, encode_jpeg, encode_png, overlay_mask


class VeritasInference:
    def __init__(self, model, device, geo: GeometryTransform, threshold: float, mask_threshold: float):
        self.model = model
        self.device = device
        self.geo = geo
        self.threshold = float(threshold)
        self.mask_threshold = float(mask_threshold)

    @torch.no_grad()
    def predict_image(self, image_rgb: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        h, w = image_rgb.shape[:2]
        payload = {"image": {"width": w, "height": h}, "faces": [], "detector": None}
        if not faces:
            payload["overlay_image"] = encode_jpeg(image_rgb)
            return payload

        tensors, canvases, metas = [], [], []
        for face in faces:
            tensor, canvas, meta = prepare_face(image_rgb, face["bbox"], self.geo)
            tensors.append(tensor)
            canvases.append(canvas)
            metas.append(meta)

        batch = torch.stack(tensors, 0).to(self.device)
        self.model.eval()
        outputs = self.model(batch)
        probs = outputs["fake_probability"].detach().cpu().numpy().reshape(-1)
        seg = outputs.get("segmentation_prob")
        if seg is not None:
            seg = seg.detach().cpu().numpy()

        rows = []
        for i, face in enumerate(faces):
            p = float(probs[i])
            label = "fake" if p >= self.threshold else "real"
            crop_mask_png = None
            full_mask_png = None
            mask_full = None
            if seg is not None:
                face_prob = seg[i, 0]
                face_bin = (face_prob >= self.mask_threshold).astype(np.float32)
                crop_overlay = overlay_mask(canvases[i], face_bin)
                crop_mask_png = encode_png(crop_overlay) if label == "fake" else None
                if label == "fake":
                    mask_full = project_mask_to_original_image(face_bin, metas[i], h, w, self.geo)
                    full_mask_png = encode_png((mask_full >= self.mask_threshold).astype(np.uint8) * 255)
            row = {
                "face_id": int(face["face_id"]),
                "bbox": [float(v) for v in face["bbox"]],
                "detector_confidence": float(face.get("confidence", 0.0)),
                "classification": {"label": label, "fake_probability": p},
                "segmentation": {
                    "available": seg is not None,
                    "mask": full_mask_png,
                    "crop_mask": crop_mask_png,
                },
                "crop": encode_jpeg(canvases[i]),
                "_mask_full": mask_full,
            }
            rows.append(row)

        overlay = annotate_full_image(image_rgb, rows)
        for row in rows:
            row.pop("_mask_full", None)
        payload["faces"] = rows
        payload["overlay_image"] = encode_jpeg(overlay)
        return payload
