"""Face-level VERITAS inference. Never collapses an image to a single label."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import torch

from .geometry import GeometryTransform
from .mask_projection import project_mask_to_original_image
from .preprocessing import prepare_face
from .visualization import annotate_full_image, encode_jpeg, encode_png, overlay_mask


class VeritasInference:
    def __init__(
        self,
        model,
        device,
        geo: GeometryTransform,
        threshold: float,
        mask_threshold: float,
        emit_masks: bool = True,
        quality_min_confidence: float = 0.0,
        quality_min_face_frac: float = 0.0,
    ):
        self.model = model
        self.device = device
        self.geo = geo
        self.threshold = float(threshold)
        self.mask_threshold = float(mask_threshold)
        self.emit_masks = bool(emit_masks)
        self.quality_min_confidence = float(quality_min_confidence)
        self.quality_min_face_frac = float(quality_min_face_frac)

    def assess_faces(self, image_rgb: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Report a status for the detected faces WITHOUT dropping any of them.

        Returns {"status": "ok"|"none"|"low_quality", "faces": [...all detected]}.
        Every face the detector found is analyzed (original behavior). The quality
        signals (confidence + size) only decide the status MESSAGE:
          - "none": the detector returned no faces at all.
          - "low_quality": faces exist but none clear the quality bar (shown as
            "face not clear enough"); the faces are still analyzed.
          - "ok": at least one face clears the quality bar.
        We do NOT claim pose or occlusion detection the detector can't provide.
        """
        if not faces:
            return {"status": "none", "faces": []}
        h, w = image_rgb.shape[:2]
        img_side = float(max(h, w)) or 1.0
        any_good = False
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            face_side = max(float(x2 - x1), float(y2 - y1))
            frac = face_side / img_side
            conf = float(f.get("confidence", 0.0))
            if conf >= self.quality_min_confidence and frac >= self.quality_min_face_frac:
                any_good = True
                break
        # Analyze all detected faces regardless; status is informational only.
        return {"status": "ok" if any_good else "low_quality", "faces": list(faces)}

    # ------------------------------------------------------------------ #
    # Stage helpers (shared by the one-shot and streaming code paths).    #
    # Each returns real telemetry so nothing has to be guessed by the UI. #
    # ------------------------------------------------------------------ #

    def _empty_payload(self, image_rgb: np.ndarray, faces_detected: int) -> Dict[str, Any]:
        h, w = image_rgb.shape[:2]
        return {
            "image": {"width": w, "height": h},
            "faces": [],
            "detector": None,
            "stages": {
                "face_detection": {"faces_detected": faces_detected},
                "preprocessing": {"faces_prepared": 0, "ms": 0.0},
                "inference": {"faces_scored": 0, "ms": 0.0},
                "explainability": {"maps_generated": 0, "available": False},
            },
        }

    def _preprocess(self, image_rgb, faces):
        t0 = time.perf_counter()
        tensors, canvases, metas = [], [], []
        for face in faces:
            tensor, canvas, meta = prepare_face(image_rgb, face["bbox"], self.geo)
            tensors.append(tensor)
            canvases.append(canvas)
            metas.append(meta)
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return tensors, canvases, metas, ms

    @torch.no_grad()
    def _run_model(self, tensors):
        t1 = time.perf_counter()
        batch = torch.stack(tensors, 0).to(self.device)
        self.model.eval()
        outputs = self.model(batch)
        probs = outputs["fake_probability"].detach().cpu().numpy().reshape(-1)
        seg = outputs.get("segmentation_prob")
        if seg is not None:
            seg = seg.detach().cpu().numpy()
        ms = round((time.perf_counter() - t1) * 1000.0, 1)
        return probs, seg, ms

    def _build_rows(self, image_rgb, faces, canvases, metas, probs, seg):
        h, w = image_rgb.shape[:2]
        maps_generated = 0
        rows = []
        for i, face in enumerate(faces):
            p = float(probs[i])
            label = "fake" if p >= self.threshold else "real"
            crop_mask_png = None
            full_mask_png = None
            crop_binary_png = None
            mask_full = None
            mask_coverage = 0.0
            if self.emit_masks and seg is not None:
                face_prob = seg[i, 0]
                face_bin = (face_prob >= self.mask_threshold).astype(np.float32)
                crop_overlay = overlay_mask(canvases[i], face_bin)
                crop_mask_png = encode_png(crop_overlay) if label == "fake" else None
                if label == "fake":
                    # Real explainability straight from the model's segmentation mask.
                    crop_binary_png = encode_png((face_bin * 255).astype(np.uint8))
                    mask_coverage = float(face_bin.mean())  # fraction of crop flagged
                    mask_full = project_mask_to_original_image(face_bin, metas[i], h, w, self.geo)
                    full_mask_png = encode_png((mask_full >= self.mask_threshold).astype(np.uint8) * 255)
                    maps_generated += 1
            rows.append({
                "face_id": int(face["face_id"]),
                "bbox": [float(v) for v in face["bbox"]],
                "detector_confidence": float(face.get("confidence", 0.0)),
                "classification": {"label": label, "fake_probability": p},
                "segmentation": {
                    "available": bool(self.emit_masks and seg is not None),
                    "mask": full_mask_png,
                    "crop_mask": crop_mask_png,
                    "crop_binary": crop_binary_png,
                    "coverage": round(mask_coverage, 4),
                },
                "crop": encode_jpeg(canvases[i]),
                "_mask_full": mask_full,
            })
        return rows, maps_generated

    def predict_image(self, image_rgb: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        h, w = image_rgb.shape[:2]
        payload = self._empty_payload(image_rgb, len(faces))
        stages = payload["stages"]
        if not faces:
            payload["overlay_image"] = encode_jpeg(image_rgb)
            return payload

        tensors, canvases, metas, prep_ms = self._preprocess(image_rgb, faces)
        stages["preprocessing"]["faces_prepared"] = len(tensors)
        stages["preprocessing"]["ms"] = prep_ms

        probs, seg, infer_ms = self._run_model(tensors)
        stages["inference"]["faces_scored"] = int(probs.shape[0])
        stages["inference"]["ms"] = infer_ms
        stages["explainability"]["available"] = bool(self.emit_masks and seg is not None)

        rows, maps_generated = self._build_rows(image_rgb, faces, canvases, metas, probs, seg)
        stages["explainability"]["maps_generated"] = maps_generated

        overlay = annotate_full_image(image_rgb, rows)
        for row in rows:
            row.pop("_mask_full", None)
        payload["faces"] = rows
        payload["overlay_image"] = encode_jpeg(overlay)
        return payload

    def predict_image_streaming(self, image_rgb: np.ndarray, faces: List[Dict[str, Any]]):
        """Yield (stage, data) tuples as each real stage completes.

        Stages, in order: face_detection, preprocessing, inference,
        explainability, and finally 'result' carrying the full payload.
        Mirrors predict_image exactly — same work, just reported live.
        """
        h, w = image_rgb.shape[:2]
        payload = self._empty_payload(image_rgb, len(faces))
        stages = payload["stages"]

        # Stage 1: face detection already happened upstream; report the count.
        yield "face_detection", {"faces_detected": len(faces)}

        if not faces:
            payload["overlay_image"] = encode_jpeg(image_rgb)
            yield "result", payload
            return

        # Stage 2: preprocessing.
        tensors, canvases, metas, prep_ms = self._preprocess(image_rgb, faces)
        stages["preprocessing"]["faces_prepared"] = len(tensors)
        stages["preprocessing"]["ms"] = prep_ms
        yield "preprocessing", dict(stages["preprocessing"])

        # Stage 3: model inference.
        probs, seg, infer_ms = self._run_model(tensors)
        stages["inference"]["faces_scored"] = int(probs.shape[0])
        stages["inference"]["ms"] = infer_ms
        stages["explainability"]["available"] = bool(self.emit_masks and seg is not None)
        yield "inference", dict(stages["inference"])

        # Stage 4: explainability maps.
        rows, maps_generated = self._build_rows(image_rgb, faces, canvases, metas, probs, seg)
        stages["explainability"]["maps_generated"] = maps_generated
        yield "explainability", dict(stages["explainability"])

        overlay = annotate_full_image(image_rgb, rows)
        for row in rows:
            row.pop("_mask_full", None)
        payload["faces"] = rows
        payload["overlay_image"] = encode_jpeg(overlay)
        yield "result", payload
