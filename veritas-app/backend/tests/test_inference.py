"""Backend tests from appAgent.md: faces, masks, invalid input, API schema."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["VERITAS_SKIP_LOAD"] = "1"

from inference.geometry import GeometryTransform, stable_face_id_order
from inference.mask_projection import project_mask_to_original_image
from inference.predictor import VeritasInference
from inference.visualization import encode_png


class FakeModel:
    segmentation = True

    def eval(self):
        return self

    def __call__(self, batch):
        import torch

        n = batch.size(0)
        # Face 0 real, remaining fake — used for mixed-face tests.
        probs = torch.zeros(n)
        if n >= 2:
            probs[1:] = 0.94
        elif n == 1:
            probs[0] = 0.08
        seg = torch.zeros(n, 1, batch.shape[-2], batch.shape[-1])
        for i in range(n):
            if float(probs[i]) >= 0.5:
                seg[i, 0, 80:200, 80:200] = 1.0
        return {"fake_probability": probs, "segmentation_prob": seg}


class ScriptedDetector:
    def __init__(self, faces):
        self.faces = faces
        self.name = "scripted"

    def detect(self, image_rgb):
        return list(self.faces)


def _rgb(w=240, h=180, color=(40, 80, 120)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _infer(faces, image=None):
    geo = GeometryTransform(320, 0.1)
    pred = VeritasInference(FakeModel(), "cpu", geo, 0.5, 0.5)
    return pred.predict_image(image if image is not None else _rgb(), faces)


def test_one_face():
    out = _infer([{"face_id": 0, "bbox": [20, 20, 100, 120], "confidence": 0.99}])
    assert len(out["faces"]) == 1
    assert out["faces"][0]["face_id"] == 0
    assert out["faces"][0]["classification"]["label"] == "real"


def test_multiple_faces():
    out = _infer(
        [
            {"face_id": 0, "bbox": [10, 10, 80, 90], "confidence": 0.9},
            {"face_id": 1, "bbox": [120, 20, 200, 110], "confidence": 0.9},
            {"face_id": 2, "bbox": [40, 100, 110, 170], "confidence": 0.8},
        ]
    )
    assert [f["face_id"] for f in out["faces"]] == [0, 1, 2]
    assert len({tuple(f["bbox"]) for f in out["faces"]}) == 3


def test_mixed_real_fake_faces():
    out = _infer(
        [
            {"face_id": 0, "bbox": [10, 10, 80, 90], "confidence": 0.9},
            {"face_id": 1, "bbox": [120, 20, 200, 110], "confidence": 0.95},
        ]
    )
    assert out["faces"][0]["classification"]["label"] == "real"
    assert out["faces"][1]["classification"]["label"] == "fake"
    assert out["faces"][0]["segmentation"]["mask"] is None
    assert out["faces"][1]["segmentation"]["mask"]
    assert out["faces"][1]["segmentation"]["available"] is True


def test_no_faces():
    out = _infer([])
    assert out["faces"] == []
    assert out["image"]["width"] == 240


def test_mask_projection():
    geo = GeometryTransform(100, 0.1)
    image = _rgb(200, 160)
    bbox = [40, 30, 120, 110]
    canvas, meta = geo.apply_image(image, bbox)
    mask = np.zeros((100, 100), dtype=np.float32)
    left, top = int(meta["left"]), int(meta["top"])
    nw, nh = int(meta["nw"]), int(meta["nh"])
    mask[top : top + nh, left : left + nw] = 1.0
    full = project_mask_to_original_image(mask, meta, 160, 200, geo)
    x1, y1, x2, y2 = [int(meta[k]) for k in ("x1", "y1", "x2", "y2")]
    assert full[0, 0] == 0
    assert full[y1 + 2, x1 + 2] > 0.5
    assert full.shape == (160, 200)
    # Must not be a naive resize of the 100x100 mask onto the whole image.
    assert full.mean() < 0.5


def test_stable_face_ids_are_spatial():
    boxes = [(200, 10, 250, 60), (10, 10, 50, 50)]
    order = stable_face_id_order(boxes)
    assert order[0] == 1


def test_invalid_image_api(monkeypatch):
    import app as appmod

    appmod.STATE["ready"] = True
    appmod.STATE["detector"] = ScriptedDetector([])
    appmod.STATE["predictor"] = VeritasInference(
        FakeModel(), "cpu", GeometryTransform(64, 0.1), 0.5, 0.5
    )
    client = TestClient(appmod.app)
    r = client.post("/predict", files={"image": ("x.txt", b"not-an-image", "text/plain")})
    assert r.status_code == 400


def test_api_response_schema(monkeypatch):
    import app as appmod

    appmod.STATE["ready"] = True
    appmod.STATE["detector"] = ScriptedDetector(
        [
            {"face_id": 0, "bbox": [10, 10, 80, 90], "confidence": 0.91},
            {"face_id": 1, "bbox": [100, 20, 180, 110], "confidence": 0.88},
        ]
    )
    appmod.STATE["predictor"] = VeritasInference(
        FakeModel(), "cpu", GeometryTransform(64, 0.1), 0.5, 0.5
    )
    client = TestClient(appmod.app)
    buf = io.BytesIO()
    Image.fromarray(_rgb()).save(buf, format="JPEG")
    r = client.post("/predict", files={"image": ("t.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert set(body["image"]) == {"width", "height"}
    assert len(body["faces"]) == 2
    f0 = body["faces"][0]
    assert set(f0) >= {"face_id", "bbox", "detector_confidence", "classification", "segmentation"}
    assert set(f0["classification"]) == {"label", "fake_probability"}
    assert f0["classification"]["label"] in {"real", "fake"}
    assert isinstance(f0["segmentation"]["available"], bool)
    encode_png(np.zeros((8, 8), dtype=np.uint8))
