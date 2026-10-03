"""Multi-face detector. MediaPipe Tasks first, OpenCV Haar fallback. Stable spatial IDs."""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .geometry import stable_face_id_order

_BLAZE_URLS = [
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
]


def _assign_ids(faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not faces:
        return []
    order = stable_face_id_order([f["bbox"] for f in faces])
    return [{**faces[old], "face_id": i} for i, old in enumerate(order)]


class MediaPipeTasksFaceDetector:
    name = "mediapipe_tasks_blaze_face"

    def __init__(self, model_path: Path, min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=float(min_confidence),
        )
        self._detector = FaceDetector.create_from_options(options)
        self._mp = mp

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        if image_rgb.ndim != 3:
            return []
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._detector.detect(mp_image)
        faces = []
        for det in result.detections or []:
            bb = det.bounding_box
            x1, y1 = float(bb.origin_x), float(bb.origin_y)
            score = float(det.categories[0].score or 0.0) if det.categories else 0.0
            faces.append(
                {
                    "face_id": 0,
                    "bbox": [x1, y1, x1 + float(bb.width), y1 + float(bb.height)],
                    "confidence": score,
                }
            )
        return _assign_ids(faces)


class OpenCVHaarFaceDetector:
    name = "opencv_haar"

    def __init__(self, min_confidence: float = 0.5):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(xml)

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        faces = [
            {"face_id": 0, "bbox": [float(x), float(y), float(x + w), float(y + h)], "confidence": 1.0}
            for x, y, w, h in boxes
        ]
        return _assign_ids(faces)


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.exists() and dest.stat().st_size > 1000
    except Exception:
        return False


def build_face_detector(min_confidence: float, cache_dir: Path):
    model_path = cache_dir / "blaze_face_short_range.tflite"
    try:
        import mediapipe  # noqa: F401

        for url in _BLAZE_URLS:
            if _download(url, model_path):
                try:
                    det = MediaPipeTasksFaceDetector(model_path, min_confidence)
                    print("Face detector:", det.name)
                    return det
                except Exception as e:
                    print("MediaPipe Tasks init failed:", e)
                    break
    except Exception as e:
        print("MediaPipe not available:", e)
    print("Face detector: opencv_haar (fallback)")
    return OpenCVHaarFaceDetector(min_confidence)
