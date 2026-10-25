"""FastAPI inference service. Model loads once at startup."""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from config import MODEL_CONFIG
from inference.face_detector import build_face_detector
from inference.geometry import GeometryTransform
from inference.model import load_veritas, model_status
from inference.predictor import VeritasInference

app = FastAPI(title="VERITAS inference", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: dict = {"predictor": None, "detector": None, "ready": False, "error": None}


def _decode_image(data: bytes) -> np.ndarray:
    try:
        pil = Image.open(io.BytesIO(data))
        pil = pil.convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read image") from e
    return np.array(pil)


@app.on_event("startup")
def startup():
    if os.environ.get("VERITAS_SKIP_LOAD") == "1":
        print("Skipping model load (VERITAS_SKIP_LOAD=1)")
        return
    cache = Path(tempfile.gettempdir()) / "veritas-app"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        model, device = load_veritas(
            Path(MODEL_CONFIG["checkpoint"]),
            MODEL_CONFIG["device"],
            MODEL_CONFIG["multi_stream"],
            MODEL_CONFIG["segmentation"],
        )
        geo = GeometryTransform(MODEL_CONFIG["image_size"], MODEL_CONFIG["face_margin"])
        STATE["predictor"] = VeritasInference(
            model,
            device,
            geo,
            MODEL_CONFIG["classification_threshold"],
            MODEL_CONFIG["mask_threshold"],
            emit_masks=MODEL_CONFIG.get("emit_masks", True),
            quality_min_confidence=MODEL_CONFIG.get("quality_min_confidence", 0.0),
            quality_min_face_frac=MODEL_CONFIG.get("quality_min_face_frac", 0.0),
        )
        STATE["detector"] = build_face_detector(MODEL_CONFIG["face_score_threshold"], cache)
        STATE["ready"] = True
        print("Inference service ready")
    except FileNotFoundError as e:
        STATE["error"] = "MODEL_NOT_CONFIGURED"
        print("Checkpoint missing:", e)
    except Exception as e:
        STATE["error"] = "MODEL_NOT_CONFIGURED"
        print("Model load failed:", e)


@app.get("/health")
def health():
    return {
        "ok": STATE["ready"],
        "model": model_status() if STATE["ready"] else STATE.get("error") or "MODEL_NOT_CONFIGURED",
        "detector": getattr(STATE.get("detector"), "name", None),
        "image_size": MODEL_CONFIG["image_size"],
        "checkpoint": Path(MODEL_CONFIG["checkpoint"]).name,
        "checkpoint_kind": MODEL_CONFIG.get("checkpoint_kind"),
        "classification_threshold": MODEL_CONFIG["classification_threshold"],
        "emit_masks": MODEL_CONFIG.get("emit_masks", True),
    }


async def _validate_upload(image: UploadFile) -> bytes:
    """Shared upload validation for both /predict and /predict/stream."""
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=STATE.get("error") or "MODEL_NOT_CONFIGURED")
    name = (image.filename or "").lower()
    suffix = Path(name).suffix
    if suffix not in MODEL_CONFIG["allowed_suffixes"]:
        raise HTTPException(status_code=400, detail=f"File type not allowed: {suffix or 'unknown'}")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MODEL_CONFIG["max_upload_bytes"]:
        raise HTTPException(status_code=400, detail="File too large")
    ctype = (image.content_type or "").split(";")[0].strip().lower()
    if ctype and ctype not in MODEL_CONFIG["allowed_mimes"] and ctype != "application/octet-stream":
        raise HTTPException(status_code=400, detail=f"MIME type not allowed: {ctype}")
    return data


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await _validate_upload(image)
    try:
        rgb = _decode_image(data)
        detected = STATE["detector"].detect(rgb)
        assessment = STATE["predictor"].assess_faces(rgb, detected)
        # Only analyze faces that clear the quality gate.
        result = STATE["predictor"].predict_image(rgb, assessment["faces"])
        result["detector"] = STATE["detector"].name
        result["detection"] = {
            "status": assessment["status"],  # ok | none | low_quality
            "detected": len(detected),
            "analyzable": len(assessment["faces"]),
        }
        return result
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed")


@app.post("/predict/stream")
async def predict_stream(image: UploadFile = File(...)):
    """Server-Sent Events variant of /predict.

    Emits one event per real pipeline stage so the UI can show each step
    going active -> done in real time. The final 'result' event carries the
    exact same payload as /predict. The one-shot /predict is untouched.
    """
    data = await _validate_upload(image)

    def sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    def generate():
        try:
            rgb = _decode_image(data)
        except HTTPException as e:
            yield sse("error", {"detail": e.detail})
            return
        except Exception:
            yield sse("error", {"detail": "Could not read image"})
            return

        try:
            # Report detection as its own live stage.
            yield sse("stage", {"name": "face_detection", "status": "active"})
            detected = STATE["detector"].detect(rgb)
            detector_name = STATE["detector"].name
            assessment = STATE["predictor"].assess_faces(rgb, detected)
            detection_meta = {
                "status": assessment["status"],  # ok | none | low_quality
                "detected": len(detected),
                "analyzable": len(assessment["faces"]),
            }

            for stage, payload in STATE["predictor"].predict_image_streaming(rgb, assessment["faces"]):
                if stage == "face_detection":
                    # Carry the real quality verdict on the detection stage.
                    payload = {**payload, **detection_meta}
                    yield sse("stage", {"name": stage, "status": "done", "data": payload})
                elif stage == "result":
                    payload["detector"] = detector_name
                    payload["detection"] = detection_meta
                    yield sse("result", payload)
                else:
                    yield sse("stage", {"name": stage, "status": "done", "data": payload})
        except Exception:
            yield sse("error", {"detail": "Inference failed"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
