"""Baseline RGB EfficientNet-B7 (classification only) for side-by-side checks in the app.

This mirrors the BaselineEffNet in the training notebooks (features.* + cls_head.*),
so a baseline checkpoint from OW_01 / OF_01 loads as-is. It has no segmentation
head, so the app shows a score and verdict but no localization mask.
The Full VERITAS loader in model.py is not touched.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7

from .architecture import ClassificationHead, VERITASModel
from .model import _strip_module, _torch_load, materialize_checkpoint


class BaselineEffNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = efficientnet_b7(weights=None)
        self.features = net.features
        self.cls_head = ClassificationHead(2560)
        self.multi_stream = False
        self.segmentation = False

    def forward(self, x):
        logits = self.cls_head(self.features(x)).squeeze(-1)
        return {"classification_logits": logits, "fake_probability": torch.sigmoid(logits)}


_LOCK = threading.Lock()
_MODEL: Optional[nn.Module] = None
_DEVICE: Optional[torch.device] = None
_LOADED_FROM: Optional[Path] = None


def baseline_loaded() -> bool:
    return _MODEL is not None


def load_baseline(checkpoint_path: Path, device_pref: str) -> Tuple[nn.Module, torch.device]:
    """Load once (thread-safe). Raises FileNotFoundError / RuntimeError on a bad checkpoint."""
    global _MODEL, _DEVICE, _LOADED_FROM
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _DEVICE
        if device_pref == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        ckpt_file = materialize_checkpoint(Path(checkpoint_path))
        print("Loading baseline checkpoint:", ckpt_file)
        ckpt = _torch_load(ckpt_file)
        state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
        if state is None:
            raise RuntimeError("Baseline checkpoint does not contain model_state_dict")
        state = _strip_module(state)
        # Notebook baseline layout is features.* ; a VERITASModel(False, False) run saves encoder.*
        if any(k.startswith("encoder.") for k in state):
            model: nn.Module = VERITASModel(multi_stream=False, segmentation=False)
        else:
            model = BaselineEffNet()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Baseline checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
        if unexpected:
            print("Baseline unexpected keys:", unexpected[:8])
        model.to(device)
        model.eval()
        _MODEL, _DEVICE, _LOADED_FROM = model, device, Path(checkpoint_path)
        print(f"Baseline loaded (epoch={ckpt.get('epoch')}, best={ckpt.get('best_metric')})")
        del ckpt
        return _MODEL, _DEVICE
