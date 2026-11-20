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
        exp_cfg = (ckpt.get("exp_cfg") or {}) if isinstance(ckpt, dict) else {}
        version = exp_cfg.get("implementation_version")
        strict = False
        if version == "ow_v3":
            # OW v3 Baseline (OWV3_01): RGB-only VERITASV3, built from the checkpoint's own metadata.
            from .architecture_v3 import VERITASV3

            fields = ("multi_stream", "segmentation", "fusion", "color_srm", "frequency", "pooling", "decoder")
            absent = [f for f in fields if f not in exp_cfg]
            if absent:
                raise RuntimeError(f"Incomplete ow_v3 baseline metadata: {absent}")
            if exp_cfg["multi_stream"] or exp_cfg["segmentation"]:
                raise RuntimeError("Checkpoint is not an RGB-only baseline (multi_stream/segmentation is on)")
            model: nn.Module = VERITASV3(pretrained=False, **{f: exp_cfg[f] for f in fields})
            strict = True
        elif version == "ow_v2":
            from .architecture_v2 import VERITASV2

            model = VERITASV2(multi_stream=False, segmentation=False, pretrained=False, fusion=exp_cfg.get("fusion", "rgb"))
            strict = True
        # Older layouts: notebook baseline saves features.* ; a VERITASModel(False, False) run saves encoder.*
        elif any(k.startswith("encoder.") for k in state):
            model = VERITASModel(multi_stream=False, segmentation=False)
        else:
            model = BaselineEffNet()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if strict and unexpected:
            raise RuntimeError(f"Baseline checkpoint has {len(unexpected)} unexpected weights, e.g. {unexpected[:3]}")
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
