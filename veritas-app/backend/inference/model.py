"""Load the trained Full VERITAS checkpoint once and keep it in memory."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import torch

from .architecture import VERITASModel

_MODEL: Optional[VERITASModel] = None
_DEVICE: Optional[torch.device] = None
_STATUS = "MODEL_NOT_CONFIGURED"


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def materialize_checkpoint(path: Path) -> Path:
    """Accept a real .pt, a Torch zip, or a downloaded zip whose members are prefixed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        _torch_load(path)
        return path
    except Exception:
        pass
    if path.suffix.lower() != ".zip":
        raise RuntimeError(f"Cannot read checkpoint: {path}")
    cache = path.with_name(path.stem + "_repacked.pt")
    if cache.exists() and cache.stat().st_size > 1_000_000:
        return cache
    with zipfile.ZipFile(path) as zin:
        names = zin.namelist()
        pkl = next((n for n in names if n.endswith("data.pkl")), None)
        if pkl is None:
            raise RuntimeError("Zip is not a PyTorch checkpoint (no data.pkl)")
        prefix = pkl[: -len("data.pkl")]
        cache.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(cache, "w", compression=zipfile.ZIP_STORED) as zout:
            for name in names:
                if prefix and not name.startswith(prefix):
                    continue
                inner = name[len(prefix) :]
                if not inner or inner.endswith("/"):
                    continue
                zout.writestr(inner, zin.read(name))
    return cache


def _strip_module(state: dict) -> dict:
    if not state:
        return state
    if all(k.startswith("module.") for k in state):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def load_veritas(checkpoint_path: Path, device_pref: str, multi_stream: bool, segmentation: bool):
    global _MODEL, _DEVICE, _STATUS
    if _MODEL is not None:
        return _MODEL, _DEVICE
    if device_pref == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("VERITAS device:", device)
    ckpt_file = materialize_checkpoint(Path(checkpoint_path))
    print("Loading checkpoint:", ckpt_file)
    ckpt = _torch_load(ckpt_file)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint does not contain model_state_dict")
    model = VERITASModel(multi_stream=multi_stream, segmentation=segmentation)
    missing, unexpected = model.load_state_dict(_strip_module(ckpt["model_state_dict"]), strict=False)
    if missing:
        print("Missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("Unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    model.to(device)
    model.eval()
    _MODEL, _DEVICE, _STATUS = model, device, "ready"
    epoch = ckpt.get("epoch")
    print(f"VERITAS loaded (epoch={epoch}, best={ckpt.get('best_metric')})")
    del ckpt
    return _MODEL, _DEVICE


def model_status() -> str:
    return _STATUS


def get_model():
    if _MODEL is None:
        raise RuntimeError("MODEL_NOT_CONFIGURED")
    return _MODEL, _DEVICE
