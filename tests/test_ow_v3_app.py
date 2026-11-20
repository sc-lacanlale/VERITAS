"""Checkpoint dispatch and exact letterbox geometry for v3 app inference."""
import importlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "veritas-app/backend"))
from inference.geometry import GeometryTransform
from inference.predictor import VeritasInference


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def _v3_config():
    return dict(implementation_version="ow_v3", multi_stream=True, segmentation=True,
                fusion="late_nonlinear", color_srm=True, frequency="local", pooling="avg_max", decoder="separable")


def _load_checkpoint(checkpoint, constructor=None):
    loader = importlib.import_module("inference.model")
    with patch.object(loader, "_MODEL", None), patch.object(loader, "_DEVICE", None), \
         patch.object(loader, "_STATUS", "MODEL_NOT_CONFIGURED"), \
         patch.object(loader, "materialize_checkpoint", return_value=Path("test.pt")), \
         patch.object(loader, "_torch_load", return_value=checkpoint), \
         patch("inference.architecture_v3.VERITASV3", return_value=TinyModel()) as v3:
        result = loader.load_veritas(Path("test.pt"), "cpu", False, False)
        if constructor is not None:
            constructor(v3)
        return result


def test_v3_checkpoint_dispatch_uses_recorded_architecture_and_no_pretraining():
    config = _v3_config()
    checkpoint = dict(exp_cfg=config, model_state_dict={"weight": torch.ones(1)})

    def assert_call(constructor):
        constructor.assert_called_once_with(pretrained=False, **{key: value for key, value in config.items()
                                                                 if key != "implementation_version"})

    model, device = _load_checkpoint(checkpoint, assert_call)
    assert model.weight.item() == 1 and not model.training and device.type == "cpu"


def test_v3_strict_loading_rejects_missing_weights():
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_checkpoint(dict(exp_cfg=_v3_config(), model_state_dict={}))


def test_v3_strict_loading_rejects_unexpected_weights():
    with pytest.raises(RuntimeError, match="Unexpected key"):
        _load_checkpoint(dict(exp_cfg=_v3_config(), model_state_dict={"weight": torch.ones(1), "unknown": torch.ones(1)}))


def test_unknown_checkpoint_implementation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported checkpoint implementation"):
        _load_checkpoint(dict(exp_cfg={"implementation_version": "ow_v99"}, model_state_dict={}))


def test_v3_incomplete_architecture_metadata_is_rejected():
    config = _v3_config()
    del config["frequency"]
    with pytest.raises(ValueError, match="architecture metadata"):
        _load_checkpoint(dict(exp_cfg=config, model_state_dict={"weight": torch.ones(1)}))


class CanvasAwareModel:
    requires_valid_mask = True
    segmentation = False

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, batch, valid_mask):
        self.calls.append((batch.clone(), valid_mask.clone()))
        return dict(fake_probability=torch.full((len(batch), 1), 0.1, device=batch.device))


def _predictor(model=None, size=9):
    return VeritasInference(model or CanvasAwareModel(), "cpu", GeometryTransform(size, 0), 0.5, 0.5)


def test_valid_canvas_mask_uses_explicit_odd_padding_and_preserves_black_content():
    model = CanvasAwareModel()
    pred = _predictor(model)
    tensors = [torch.zeros(3, 9, 9), torch.zeros(3, 9, 9)]
    metas = [dict(left=2., top=1., nw=4., nh=7.), dict(left=0., top=3., nw=9., nh=3.)]
    probabilities, segmentation, _ = pred._run_model(tensors, metas)
    mask = model.calls[0][1]
    expected = torch.zeros(2, 1, 9, 9)
    expected[0, :, 1:8, 2:6] = 1
    expected[1, :, 3:6, :] = 1
    torch.testing.assert_close(mask, expected)
    assert list(probabilities.shape) == [2] and segmentation is None
    assert mask.sum().item() == 55


@pytest.mark.parametrize("metas", [None, [], [{}], [dict(left=-1, top=0, nw=4, nh=7)],
                                  [dict(left=2, top=1, nw=9, nh=7)],
                                  [dict(left=1.5, top=0, nw=4, nh=7)],
                                  [dict(left=float("nan"), top=0, nw=4, nh=7)]])
def test_missing_or_invalid_letterbox_metadata_raises(metas):
    with pytest.raises(ValueError, match="[Ll]etterbox"):
        _predictor()._run_model([torch.zeros(3, 9, 9)], metas)


def test_legacy_models_receive_no_new_keyword_argument():
    class LegacyModel:
        def eval(self):
            return self

        def __call__(self, batch):
            return dict(fake_probability=torch.zeros(len(batch)))

    probabilities, segmentation, _ = _predictor(LegacyModel())._run_model([torch.zeros(3, 9, 9)])
    assert probabilities.tolist() == [0] and segmentation is None


@pytest.mark.parametrize("streaming", [False, True])
def test_one_shot_and_streaming_predictions_forward_real_geometry(streaming):
    model = CanvasAwareModel()
    predictor = _predictor(model, size=7)
    image = np.zeros((5, 4, 3), dtype=np.uint8)
    faces = [dict(face_id=0, bbox=[0, 0, 4, 5], confidence=0.9)]
    if streaming:
        events = list(predictor.predict_image_streaming(image, faces))
        assert [name for name, _ in events] == ["face_detection", "preprocessing", "inference", "explainability", "result"]
        payload = events[-1][1]
    else:
        payload = predictor.predict_image(image, faces)
    _, meta = predictor.geo.apply_image(image, faces[0]["bbox"])
    expected = torch.zeros(1, 1, 7, 7)
    top, left, nh, nw = (int(meta[key]) for key in ("top", "left", "nh", "nw"))
    expected[:, :, top:top + nh, left:left + nw] = 1
    torch.testing.assert_close(model.calls[0][1], expected)
    assert len(payload["faces"]) == 1 and payload["faces"][0]["classification"]["label"] == "real"
