"""Behavioral contracts for the new study; no dataset or pretrained download."""
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import nbformat
import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "veritas-app/backend"))
from inference.architecture_v2 import ForensicStreams, VERITASV2, expand_rgb_stem
from ow_v2_training import VERITASLossV2, validate_resume_checkpoint
import make_ow_v2_notebooks as generator

torch.set_num_threads(2)


def test_expanded_stem_preserves_rgb_and_learns_auxiliary_channels():
    torch.manual_seed(42)
    rgb_stem = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    stem = expand_rgb_stem(rgb_stem)
    rgb, aux = torch.randn(2, 3, 16, 16), torch.randn(2, 6, 16, 16)
    result = stem(torch.cat([rgb, aux], dim=1))
    torch.testing.assert_close(result, rgb_stem(rgb))
    result.square().mean().backward()
    assert stem.weight.grad[:, 3:].abs().sum() > 0


@pytest.mark.parametrize("black", [False, True])
def test_streams_are_finite_at_600px_under_autocast(black):
    x = torch.randn(1, 3, 600, 600)
    if black:
        x = -torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = x.expand(1, 3, 600, 600).clone()
    before = x.clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        srm, fft = ForensicStreams()(x)
    assert srm.shape == fft.shape == x.shape
    assert srm.dtype == fft.dtype == torch.float32
    assert torch.isfinite(srm).all() and torch.isfinite(fft).all()
    assert srm.abs().max() <= 1 and fft.abs().max() <= 1.00001
    torch.testing.assert_close(x, before)


def test_real_b7_initial_logits_match_rgb_for_both_fusion_variants():
    x = torch.randn(1, 3, 64, 64)
    torch.manual_seed(7)
    baseline = VERITASV2(False, False, pretrained=False, fusion="rgb").eval()
    with torch.no_grad():
        expected = baseline(x)["classification_logits"]
    baseline_count = sum(p.numel() for p in baseline.parameters())
    del baseline
    for fusion in ("early", "late"):
        torch.manual_seed(7)
        model = VERITASV2(True, False, pretrained=False, fusion=fusion).eval()
        with torch.no_grad():
            actual = model(x)["classification_logits"]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        if fusion == "late":
            assert sum(p.numel() for p in model.parameters()) - baseline_count < 1_000_000
        del model


@pytest.mark.parametrize("fusion", ["early", "late"])
def test_real_b7_segmentation_forward_at_actual_600px(fusion):
    model = VERITASV2(True, True, pretrained=False, fusion=fusion).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 600, 600))
    assert outputs["segmentation_logits"].shape == (1, 1, 600, 600)
    assert all(torch.isfinite(x).all() for x in outputs.values())


@pytest.mark.parametrize("fusion", ["early", "late", "rgb"])
def test_classification_and_segmentation_backward(fusion):
    model = VERITASV2(fusion != "rgb", True, pretrained=False, fusion=fusion).train()
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)
    assert outputs["classification_logits"].shape == (2, 1)
    assert outputs["segmentation_logits"].shape == (2, 1, 64, 64)
    criterion = VERITASLossV2(segmentation=True)
    losses = criterion(outputs, torch.tensor([[0.], [1.]]), torch.zeros(2, 1, 64, 64))
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total"].backward()
    assert model.encoder[0][0].weight.grad.abs().sum() > 0
    assert model.seg_head.decoder[-1].weight.grad.abs().sum() > 0
    if fusion == "late":
        assert model.aux_projection.weight.grad.abs().sum() > 0
        # A second step lets the initially zero projection pass gradients upstream.
        with torch.no_grad():
            model.aux_projection.weight.add_(-0.01 * model.aux_projection.weight.grad)
        model.zero_grad(set_to_none=True)
        criterion(model(x), torch.tensor([[0.], [1.]]), torch.zeros(2, 1, 64, 64))["total"].backward()
        assert model.fft_encoder[0].weight.grad.abs().sum() > 0
        assert model.srm_encoder[0].weight.grad.abs().sum() > 0


def test_auxiliary_loss_is_averaged_and_classification_is_unchanged():
    outputs = dict(classification_logits=torch.tensor([[0.4], [-0.3]], requires_grad=True),
                   segmentation_logits=torch.zeros(2, 1, 8, 8, requires_grad=True))
    labels, masks = torch.tensor([[1.], [0.]]), torch.zeros(2, 1, 8, 8)
    loss = VERITASLossV2(lambda_seg=0.25, segmentation=True)(outputs, labels, masks)
    expected = nn.functional.binary_cross_entropy_with_logits(outputs["classification_logits"], labels)
    torch.testing.assert_close(loss["classification"], expected)
    torch.testing.assert_close(loss["segmentation"], (loss["segmentation_bce"] + loss["segmentation_dice"]) / 2)
    torch.testing.assert_close(loss["total"], expected + 0.25 * loss["segmentation"])
    with pytest.raises(ValueError):
        VERITASLossV2(segmentation=True)(outputs, labels)


def test_dice_reductions_do_not_overflow_at_600px_float16():
    logits = torch.zeros(1, 1, 600, 600, dtype=torch.float16, requires_grad=True)
    outputs = dict(classification_logits=torch.zeros(1, 1, requires_grad=True), segmentation_logits=logits)
    losses = VERITASLossV2(segmentation=True)(outputs, torch.ones(1, 1), torch.ones_like(logits))
    torch.testing.assert_close(losses["segmentation_dice"], torch.tensor(1 / 3))
    losses["total"].backward()
    assert torch.isfinite(logits.grad).all()


def test_resume_rejects_legacy_smoke_and_other_architectures():
    cfg = dict(implementation_version="ow_v2", fusion="early")
    ck = dict(experiment_name="full_veritas_ow_v2", exp_cfg=cfg, protocol_hash="full-protocol")
    validate_resume_checkpoint(ck, ck["experiment_name"], cfg, "full-protocol")
    for change in (dict(protocol_hash="smoke-protocol"), dict(exp_cfg={}), dict(experiment_name="legacy")):
        with pytest.raises(ValueError):
            validate_resume_checkpoint({**ck, **change}, ck["experiment_name"], cfg, "full-protocol")


def test_best_checkpoint_evaluation_loads_when_training_resume_is_disabled(tmp_path):
    variant, fusion = generator.variants()[0]
    nb = generator.build_training_nb(variant, fusion)
    cfg = dict(implementation_version="ow_v2", fusion="rgb")
    env = dict(Path=Path, torch=torch, CONFIG={"experiment": variant["exp"], "resume": False},
               EXP_CFG=cfg, PROTOCOL_HASH="matched", validate_resume_checkpoint=validate_resume_checkpoint)
    exec(generator.cell_starting(nb, 'EXP = CONFIG["experiment"]').source, env)
    checkpoint = tmp_path / "best.pt"
    torch.save(dict(experiment_name=variant["exp"], exp_cfg=cfg, protocol_hash="matched",
                    model_state_dict={"weight": torch.ones(1)}, epoch=1, best_metric=0.9), checkpoint)
    model = FakeModel()
    assert model.weight.item() == 0
    epoch, best, _ = env["load_if_any"](checkpoint, model)
    assert epoch == 1 and best == 0.9 and model.weight.item() == 1
    training = generator.cell_starting(nb, "def mask_metrics_from_tensors").source
    assert 'if CONFIG.get("resume", True):\n        start_epoch, best_metric, resumed = load_if_any' in training


def test_all_generated_notebooks_validate_and_embed_the_current_model():
    params = []
    for variant, fusion in generator.variants():
        nb = generator.build_training_nb(variant, fusion)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                compile(cell.source, variant["file"], "exec")
        config = generator.cell_starting(nb, "CONFIG = {").source
        assert '"test_eval_when": "final"' in config
        assert '"lambda_seg": 0.25' in config
        assert '"implementation_version": "ow_v2"' in config
        assert f'"fusion": "{fusion}"' in config
        training = generator.cell_starting(nb, "def mask_metrics_from_tensors").source
        assert 'score = float(val_metrics["accuracy"])' in training
        assert '"split": "test"' in training
        model = next(c.source for c in nb.cells if c.cell_type == "code" and "class VERITASV2(" in c.source)
        assert generator.model_source() == model
        assert "def unpack_mask_outputs(" in generator.cell_starting(nb, "EXTS = {").source
        params.append(variant["exp"])
    assert len(params) == len(set(params)) == 5


class FakeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


@pytest.mark.parametrize("v2", [False, True])
def test_app_dispatches_legacy_and_v2_checkpoints_without_changing_legacy(v2):
    loader = importlib.import_module("inference.model")
    cfg = dict(implementation_version="ow_v2", fusion="late", multi_stream=True, segmentation=True) if v2 else {}
    ck = dict(model_state_dict={"weight": torch.ones(1)}, exp_cfg=cfg)
    with patch.object(loader, "_MODEL", None), patch.object(loader, "_torch_load", return_value=ck), \
         patch.object(loader, "materialize_checkpoint", return_value=Path("fake.pt")), \
         patch.object(loader, "VERITASModel", FakeModel), \
         patch("inference.architecture_v2.VERITASV2", FakeModel):
        model, device = loader.load_veritas(Path("fake.pt"), "cpu", True, True)
        assert model.weight.item() == 1
        assert device.type == "cpu" and not model.training


def test_dataset_flip_keeps_odd_letterbox_padding_aligned(tmp_path):
    import cv2
    import pandas as pd
    variant, fusion = generator.variants()[0]
    nb = generator.build_training_nb(variant, fusion)
    src = generator.cell_starting(nb, "IMAGENET_MEAN_NP =").source
    # Stop before loaders are constructed, preserving the actual dataset code.
    src = src[:src.index('if int(CONFIG["image_size"])')]
    env = dict(np=np, torch=torch, cv2=cv2, pd=pd, random=__import__("random"),
               Dataset=torch.utils.data.Dataset, DataLoader=torch.utils.data.DataLoader,
               CONFIG=dict(seed=42, image_size=600, batch_size=4, num_workers=0))
    exec(src, env)
    rgb = np.zeros((5, 4, 3), dtype=np.uint8)
    rgb[:, 0] = 255
    image, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    cv2.imwrite(str(image), rgb)
    canvas = env["letterbox_rgb"](rgb, 7)
    cv2.imwrite(str(mask_path), canvas[:, :, 0])
    row = dict(path=str(image), mask_path=str(mask_path), crop_x0=0, crop_y0=0, crop_x1=4, crop_y1=5,
               label=1, file_name=image.name)
    dataset = env["GeneratedMaskDataset"](pd.DataFrame([row]), 7, augment=True)
    # No photometric change; apply the geometric flip.
    with patch.object(env["random"], "Random") as rng:
        rng.return_value.random.side_effect = [1.0, 0.0]
        sample = dataset[0]
    recovered = sample["image"][0] * 0.229 + 0.485
    torch.testing.assert_close((recovered > 0.5).float(), sample["mask"][0])


@pytest.mark.parametrize("invalid", [None, "smoke", "protocol", "missing", "duplicate", "nan"])
def test_compare_requires_matched_final_results_and_reports_delta(tmp_path, monkeypatch, invalid):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    rows = []
    for i, (variant, _) in enumerate(generator.variants()):
        rows.append(dict(experiment=variant["exp"], protocol_hash="shared", split="test",
                         implementation_version="ow_v2", epochs_completed=3, n=10905,
                         accuracy=0.8 + i * 0.01, f1=0.8, precision=0.8, recall=0.8))
    if invalid == "smoke":
        rows[-1]["epochs_completed"] = 1
    elif invalid == "protocol":
        rows[-1]["protocol_hash"] = "different"
    elif invalid == "missing":
        rows.pop()
    elif invalid == "duplicate":
        rows.append(dict(rows[-1]))
    elif invalid == "nan":
        rows[-1]["accuracy"] = float("nan")
    pd.DataFrame(rows).to_csv(tmp_path / "experiment_results.csv", index=False)
    if invalid:
        with pytest.raises(ValueError):
            exec(generator.COMPARE_SOURCE, {})
    else:
        exec(generator.COMPARE_SOURCE, {})
        result = pd.read_csv(tmp_path / "kaggle_working/ow_v2_compare.csv")
        assert result.accuracy_delta_pp_vs_b7.iloc[0] == 0
        assert result.accuracy_delta_pp_vs_b7.iloc[-1] == pytest.approx(4)


def test_generated_training_resumes_without_losing_accumulated_tail_or_rng(tmp_path):
    """An interrupted epoch must match uninterrupted optimization and loss logs."""
    import gc
    import math
    import os
    import random
    import time
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    variant, fusion = generator.variants()[0]
    nb = generator.build_training_nb(variant, fusion)

    class TinyDataset(torch.utils.data.Dataset):
        epoch = 0

        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return dict(image=torch.tensor([idx / 8]), label=torch.tensor([float(idx % 2)]),
                        mask=torch.zeros(1, 1, 1), file_name=str(idx), path=str(idx))

    class TinyModel(nn.Module):
        segmentation = False

        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1, 1))

        def forward(self, x):
            logits = self.head(x)
            return dict(classification_logits=logits, fake_probability=logits.sigmoid())

    class Progress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            pass

    def set_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def build_model():
        set_seed(7)
        return TinyModel()

    env = dict(torch=torch, nn=nn, np=np, random=random, math=math, time=time, os=os, gc=gc, json=json,
               Path=Path, pd=__import__("pandas"), cv2=__import__("cv2"), tqdm=Progress,
               Dataset=torch.utils.data.Dataset, DataLoader=torch.utils.data.DataLoader,
               DEVICE=torch.device("cpu"), CONFIG=dict(seed=42),
               accuracy_score=accuracy_score, confusion_matrix=confusion_matrix, f1_score=f1_score,
               precision_score=precision_score, recall_score=recall_score, set_seed=set_seed, build_model=build_model,
               VERITASLossV2=VERITASLossV2, validate_resume_checkpoint=validate_resume_checkpoint,
               EXP_CFG={"implementation_version": "ow_v2", "fusion": "rgb"}, PROTOCOL_HASH="test")
    src = generator.cell_starting(nb, "IMAGENET_MEAN_NP =").source
    exec(src[:src.index('if int(CONFIG["image_size"])')], env)
    config = dict(experiment="baseline_effb7_ow_v2", seed=42, epochs=1, epochs_per_run=1, batch_size=2,
                  learning_rate=0.01, weight_decay=0, lambda_cls=1, lambda_seg=0.25, dice_weight=1,
                  use_amp=False, gradient_accumulation=3, resume=True, resume_from_input=False,
                  skip_training=False, mid_epoch_checkpoint_minutes=30, time_budget_hours=1,
                  classification_threshold=0.5, mask_threshold=0.5)
    env["CONFIG"] = config
    checkpoint = generator.cell_starting(nb, 'EXP = CONFIG["experiment"]').source
    training = generator.cell_starting(nb, "def mask_metrics_from_tensors").source
    exec(checkpoint, env)
    exec(training[:training.index("MODEL, EPOCHS_DONE")], env)

    def run(folder, interrupt=False):
        work = tmp_path / folder
        (work / "checkpoints").mkdir(parents=True, exist_ok=True)
        (work / "logs").mkdir(exist_ok=True)
        dataset = TinyDataset()
        sampler = env["EpochSeededSampler"](len(dataset), 42)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler,
                                            generator=torch.Generator().manual_seed(42))
        env.update(WORKING=work, TRAIN_LOADER=loader, VAL_LOADER=loader, NOTEBOOK_START_TIME=time.time())
        config["time_budget_hours"] = -1 if interrupt else 1
        return env["train_one_experiment"]()

    expected, epochs, _, expected_history = run("whole")
    _, epochs, _, _ = run("resumed", interrupt=True)
    assert epochs == 0
    partial = torch.load(tmp_path / "resumed/checkpoints/baseline_effb7_ow_v2_last.pt")["partial"]
    assert partial["step"] == 3 and partial["seen"] == 6
    actual, epochs, _, actual_history = run("resumed")
    assert epochs == 1
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], tensor)
    assert actual_history[-1]["train_loss"] == pytest.approx(expected_history[-1]["train_loss"])
