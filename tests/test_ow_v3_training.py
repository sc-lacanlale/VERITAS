"""Loss and optimization contracts for fresh OW v3 experiments."""
import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ow_v3_training import VERITASLossV3, build_optimizer, build_scheduler, freeze_backbone_bn, validate_resume_checkpoint

torch.set_num_threads(2)


def _outputs(n=2, size=8, dtype=torch.float32):
    return dict(classification_logits=torch.zeros(n, 1, requires_grad=True),
                segmentation_logits=torch.zeros(n, 1, size, size, dtype=dtype, requires_grad=True))


def test_real_masks_are_zeroed_in_memory_and_receive_full_bce():
    output = _outputs()
    teacher_masks = torch.ones(2, 1, 8, 8)
    before = teacher_masks.clone()
    labels = torch.zeros(2, 1)
    losses = VERITASLossV3(segmentation=True)(output, labels, teacher_masks, seg_valid=torch.zeros(2))
    expected = F.binary_cross_entropy_with_logits(output["segmentation_logits"], torch.zeros_like(teacher_masks))
    torch.testing.assert_close(losses["segmentation"], expected)
    assert losses["segmentation_dice"] == 0 and losses["segmentation_positive_samples"] == 0
    assert losses["segmentation_valid_samples"] == 2
    torch.testing.assert_close(teacher_masks, before)
    losses["total"].backward()
    assert (output["segmentation_logits"].grad > 0).all()


def test_mixed_targets_use_per_image_empty_aware_weighting():
    output = _outputs()
    masks = torch.ones(2, 1, 8, 8)
    labels = torch.tensor([[0.], [1.]])
    losses = VERITASLossV3(lambda_seg=0.25, segmentation=True)(output, labels, masks)
    bce = torch.tensor(2.).log()
    dice = 1 - (2 * 32 + 1e-6) / (32 + 64 + 1e-6)
    expected_seg = (bce + (bce + dice) / 2) / 2
    torch.testing.assert_close(losses["segmentation"], expected_seg)
    torch.testing.assert_close(losses["total"], bce + 0.25 * expected_seg)
    assert losses["segmentation_valid_samples"] == 2 and losses["segmentation_positive_samples"] == 1


def test_empty_or_rejected_fake_masks_do_not_train_segmentation():
    output = _outputs()
    labels = torch.ones(2, 1)
    masks = torch.zeros(2, 1, 8, 8)
    masks[1] = 1
    losses = VERITASLossV3(segmentation=True)(output, labels, masks, seg_valid=torch.tensor([1, 0]))
    assert losses["segmentation"] == 0 and losses["segmentation_ignored_fake_samples"] == 2
    assert losses["segmentation_valid_samples"] == 0
    losses["total"].backward()
    assert output["segmentation_logits"].grad is not None
    assert output["segmentation_logits"].grad.abs().sum() == 0
    assert output["classification_logits"].grad.abs().sum() > 0


def test_letterbox_pixels_never_contribute_to_segmentation():
    output = _outputs(n=1)
    output["segmentation_logits"].data.fill_(3)
    valid = torch.zeros(1, 1, 8, 8)
    valid[:, :, 2:6, 1:7] = 1
    losses = VERITASLossV3(segmentation=True)(output, torch.zeros(1, 1), torch.ones_like(valid), valid_mask=valid)
    torch.testing.assert_close(losses["segmentation"], F.softplus(torch.tensor(3.)))
    assert losses["segmentation_valid_pixels"] == 24
    losses["total"].backward()
    gradient = output["segmentation_logits"].grad
    assert gradient[valid == 0].abs().sum() == 0
    assert (gradient[valid == 1] > 0).all()


def test_mask_with_positive_pixels_only_in_padding_is_unavailable():
    output = _outputs(n=1)
    valid = torch.ones(1, 1, 8, 8)
    valid[:, :, 0] = 0
    masks = 1 - valid
    losses = VERITASLossV3(segmentation=True)(output, torch.ones(1, 1), masks, valid_mask=valid)
    assert losses["segmentation"] == 0 and losses["segmentation_ignored_fake_samples"] == 1


def test_all_invalid_canvas_gives_differentiable_zero():
    output = _outputs()
    masks = torch.ones(2, 1, 8, 8)
    losses = VERITASLossV3(segmentation=True)(output, torch.tensor([[0.], [1.]]), masks, valid_mask=torch.zeros_like(masks))
    assert losses["segmentation"] == 0 and losses["segmentation_valid_pixels"] == 0
    losses["total"].backward()
    assert torch.isfinite(output["segmentation_logits"].grad).all()


def test_600px_half_precision_dice_stays_finite():
    output = _outputs(n=2, size=600, dtype=torch.float16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses = VERITASLossV3(segmentation=True)(output, torch.tensor([[0.], [1.]]), torch.ones(2, 1, 600, 600))
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total"].backward()
    assert torch.isfinite(output["segmentation_logits"].grad).all()
    assert output["segmentation_logits"].grad.abs().sum() > 0


def test_classification_only_is_unchanged_bce():
    logits = torch.tensor([[0.3], [-1.2]], requires_grad=True)
    labels = torch.tensor([[1.], [0.]])
    losses = VERITASLossV3()(dict(classification_logits=logits), labels)
    torch.testing.assert_close(losses["total"], F.binary_cross_entropy_with_logits(logits, labels))
    assert losses["segmentation"] == 0


@pytest.mark.parametrize("bad", [float("nan"), -1., 0.2])
def test_binary_target_validation_rejects_invalid_labels(bad):
    with pytest.raises(ValueError):
        VERITASLossV3()(_outputs(n=1), torch.tensor([[bad]]))


def test_shape_and_mask_validation():
    criterion = VERITASLossV3(segmentation=True)
    with pytest.raises(ValueError, match="same"):
        criterion(_outputs(), torch.zeros(2))
    with pytest.raises(ValueError, match="required"):
        criterion(_outputs(), torch.zeros(2, 1))
    masks = torch.ones(2, 1, 8, 8)
    with pytest.raises(ValueError, match="valid_mask"):
        criterion(_outputs(), torch.zeros(2, 1), masks, valid_mask=torch.ones(2, 8, 8))
    with pytest.raises(ValueError, match="seg_valid"):
        criterion(_outputs(), torch.zeros(2, 1), masks, seg_valid=torch.ones(2, 2))
    masks[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="masks"):
        criterion(_outputs(), torch.zeros(2, 1), masks)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4))
        self.head = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 1))


def test_optimizer_covers_each_trainable_parameter_exactly_once():
    model = TinyModel()
    model.encoder[0].bias.requires_grad_(False)
    optimizer = build_optimizer(model, 1e-4, 1e-3, 0.1)
    assert [group["name"] for group in optimizer.param_groups] == ["backbone", "heads"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-4)
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    assert len(actual) == len(set(actual)) and set(actual) == set(expected)


def test_scheduler_warms_up_and_decays_after_optimizer_updates():
    model = TinyModel()
    optimizer = build_optimizer(model, 0.1, 0)
    scheduler = build_scheduler(optimizer, total_updates=10, warmup_fraction=0.3)
    rates = [optimizer.param_groups[1]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[1]["lr"])
    assert rates[:4] == pytest.approx([0.1 / 3, 0.2 / 3, 0.1, 0.1])
    assert all(a >= b for a, b in zip(rates[3:], rates[4:]))
    assert rates[-1] == 0


def test_scheduler_and_optimizer_state_resume_with_identical_next_update():
    model = TinyModel()
    optimizer = build_optimizer(model, 0.1, 0)
    scheduler = build_scheduler(optimizer, 20, 0.2)
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    opt_state, sched_state = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
    restored_optimizer = build_optimizer(TinyModel(), 0.1, 0)
    restored_scheduler = build_scheduler(restored_optimizer, 20, 0.2)
    restored_optimizer.load_state_dict(opt_state)
    restored_scheduler.load_state_dict(sched_state)
    for _ in range(4):
        optimizer.step(); scheduler.step()
        restored_optimizer.step(); restored_scheduler.step()
        assert [g["lr"] for g in optimizer.param_groups] == [g["lr"] for g in restored_optimizer.param_groups]


def test_frozen_backbone_bn_keeps_affine_gradients_and_other_bn_training():
    model = TinyModel().train()
    before_mean = model.encoder[1].running_mean.clone()
    freeze_backbone_bn(model)
    assert not model.encoder[1].training and model.head[0].training
    assert model.encoder[1].weight.requires_grad and model.encoder[1].bias.requires_grad
    model.head(model.encoder(torch.randn(8, 2))).sum().backward()
    assert model.encoder[1].weight.grad is not None
    torch.testing.assert_close(model.encoder[1].running_mean, before_mean)


@pytest.mark.parametrize("settings", [dict(learning_rate=0, weight_decay=0),
                                     dict(learning_rate=0.1, weight_decay=-1),
                                     dict(learning_rate=0.1, weight_decay=0, backbone_lr_multiplier=float("nan"))])
def test_optimizer_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        build_optimizer(TinyModel(), **settings)


def test_scheduler_and_checkpoint_reject_incompatible_protocols():
    optimizer = build_optimizer(TinyModel(), 0.1, 0)
    for updates, fraction in ((0, 0.1), (1.5, 0.1), (10, 1), (10, float("nan"))):
        with pytest.raises(ValueError):
            build_scheduler(optimizer, updates, fraction)
    cfg = {"implementation_version": "ow_v3", "fusion": "late_nonlinear"}
    checkpoint = dict(experiment_name="full_v3", exp_cfg=cfg, protocol_hash="full-data")
    validate_resume_checkpoint(checkpoint, "full_v3", cfg, "full-data")
    with pytest.raises(ValueError):
        validate_resume_checkpoint(checkpoint, "full_v3", cfg, "smoke-data")
