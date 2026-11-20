"""Actual-model contracts and signal-processing checks, without pretrained downloads."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "veritas-app/backend"))
from inference.architecture import IMAGENET_MEAN, IMAGENET_STD
from inference.architecture_v2 import VERITASV2
from inference.architecture_v3 import ForensicStreamsV3, ResidualFusionV3, VERITASV3

torch.set_num_threads(2)


def normalize(rgb):
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def test_local_fft_ignores_explicit_outer_padding_including_odd_offset():
    torch.manual_seed(13)
    content = torch.rand(1, 3, 37, 45)
    streams = ForensicStreamsV3(color_srm=True, frequency="local")
    _, expected = streams(normalize(content), torch.ones(1, 1, 37, 45))
    canvas = torch.zeros(1, 3, 64, 80)
    canvas[:, :, 13:50, 18:63] = content
    mask = torch.zeros(1, 1, 64, 80)
    mask[:, :, 13:50, 18:63] = 1
    _, actual = streams(normalize(canvas), mask)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    # Flip moves odd padding to the opposite side; content coordinates still agree.
    _, flipped = streams(normalize(canvas.flip(-1)), mask.flip(-1))
    _, flipped_expected = streams(normalize(content.flip(-1)), torch.ones(1, 1, 37, 45))
    torch.testing.assert_close(flipped, flipped_expected, atol=1e-7, rtol=1e-5)
    assert actual.shape == (1, 9, 32, 32)


@pytest.mark.parametrize("bad", ["empty", "hole", "shape", "fractional", "nonfinite"])
def test_invalid_content_masks_fail_clearly(bad):
    streams = ForensicStreamsV3(frequency="local")
    mask = torch.ones(1, 1, 32, 32)
    if bad == "empty":
        mask.zero_()
    elif bad == "hole":
        mask[:, :, 2, 3] = 0
    elif bad == "shape":
        mask = mask.squeeze(1)
    elif bad == "fractional":
        mask[:, :, 2, 3] = 0.5
    else:
        mask[:, :, 2, 3] = float("nan")
    with pytest.raises(ValueError, match="valid_mask"):
        streams(torch.zeros(1, 3, 32, 32), mask)


def test_color_srm_retains_luminance_cancelling_chromatic_residuals():
    rgb = torch.full((1, 3, 64, 64), 0.5)
    checker = ((torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2 * 2 - 1) * 0.1
    altered = rgb.clone()
    altered[:, 0] += checker
    altered[:, 1] -= checker * (0.299 / 0.587)
    grayscale = ForensicStreamsV3(color_srm=False)
    color = ForensicStreamsV3(color_srm=True)
    before_gray, _ = grayscale(normalize(rgb))
    after_gray, _ = grayscale(normalize(altered))
    before_color, _ = color(normalize(rgb))
    after_color, _ = color(normalize(altered))
    torch.testing.assert_close(before_gray, after_gray, atol=1e-6, rtol=1e-5)
    assert before_color.shape[1] == 9
    assert (before_color - after_color).abs().max() > 0.01


@pytest.mark.parametrize("frequency", ["global", "local"])
@pytest.mark.parametrize("black", [False, True])
def test_streams_are_float32_finite_under_autocast_for_constant_and_normal_inputs(frequency, black):
    rgb = torch.zeros(1, 3, 600, 600) if black else torch.rand(1, 3, 600, 600)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        srm, spectrum = ForensicStreamsV3(color_srm=True, frequency=frequency)(normalize(rgb))
    assert srm.dtype == spectrum.dtype == torch.float32
    assert torch.isfinite(srm).all() and torch.isfinite(spectrum).all()
    assert srm.shape == (1, 9, 600, 600)
    assert spectrum.shape == ((1, 9, 32, 32) if frequency == "local" else (1, 3, 600, 600))


def test_baseline_initial_logits_are_preserved_across_candidates():
    x = torch.randn(1, 3, 64, 64)
    torch.manual_seed(7)
    baseline = VERITASV3(pretrained=False).eval()
    with torch.no_grad():
        expected = baseline(x)["classification_logits"]
    del baseline
    options = [dict(multi_stream=True, fusion="early"),
               dict(multi_stream=True, fusion="late_additive"),
               dict(multi_stream=True, fusion="late_nonlinear", color_srm=True, frequency="local", pooling="avg_max"),
               dict(pooling="avg_max", segmentation=True, decoder="separable")]
    for cfg in options:
        torch.manual_seed(7)
        model = VERITASV3(pretrained=False, **cfg).eval()
        with torch.no_grad():
            actual = model(x)["classification_logits"]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        del model


def test_additive_reference_matches_v2_after_nonzero_auxiliary_projection():
    torch.manual_seed(5)
    reference = VERITASV2(True, False, pretrained=False, fusion="late").eval()
    with torch.no_grad():
        reference.aux_projection.weight.normal_(0, 0.02)
    candidate = VERITASV3(True, False, pretrained=False, fusion="late_additive").eval()
    missing, unexpected = candidate.load_state_dict(reference.state_dict(), strict=False)
    assert set(missing) == {"streams.fft_window", "streams.fft_bands"} and not unexpected
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        actual = candidate(x)["classification_logits"]
        expected = reference(x)["classification_logits"]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_residual_mlp_can_represent_cross_stream_interactions():
    torch.manual_seed(9)
    head = ResidualFusionV3(8, hidden=16).double().eval()
    assert not torch.count_nonzero(head[-1].weight)
    assert torch.count_nonzero(head[1].weight) > 0
    with torch.no_grad():
        head[-1].weight.normal_(0, 0.1)
    x = torch.randn(1, 8, dtype=torch.float64)
    rgb_delta = torch.zeros_like(x)
    rgb_delta[:, :4] = 0.5
    aux_delta = torch.zeros_like(x)
    aux_delta[:, 4:] = -0.3
    interaction = head(x + rgb_delta + aux_delta) - head(x + rgb_delta) - head(x + aux_delta) + head(x)
    assert interaction.abs().item() > 1e-6


@pytest.mark.parametrize("decoder", ["dense", "separable"])
def test_real_b7_classification_segmentation_backward_and_auxiliary_learning(decoder):
    model = VERITASV3(True, True, pretrained=False, fusion="late_nonlinear", color_srm=True,
                       frequency="local", pooling="avg_max", decoder=decoder).train()
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x, valid_mask=torch.ones(2, 1, 64, 64))
    assert outputs["classification_logits"].shape == (2, 1)
    assert outputs["segmentation_logits"].shape == (2, 1, 64, 64)
    loss = nn.functional.binary_cross_entropy_with_logits(outputs["classification_logits"], torch.tensor([[0.], [1.]]))
    loss = loss + outputs["segmentation_logits"].square().mean()
    loss.backward()
    assert model.encoder[0][0].weight.grad.abs().sum() > 0
    assert model.seg_head.decoder[-1].weight.grad.abs().sum() > 0
    assert model.fusion_head[-1].weight.grad.abs().sum() > 0
    # The final residual output starts at zero; one update opens its upstream path.
    with torch.no_grad():
        model.fusion_head[-1].weight.add_(-0.1 * model.fusion_head[-1].weight.grad)
    model.zero_grad(set_to_none=True)
    logits = model(x)["classification_logits"]
    nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[0.], [1.]])).backward()
    assert model.srm_encoder[0].weight.grad.abs().sum() > 0
    assert model.fft_encoder[0].weight.grad.abs().sum() > 0


def test_improved_full_model_forward_at_actual_600px():
    model = VERITASV3(True, True, pretrained=False, fusion="late_nonlinear", color_srm=True,
                       frequency="local", pooling="avg_max", decoder="separable").eval()
    assert model.requires_valid_mask
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 600, 600), torch.ones(1, 1, 600, 600))
    assert outputs["classification_logits"].shape == (1, 1)
    assert outputs["segmentation_logits"].shape == (1, 1, 600, 600)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_early_reference_rejects_uncontrolled_stream_changes():
    with pytest.raises(ValueError, match="corrected early"):
        VERITASV3(True, False, pretrained=False, fusion="early", color_srm=True)
    with pytest.raises(ValueError, match="corrected early"):
        VERITASV3(True, False, pretrained=False, fusion="early", frequency="local")
