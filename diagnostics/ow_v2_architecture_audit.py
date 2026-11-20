"""Read-only CPU probes of OW v2 design properties; no training or weight download.

Run from the repo root:
    .venv-ow-v2/Scripts/python.exe diagnostics/ow_v2_architecture_audit.py
Synthetic results establish properties, not validation-accuracy effects.
"""
import json
from pathlib import Path
import sys

import torch
from torchvision.models import efficientnet_b7

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "veritas-app/backend"))
from inference.architecture import IMAGENET_MEAN, IMAGENET_STD
from inference.architecture_v2 import ForensicStreams, VERITASV2
from ow_v2_training import VERITASLossV2


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    result = {"torch_version": torch.__version__, "device": "cpu"}

    # Simulate nonzero learned projection weights, rather than its zero init.
    rgb_features, auxiliary = torch.randn(4, 2560), torch.randn(4, 256)
    projection = torch.nn.Linear(256, 2560, bias=False)
    classifier = torch.nn.Linear(2560, 1)
    full = classifier(rgb_features + projection(auxiliary))
    collapsed = torch.nn.functional.linear(rgb_features, classifier.weight, classifier.bias)
    collapsed += torch.nn.functional.linear(auxiliary, classifier.weight @ projection.weight)
    torch.testing.assert_close(full, collapsed, atol=1e-5, rtol=1e-5)
    result["late_fusion"] = {
        "max_collapse_error": float((full - collapsed).abs().max()),
        "projection_parameters": projection.weight.numel(),
        "equivalent_auxiliary_weights": 256,
        "equation": "z = w.T @ rgb + (w.T @ P) @ [srm; fft] + b",
    }

    streams = ForensicStreams()
    result["constant_gray_geometry"] = {}
    for name, start, end in [("square", 0, 600), ("rectangular", 150, 450)]:
        rgb = torch.zeros(1, 3, 600, 600)
        rgb[:, :, start:end, :] = 0.5
        srm, spectrum = streams((rgb - IMAGENET_MEAN) / IMAGENET_STD)
        assert torch.isfinite(srm).all() and torch.isfinite(spectrum).all()
        result["constant_gray_geometry"][name] = {
            "srm_mean_absolute": float(srm.abs().mean()),
            "srm_absolute_sum": float(srm.abs().sum()),
            "fft_mean_absolute": float(spectrum.abs().mean()),
            "fft_maximum": float(spectrum.max()),
        }

    rgb = torch.full((1, 3, 64, 64), 0.5)
    checker = ((torch.arange(64).view(-1, 1) + torch.arange(64).view(1, -1)) % 2 * 2 - 1) * 0.1
    changed = rgb.clone()
    changed[:, 0] += checker
    changed[:, 1] -= checker * (0.299 / 0.587)
    srm_before, _ = streams((rgb - IMAGENET_MEAN) / IMAGENET_STD)
    srm_after, _ = streams((changed - IMAGENET_MEAN) / IMAGENET_STD)
    rgb_change = float((changed - rgb).abs().max())
    srm_change = float((srm_after - srm_before).abs().max())
    assert rgb_change > 0.09 and srm_change < 1e-6
    result["chromatic_checkerboard"] = {
        "maximum_rgb_change": rgb_change,
        "maximum_srm_change": srm_change,
    }

    encoder = efficientnet_b7(weights=None).features.eval()
    x = torch.zeros(1, 3, 600, 600)
    expected = [(64, 300), (32, 300), (48, 150), (80, 75), (160, 38),
                (224, 38), (384, 19), (640, 19), (2560, 19)]
    result["encoder_shapes_600"] = {}
    with torch.no_grad():
        for i, stage in enumerate(encoder):
            x = stage(x)
            channels, size = expected[i]
            assert tuple(x.shape) == (1, channels, size, size)
            result["encoder_shapes_600"][str(i)] = list(x.shape)
    assert len(encoder) == len(expected)
    del encoder, x

    # Standard DeepLab OS16 rates: padding coverage is an ablation opportunity,
    # not evidence that these conventional rates are incorrectly implemented.
    height = 38
    result["aspp_sampling"] = {
        str(rate): {
            "mean_valid_tap_fraction": (3 - 2 * rate / height) ** 2 / 9,
            "fully_in_bounds_center_fraction": max(height - 2 * rate, 0) ** 2 / height**2,
        }
        for rate in (6, 12, 18)
    }

    model = VERITASV2(True, True, pretrained=False, fusion="late").eval()
    assert isinstance(model.aux_projection, torch.nn.Linear)
    assert [type(layer) for layer in model.cls_head.fc] == [torch.nn.Dropout, torch.nn.Linear]
    output = model(torch.randn(1, 3, 64, 64))
    output["segmentation_logits"].mean().backward()
    reach = {str(i): any(p.grad is not None for p in stage.parameters())
             for i, stage in enumerate(model.encoder)}
    assert reach == {str(i): i <= 5 for i in range(9)}
    result["segmentation_only_gradient_reach"] = reach
    del model, output

    # The loss returns detached component metrics. Reconstruct its current Dice
    # expression for component gradients, and verify against the actual loss.
    logits = torch.zeros(1, 1, 600, 600, requires_grad=True)
    masks = torch.zeros_like(logits)
    criterion = VERITASLossV2(lambda_seg=1.0, segmentation=True)
    actual = criterion({"classification_logits": torch.zeros(1, 1, requires_grad=True),
                        "segmentation_logits": logits}, torch.zeros(1, 1), masks)
    probs = logits.float().sigmoid()
    dims = (1, 2, 3)
    dice = 1 - ((2 * (probs * masks).sum(dims) + 1e-6) /
                (probs.sum(dims) + masks.sum(dims) + 1e-6)).mean()
    bce = criterion.bce(logits, masks)
    torch.testing.assert_close(actual["segmentation_dice"], dice.detach())
    torch.testing.assert_close(actual["segmentation_bce"], bce.detach())
    dice_grad = torch.autograd.grad(dice, logits)[0]
    bce_grad = torch.autograd.grad(bce, logits)[0]
    actual_grad = torch.autograd.grad(actual["total"], logits)[0]
    torch.testing.assert_close(actual_grad, (dice_grad + bce_grad) / 2)
    ratio = float(dice_grad.abs().mean() / bce_grad.abs().mean())
    assert torch.isfinite(actual_grad).all() and 0 < ratio < 1e-8
    result["empty_mask_600_at_zero_logits"] = {
        "dice_loss": float(actual["segmentation_dice"]),
        "bce_loss": float(actual["segmentation_bce"]),
        "dice_mean_absolute_logit_gradient": float(dice_grad.abs().mean()),
        "bce_mean_absolute_logit_gradient": float(bce_grad.abs().mean()),
        "dice_to_bce_gradient_ratio": ratio,
        "actual_averaged_segmentation_mean_absolute_gradient": float(actual_grad.abs().mean()),
    }

    result["decoder_3x3_mac_per_image_600"] = {
        "current_dense": (304 * 256 * 9 + 256 * 256 * 9) * 150**2,
        "depthwise_separable_candidate": (304 * 9 + 304 * 256 + 256 * 9 + 256 * 256) * 150**2,
    }
    result["self_checks_passed"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
