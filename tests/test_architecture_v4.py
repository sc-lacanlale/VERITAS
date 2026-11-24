"""Unit tests verifying VERITASV4 & VERITASV4.1 shapes, gradient flow, active seg->cls coupling,
RGB/Forensic decoupling, letterbox safety, and DDP unused-parameter safety."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "veritas-app"))

from backend.inference.architecture_v4 import VERITASV4


def make_letterboxed_input(batch=2, size=256):
    x = torch.randn(batch, 3, size, size)
    valid = torch.zeros(batch, 1, size, size)
    valid[:, :, 24 : size - 24, 32 : size - 32] = 1.0
    x = x * valid
    return x, valid


def test_v4_all():
    torch.manual_seed(0)
    x, valid = make_letterboxed_input(batch=2, size=256)

    # 1. Baseline variant
    base = VERITASV4(variant="baseline", pretrained=False)
    out_b = base(x, valid)
    assert out_b["classification_logits"].shape == (2, 1)
    assert out_b["fake_probability"].shape == (2, 1)
    assert "segmentation_logits" not in out_b

    # 2. Legacy VERITAS v4 variant
    model = VERITASV4(variant="veritas_v4", pretrained=False)
    out = model(x, valid)
    assert out["classification_logits"].shape == (2, 1)
    assert out["aux_forensic_logits"].shape == (2, 1)
    assert out["segmentation_logits"].shape == (2, 1, 256, 256)
    assert out["stream_gates"].shape == (2, 4)
    assert torch.allclose(out["stream_gates"].sum(dim=1), torch.ones(2), atol=1e-5)

    # 3. Letterbox safety for v4
    srm, freq = model.streams(x, valid)
    assert srm.shape == (2, 9, 256, 256) and freq.shape == (2, 9, 256, 256)
    assert torch.isfinite(srm).all() and torch.isfinite(freq).all()
    assert srm[:, :, :24, :].abs().max().item() == 0.0, "SRM leaked into letterbox padding!"
    assert freq[:, :, :24, :].abs().max().item() == 0.0, "Freq leaked into letterbox padding!"

    # 4. VERITAS v4.1 variant (pruned Freq + DecoupledForensicFusionV41)
    m41 = VERITASV4(variant="veritas_v4_1", pretrained=False)
    out41 = m41(x, valid)
    assert out41["classification_logits"].shape == (2, 1)
    assert out41["rgb_logits"].shape == (2, 1)
    assert out41["aux_forensic_logits"].shape == (2, 1)
    assert out41["segmentation_logits"].shape == (2, 1, 256, 256)
    assert out41["stream_gates"].shape == (2, 2)
    assert torch.allclose(out41["stream_gates"].sum(dim=1), torch.ones(2), atol=1e-5)
    alpha = out41["stream_gates"][0, 0].item()
    assert 0.35 <= alpha <= 0.65, f"Alpha out of bounds: {alpha}"

    # 5. Letterbox safety for v4.1 SRM stream
    srm41 = m41.streams(x, valid)
    assert srm41.shape == (2, 9, 256, 256)
    assert torch.isfinite(srm41).all()
    assert srm41[:, :, :24, :].abs().max().item() == 0.0, "v4.1 SRM leaked into letterbox padding!"

    # 6. Decoupling check: perturbing seg_logits must NOT change rgb_logits, MUST change aux & cls logits!
    m41.eval()
    with torch.no_grad():
        o_orig = m41(x, valid)
        hook = m41.seg_head.register_forward_hook(lambda m, inp, res: res + 5.0)
        o_pert = m41(x, valid)
        hook.remove()
    rgb_delta = (o_pert["rgb_logits"] - o_orig["rgb_logits"]).abs().max().item()
    aux_delta = (o_pert["aux_forensic_logits"] - o_orig["aux_forensic_logits"]).abs().max().item()
    cls_delta = (o_pert["classification_logits"] - o_orig["classification_logits"]).abs().max().item()
    assert rgb_delta == 0.0, f"RGB head leaked segmentation mask (delta={rgb_delta})!"
    assert aux_delta > 1e-4, f"Aux forensic head did not react to mask (delta={aux_delta})!"
    assert cls_delta > 1e-4, f"Fused cls head did not react to mask (delta={cls_delta})!"

    # 7. Segmentation loss sends gradients to B7 stage 7 AND SRM encoder in v4.1
    m41.train()
    m41.zero_grad()
    o_seg = m41(x, valid)
    o_seg["segmentation_logits"].pow(2).mean().backward()
    stage7_grad = sum(p.grad.abs().sum().item() for p in m41.encoder[7].parameters() if p.grad is not None)
    srm_grad = sum(p.grad.abs().sum().item() for p in m41.srm_encoder.parameters() if p.grad is not None)
    assert stage7_grad > 0, "v4.1 Stage 7 got zero gradient from segmentation!"
    assert srm_grad > 0, "v4.1 SRM encoder got zero gradient from segmentation!"

    # 8. DDP safety in BOTH Stage 1 (frozen B7) and Stage 2 (unfrozen B7)
    for stage_name, freeze in [("Stage1-Frozen", True), ("Stage2-Unfrozen", False)]:
        m41.freeze_backbone(freeze)
        m41.train()
        m41.zero_grad()
        o_step = m41(x, valid)
        loss = (
            o_step["classification_logits"].pow(2).mean()
            + 0.5 * o_step["rgb_logits"].pow(2).mean()
            + 0.5 * o_step["aux_forensic_logits"].pow(2).mean()
            + 0.5 * o_step["segmentation_logits"].pow(2).mean()
        )
        loss.backward()
        unused = [n for n, p in m41.named_parameters() if p.requires_grad and p.grad is None]
        assert not unused, f"v4.1 {stage_name} unused params would break DDP: {unused[:10]}"

    # 9. Parameter counts & 600x600 timing comparison (v4 vs v4.1)
    p_base = sum(p.numel() for p in base.parameters()) / 1e6
    p_v4 = sum(p.numel() for p in model.parameters()) / 1e6
    p_v41 = sum(p.numel() for p in m41.parameters()) / 1e6

    x600, v600 = make_letterboxed_input(batch=1, size=600)
    m41.eval()
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(x600, v600)
        dt4 = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        o41 = m41(x600, v600)
        dt41 = (time.perf_counter() - t1) * 1000
    assert o41["segmentation_logits"].shape == (1, 1, 600, 600)

    print("ALL VERITAS V4 & V4.1 TESTS PASSED!")
    print(f"Params (M): baseline={p_base:.2f}M, v4={p_v4:.2f}M (+{p_v4-p_base:.2f}M), v4.1={p_v41:.2f}M (+{p_v41-p_base:.2f}M)")
    print(f"Decoupling check: rgb_delta={rgb_delta:.6f} (exact 0!), aux_delta={aux_delta:.5f}, cls_delta={cls_delta:.5f}")
    print(f"v4.1 Seg grads -> stage7={stage7_grad:.4f}, srm={srm_grad:.4f}")
    print(f"CPU 600x600 single-image forward: v4={dt4:.0f}ms -> v4.1={dt41:.0f}ms")


if __name__ == "__main__":
    test_v4_all()
