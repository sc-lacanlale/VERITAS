"""VERITAS OW v4 & v4.1 architecture: zero-sync spatial forensic streams,
multi-scale deep-supervised segmentation, and decoupled dual-expert forensic fusion.

Key architectural evolution:
- OW v4 (`variant="veritas_v4"`):
  1. Zero GPU->CPU syncs in ForensicStreamsV4 (vectorized eroded color SRM +
     content-mean-filled batched 2D rfft2/irfft2 directional band-pass).
  2. Spatial multi-scale SRM & frequency encoders (38x38 and 19x19).
  3. Multi-scale ActiveSegmentationHeadV4 tapping EfficientNet-B7 stages 2, 5, 7
     plus spatial forensic maps (38x38).
  4. MaskGuidedContrastiveFusionV4 with 4-stream softmax gating + aux forensic head.

- OW v4.1 (`variant="veritas_v4_1"` — empirically refined from 10,905-image v4
  prototype diagnostics):
  1. Prunes dead Freq FFT stream (learned gate 0.007) and heavy MaskContrast MLP
     (learned gate 0.011), cutting ~40-50% of extra latency and ~1.5 hrs of 140k
     training time while keeping the winning 9-ch color SRM + SpatialForensicEncoderV4
     and ActiveSegmentationHeadV4 (Fake Dice ~0.89, mIoU ~0.85).
  2. Fixes RGB-to-mask co-adaptation collapse (r(v4, Aux) = 0.9918 in v4) via
     DecoupledForensicFusionV41:
       - Pure Global RGB Expert (`rgb_logits`): valid-mask GAP over B7 stage 8
         with ZERO segmentation mask input, preserving full B7 Fake recall.
       - Pure Forensic SRM + Active Mask Expert (`aux_forensic_logits`): pools
         spatial SRM over [global, fg, bg, |fg - bg|] + mask summary stats with
         ZERO direct RGB features, preserving v4 Aux's strong Real FP rejection.
       - Bounded Dual-Expert Logit Fusion (`cls_logits`): learns alpha strictly
         within [0.35, 0.65] (initialized at 0.50) plus a small bounded (+/-0.25)
         cross-modal residual, guaranteeing genuine RGB + Forensic complementarity.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

from .architecture import ClassificationHead, IMAGENET_MEAN, IMAGENET_STD

IMPLEMENTATION_VERSION = "ow_v4_1"


def _srm_kernels_v4() -> torch.Tensor:
    """Pure-PyTorch 3x1x5x5 SRM high-pass kernels (NumPy-version independent)."""
    raw = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=torch.float32,
    )
    norms = raw.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    return (raw / norms).unsqueeze(1)


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, channels), channels)


class SeparableConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )


class ResidualForensicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = SeparableConvBlock(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            _gn(out_ch),
        )
        self.skip = (
            nn.Identity()
            if (stride == 1 and in_ch == out_ch)
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), _gn(out_ch))
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class SRMStreamV41(nn.Module):
    """Vectorized, zero-sync 9-channel color SRM residual extractor (v4.1).

    Applies 3 fixed 5x5 SRM high-pass filters per RGB channel with tanh
    bounding and 5x5 eroded valid_mask gating so letterbox borders never
    produce spurious high-pass spikes.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("srm_weight", _srm_kernels_v4())

    @staticmethod
    def _erode_valid_mask(valid_mask: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)

    def forward(self, normalized_rgb: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        with torch.autocast(device_type=normalized_rgb.device.type, enabled=False):
            x = normalized_rgb.float()
            mean = IMAGENET_MEAN.to(device=x.device, dtype=torch.float32)
            std = IMAGENET_STD.to(device=x.device, dtype=torch.float32)
            rgb = (x * std + mean).clamp(0.0, 1.0)

            if valid_mask is None:
                eroded = torch.ones((rgb.shape[0], 1, rgb.shape[-2], rgb.shape[-1]), device=rgb.device, dtype=torch.float32)
                rgb_filled = rgb
            else:
                vm = valid_mask.to(device=rgb.device, dtype=torch.float32)
                eroded = self._erode_valid_mask(vm)
                denom = vm.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
                content_mean = (rgb * vm).sum(dim=(-2, -1), keepdim=True) / denom
                rgb_filled = rgb * vm + content_mean * (1.0 - vm)

            srm_w = self.srm_weight.float().repeat(3, 1, 1, 1)
            srm = F.conv2d(rgb_filled, srm_w, padding=2, groups=3).tanh() * eroded
        return srm


class ForensicStreamsV4(nn.Module):
    """Vectorized, zero-sync color SRM + spatial directional FFT band-pass (v4)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("srm_weight", _srm_kernels_v4())
        self._band_cache: dict[tuple[int, int, str, int | None], torch.Tensor] = {}

    def _get_freq_masks(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device.type, device.index)
        cached = self._band_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(h, device=device)[:, None]
        fx = torch.fft.rfftfreq(w, device=device)[None, :]
        radius = torch.sqrt(fy.square() + fx.square())
        abs_fy = fy.abs()
        abs_fx = fx.abs()

        low_cut = torch.sigmoid((radius - 0.08) * 40.0)
        mid_hi = torch.sigmoid((radius - 0.25) * 40.0)
        mid_band = low_cut * (1.0 - mid_hi)

        hv_ratio = torch.abs(abs_fx - abs_fy) / (abs_fx + abs_fy + 1e-6)
        high_hv = mid_hi * hv_ratio
        high_diag = mid_hi * (1.0 - hv_ratio)

        masks = torch.stack([mid_band, high_hv, high_diag], dim=0)
        if len(self._band_cache) > 8:
            self._band_cache.clear()
        self._band_cache[key] = masks
        return masks

    @staticmethod
    def _erode_valid_mask(valid_mask: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-valid_mask, kernel_size=5, stride=1, padding=2)

    def forward(
        self, normalized_rgb: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=normalized_rgb.device.type, enabled=False):
            x = normalized_rgb.float()
            mean = IMAGENET_MEAN.to(device=x.device, dtype=torch.float32)
            std = IMAGENET_STD.to(device=x.device, dtype=torch.float32)
            rgb = (x * std + mean).clamp(0.0, 1.0)

            if valid_mask is None:
                vm = torch.ones((rgb.shape[0], 1, rgb.shape[-2], rgb.shape[-1]), device=rgb.device, dtype=torch.float32)
                eroded = vm
                rgb_filled = rgb
            else:
                vm = valid_mask.to(device=rgb.device, dtype=torch.float32)
                eroded = self._erode_valid_mask(vm)
                denom = vm.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
                content_mean = (rgb * vm).sum(dim=(-2, -1), keepdim=True) / denom
                rgb_filled = rgb * vm + content_mean * (1.0 - vm)

            srm_w = self.srm_weight.float().repeat(3, 1, 1, 1)
            srm = F.conv2d(rgb_filled, srm_w, padding=2, groups=3).tanh() * eroded

            h, w = rgb_filled.shape[-2:]
            spec = torch.fft.rfft2(rgb_filled, norm="ortho")
            masks = self._get_freq_masks(h, w, rgb_filled.device)
            filtered = (spec.unsqueeze(2) * masks.unsqueeze(0).unsqueeze(0)).flatten(1, 2)
            spatial_freq = torch.fft.irfft2(filtered, s=(h, w), norm="ortho")
            freq = torch.tanh(spatial_freq * 8.0) * eroded

        return srm, freq


class SpatialForensicEncoderV4(nn.Module):
    """Multi-scale spatial encoder aligned with EfficientNet-B7 stages 5 & 7/8.

    Produces:
      - `mid` at ~H/16 (38x38 for 600px, 128 ch) -> feeds ASPP segmentation head
      - `top` at ~H/32 (19x19 for 600px, 256 ch) -> feeds forensic classification
    """

    def __init__(self, in_ch: int = 9, mid_ch: int = 128, top_ch: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            _gn(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = ResidualForensicBlock(32, 64, stride=2)   # /4  (150)
        self.stage2 = ResidualForensicBlock(64, 96, stride=2)   # /8  (75)
        self.stage3 = nn.Sequential(
            ResidualForensicBlock(96, mid_ch, stride=2),        # /16 (38)
            ResidualForensicBlock(mid_ch, mid_ch, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResidualForensicBlock(mid_ch, top_ch, stride=2),    # /32 (19)
            ResidualForensicBlock(top_ch, top_ch, stride=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        mid = self.stage3(x)
        top = self.stage4(mid)
        return mid, top


class MultiScaleSeparableASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, rates: tuple[int, ...] = (3, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), _gn(out_ch), nn.SiLU(inplace=True))]
        )
        self.branches.extend(SeparableConvBlock(in_ch, out_ch, dilation=r) for r in rates)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            _gn(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.gap(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class ActiveSegmentationHeadV4(nn.Module):
    """Deeply supervised multi-scale segmentation decoder.

    Fuses EfficientNet-B7 stage 7 (640 ch, 19x19) + stage 5 (224 ch, 38x38) +
    spatial forensic maps (128 ch for v4.1 SRM-only, or 256 ch for v4 SRM+Freq)
    before ASPP, then refines with stage 2 (48 ch, 150x150).
    """

    def __init__(
        self,
        high_ch: int = 640,
        mid_ch: int = 224,
        low_ch: int = 48,
        forensic_mid_ch: int = 128,
        use_forensic: bool = True,
        num_forensic_streams: int = 2,
    ):
        super().__init__()
        self.use_forensic = bool(use_forensic)
        self.num_forensic_streams = int(num_forensic_streams)
        self.high_proj = nn.Sequential(nn.Conv2d(high_ch, 192, 1, bias=False), _gn(192), nn.SiLU(inplace=True))
        self.mid_proj = nn.Sequential(nn.Conv2d(mid_ch, 128, 1, bias=False), _gn(128), nn.SiLU(inplace=True))
        fused_ch = 192 + 128
        if self.use_forensic:
            self.forensic_proj = nn.Sequential(
                nn.Conv2d(forensic_mid_ch * self.num_forensic_streams, 128, 1, bias=False),
                _gn(128),
                nn.SiLU(inplace=True),
            )
            fused_ch += 128
        self.aspp = MultiScaleSeparableASPP(fused_ch, out_ch=256, rates=(3, 6, 12))
        self.low_proj = nn.Sequential(nn.Conv2d(low_ch, 48, 1, bias=False), _gn(48), nn.SiLU(inplace=True))
        self.decoder = nn.Sequential(
            SeparableConvBlock(256 + 48, 192),
            SeparableConvBlock(192, 128),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(
        self,
        high: torch.Tensor,
        mid: torch.Tensor,
        low: torch.Tensor,
        out_size: tuple[int, int],
        srm_mid: torch.Tensor | None = None,
        freq_mid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_mid = mid.shape[-2:]
        parts = [
            F.interpolate(self.high_proj(high), size=target_mid, mode="bilinear", align_corners=False),
            self.mid_proj(mid),
        ]
        if self.use_forensic and srm_mid is not None:
            if srm_mid.shape[-2:] != target_mid:
                srm_mid = F.interpolate(srm_mid, size=target_mid, mode="bilinear", align_corners=False)
            if self.num_forensic_streams == 1:
                parts.append(self.forensic_proj(srm_mid))
            elif freq_mid is not None:
                if freq_mid.shape[-2:] != target_mid:
                    freq_mid = F.interpolate(freq_mid, size=target_mid, mode="bilinear", align_corners=False)
                parts.append(self.forensic_proj(torch.cat([srm_mid, freq_mid], dim=1)))

        aspp_out = self.aspp(torch.cat(parts, dim=1))
        up = F.interpolate(aspp_out, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([up, self.low_proj(low)], dim=1))
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class MaskGuidedContrastiveFusionV4(nn.Module):
    """Original v4 4-stream contrastive fusion (kept for backward compatibility)."""

    def __init__(self, rgb_ch: int = 2560, forensic_ch: int = 256, embed_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.rgb_reduce = nn.Sequential(
            nn.Conv2d(rgb_ch, embed_dim, kernel_size=1, bias=False),
            _gn(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.rgb_proj = self._stream_mlp(embed_dim * 4, embed_dim, dropout)
        self.srm_proj = self._stream_mlp(forensic_ch * 4, embed_dim, dropout)
        self.freq_proj = self._stream_mlp(forensic_ch * 4, embed_dim, dropout)
        self.contrast_proj = self._stream_mlp(embed_dim + forensic_ch * 2 + 4, embed_dim, dropout)

        self.gate_net = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 4),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.aux_forensic_cls = nn.Sequential(
            nn.LayerNorm(embed_dim * 2 + 4),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2 + 4, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _stream_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(out_dim),
        )

    @staticmethod
    def _masked_stats(
        feat: torch.Tensor, mask_prob: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-5
        w_glob = valid
        w_fg = mask_prob * valid
        w_bg = (1.0 - mask_prob) * valid

        glob = (feat * w_glob).sum(dim=(-2, -1)) / w_glob.sum(dim=(-2, -1)).clamp_min(eps)
        fg = (feat * w_fg).sum(dim=(-2, -1)) / w_fg.sum(dim=(-2, -1)).clamp_min(eps)
        bg = (feat * w_bg).sum(dim=(-2, -1)) / w_bg.sum(dim=(-2, -1)).clamp_min(eps)
        fg_frac = w_fg.sum(dim=(-2, -1)) / w_glob.sum(dim=(-2, -1)).clamp_min(eps)
        bg_frac = w_bg.sum(dim=(-2, -1)) / w_glob.sum(dim=(-2, -1)).clamp_min(eps)
        contrast_gate = (4.0 * fg_frac * bg_frac).clamp(0.0, 1.0)
        diff = (fg - bg).abs() * contrast_gate
        return torch.cat([glob, fg, bg, diff], dim=1), diff

    def forward(
        self,
        rgb_top: torch.Tensor,
        srm_top: torch.Tensor,
        freq_top: torch.Tensor,
        seg_prob: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = rgb_top.shape[-2:]
        rgb_feat = self.rgb_reduce(rgb_top)
        if srm_top.shape[-2:] != grid:
            srm_top = F.interpolate(srm_top, size=grid, mode="bilinear", align_corners=False)
        if freq_top.shape[-2:] != grid:
            freq_top = F.interpolate(freq_top, size=grid, mode="bilinear", align_corners=False)

        m = F.interpolate(seg_prob.float(), size=grid, mode="bilinear", align_corners=False).to(rgb_feat.dtype)
        if valid_mask is None:
            v = torch.ones_like(m)
        else:
            v = F.interpolate(valid_mask.float(), size=grid, mode="area").clamp(0.0, 1.0).to(rgb_feat.dtype)
            v = (v > 0.1).to(rgb_feat.dtype)

        rgb_vec, rgb_diff = self._masked_stats(rgb_feat, m, v)
        srm_vec, srm_diff = self._masked_stats(srm_top, m, v)
        freq_vec, freq_diff = self._masked_stats(freq_top, m, v)

        v_sum = v.sum(dim=(-2, -1)).clamp_min(1e-5)
        m_mean = (m * v).sum(dim=(-2, -1)) / v_sum
        m_max = (m * v).amax(dim=(-2, -1))
        m_var = (((m - m_mean[:, :, None, None]) ** 2) * v).sum(dim=(-2, -1)) / v_sum
        m_std = torch.sqrt(m_var.clamp_min(1e-6))
        m_hi = ((m > 0.5).to(m.dtype) * v).sum(dim=(-2, -1)) / v_sum
        mask_stats = torch.cat([m_mean, m_max, m_std, m_hi], dim=1)

        t_rgb = self.rgb_proj(rgb_vec)
        t_srm = self.srm_proj(srm_vec)
        t_freq = self.freq_proj(freq_vec)
        t_con = self.contrast_proj(torch.cat([rgb_diff, srm_diff, freq_diff, mask_stats], dim=1))

        tokens = torch.cat([t_rgb, t_srm, t_freq, t_con], dim=1)
        gates = torch.softmax(self.gate_net(tokens), dim=1)
        gated_tokens = torch.cat(
            [
                t_rgb * (1.0 + gates[:, 0:1]),
                t_srm * (1.0 + gates[:, 1:2]),
                t_freq * (1.0 + gates[:, 2:3]),
                t_con * (1.0 + gates[:, 3:4]),
            ],
            dim=1,
        )
        cls_logits = self.classifier(gated_tokens)
        aux_forensic_logits = self.aux_forensic_cls(torch.cat([t_srm, t_freq, mask_stats], dim=1))
        return cls_logits, aux_forensic_logits, gates


class DecoupledForensicFusionV41(nn.Module):
    """Decoupled dual-expert logit fusion with bounded complementarity (v4.1).

    Prevents RGB-to-mask co-adaptation collapse by keeping two strictly
    separated experts:
      1) Pure Global RGB Expert (`rgb_logits`): valid-mask-aware GAP over B7
         stage 8 (2560-d) with ZERO segmentation mask input.
      2) Pure Forensic SRM + Active Mask Expert (`aux_forensic_logits`):
         mask-guided [global, fg, bg, |fg - bg|] pooling over spatial SRM
         features (256-d) + 4 valid-area mask summary statistics, with ZERO
         direct RGB features.
      3) Bounded Logit Fusion (`cls_logits`):
           cls_logits = alpha * rgb_logits + (1 - alpha) * aux_forensic_logits
                        + res_scale * tanh(res_net([rgb, aux, |rgb-aux|, mask_stats]))
         where alpha is strictly bounded to [0.35, 0.65] (init 0.50) and
         res_scale is bounded to [-0.25, +0.25] (init ~0.025).
    """

    def __init__(self, rgb_ch: int = 2560, forensic_ch: int = 256, embed_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.rgb_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(rgb_ch, 1),
        )
        self.srm_proj = nn.Sequential(
            nn.LayerNorm(forensic_ch * 4),
            nn.Dropout(dropout),
            nn.Linear(forensic_ch * 4, embed_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(embed_dim),
        )
        self.forensic_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 4),
            nn.Dropout(dropout),
            nn.Linear(embed_dim + 4, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )
        # Learnable fusion weight bounded strictly to [0.35, 0.65], initialized at 0.50
        self.raw_alpha = nn.Parameter(torch.zeros(1))
        # Lightweight bounded cross-modal residual (max +/- 0.25 logit adjustment)
        self.raw_res_scale = nn.Parameter(torch.full((1,), 0.1))
        self.res_net = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, 32),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    @staticmethod
    def _masked_stats(
        feat: torch.Tensor, mask_prob: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        eps = 1e-5
        w_glob = valid
        w_fg = mask_prob * valid
        w_bg = (1.0 - mask_prob) * valid

        glob = (feat * w_glob).sum(dim=(-2, -1)) / w_glob.sum(dim=(-2, -1)).clamp_min(eps)
        fg = (feat * w_fg).sum(dim=(-2, -1)) / w_fg.sum(dim=(-2, -1)).clamp_min(eps)
        bg = (feat * w_bg).sum(dim=(-2, -1)) / w_bg.sum(dim=(-2, -1)).clamp_min(eps)
        fg_frac = w_fg.sum(dim=(-2, -1)) / w_glob.sum(dim=(-2, -1)).clamp_min(eps)
        bg_frac = w_bg.sum(dim=(-2, -1)) / w_glob.sum(dim=(-2, -1)).clamp_min(eps)
        contrast_gate = (4.0 * fg_frac * bg_frac).clamp(0.0, 1.0)
        diff = (fg - bg).abs() * contrast_gate
        return torch.cat([glob, fg, bg, diff], dim=1)

    def forward(
        self,
        rgb_top: torch.Tensor,
        srm_top: torch.Tensor,
        seg_prob: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = rgb_top.shape[-2:]
        if srm_top.shape[-2:] != grid:
            srm_top = F.interpolate(srm_top, size=grid, mode="bilinear", align_corners=False)

        m = F.interpolate(seg_prob.float(), size=grid, mode="bilinear", align_corners=False).to(rgb_top.dtype)
        if valid_mask is None:
            v = torch.ones_like(m)
        else:
            v = F.interpolate(valid_mask.float(), size=grid, mode="area").clamp(0.0, 1.0).to(rgb_top.dtype)
            v = (v > 0.1).to(rgb_top.dtype)

        v_sum = v.sum(dim=(-2, -1)).clamp_min(1e-5)

        # Expert 1: Pure Global RGB (no segmentation mask gating!)
        rgb_glob = (rgb_top * v).sum(dim=(-2, -1)) / v_sum
        rgb_logits = self.rgb_head(rgb_glob)

        # Expert 2: Pure SRM + Active Mask Forensics (no RGB features!)
        srm_vec = self._masked_stats(srm_top, m, v)
        m_mean = (m * v).sum(dim=(-2, -1)) / v_sum
        m_max = (m * v).amax(dim=(-2, -1))
        m_var = (((m - m_mean[:, :, None, None]) ** 2) * v).sum(dim=(-2, -1)) / v_sum
        m_std = torch.sqrt(m_var.clamp_min(1e-6))
        m_hi = ((m > 0.5).to(m.dtype) * v).sum(dim=(-2, -1)) / v_sum
        mask_stats = torch.cat([m_mean, m_max, m_std, m_hi], dim=1)

        t_srm = self.srm_proj(srm_vec)
        aux_forensic_logits = self.forensic_head(torch.cat([t_srm, mask_stats], dim=1))

        # Bounded Dual-Expert Logit Fusion + small cross-modal residual
        alpha = 0.35 + 0.30 * torch.sigmoid(self.raw_alpha)
        res_scale = 0.25 * torch.tanh(self.raw_res_scale)
        res_in = torch.cat(
            [
                rgb_logits.detach(),
                aux_forensic_logits.detach(),
                (rgb_logits - aux_forensic_logits).abs().detach(),
                mask_stats,
            ],
            dim=1,
        )
        cls_logits = (
            alpha * rgb_logits
            + (1.0 - alpha) * aux_forensic_logits
            + res_scale * torch.tanh(self.res_net(res_in))
        )
        n = rgb_top.shape[0]
        gates = torch.cat([alpha.expand(n, 1), (1.0 - alpha).expand(n, 1)], dim=1)
        return cls_logits, rgb_logits, aux_forensic_logits, gates


class VERITASV4(nn.Module):
    """Unified OW v4 / v4.1 model supporting `baseline`, `veritas_v4`, and `veritas_v4_1`."""

    def __init__(self, variant: str = "veritas_v4_1", pretrained: bool = True, dropout: float = 0.25):
        super().__init__()
        if variant not in ("baseline", "veritas_v4", "veritas_v4_1"):
            raise ValueError("variant must be 'baseline', 'veritas_v4', or 'veritas_v4_1'")
        self.variant = variant
        self.multi_stream = variant in ("veritas_v4", "veritas_v4_1")
        self.segmentation = variant in ("veritas_v4", "veritas_v4_1")

        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = efficientnet_b7(weights=weights).features

        if variant == "baseline":
            self.cls_head = ClassificationHead(2560)
            self.streams = None
            self.srm_encoder = None
            self.freq_encoder = None
            self.seg_head = None
            self.fusion_head = None
        elif variant == "veritas_v4":
            self.cls_head = None
            self.streams = ForensicStreamsV4()
            self.srm_encoder = SpatialForensicEncoderV4(in_ch=9, mid_ch=128, top_ch=256)
            self.freq_encoder = SpatialForensicEncoderV4(in_ch=9, mid_ch=128, top_ch=256)
            self.seg_head = ActiveSegmentationHeadV4(
                high_ch=640, mid_ch=224, low_ch=48, forensic_mid_ch=128, use_forensic=True, num_forensic_streams=2
            )
            self.fusion_head = MaskGuidedContrastiveFusionV4(
                rgb_ch=2560, forensic_ch=256, embed_dim=256, dropout=dropout
            )
        else:  # veritas_v4_1
            self.cls_head = None
            self.streams = SRMStreamV41()
            self.srm_encoder = SpatialForensicEncoderV4(in_ch=9, mid_ch=128, top_ch=256)
            self.freq_encoder = None
            self.seg_head = ActiveSegmentationHeadV4(
                high_ch=640, mid_ch=224, low_ch=48, forensic_mid_ch=128, use_forensic=True, num_forensic_streams=1
            )
            self.fusion_head = DecoupledForensicFusionV41(
                rgb_ch=2560, forensic_ch=256, embed_dim=256, dropout=dropout
            )

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        size = x.shape[-2:]
        low = mid = high = None
        feat = x
        for i, stage in enumerate(self.encoder):
            feat = stage(feat)
            if i == 2:
                low = feat
            elif i == 5:
                mid = feat
            elif i == 7:
                high = feat

        if self.variant == "baseline":
            logits = self.cls_head(feat)
            return {
                "classification_logits": logits,
                "fake_probability": logits.sigmoid(),
            }

        if self.variant == "veritas_v4":
            srm_map, freq_map = self.streams(x, valid_mask)
            srm_mid, srm_top = self.srm_encoder(srm_map)
            freq_mid, freq_top = self.freq_encoder(freq_map)

            seg_logits = self.seg_head(high, mid, low, out_size=size, srm_mid=srm_mid, freq_mid=freq_mid)
            seg_prob = seg_logits.sigmoid()

            cls_logits, aux_forensic_logits, stream_gates = self.fusion_head(
                feat, srm_top, freq_top, seg_prob, valid_mask=valid_mask
            )
            return {
                "classification_logits": cls_logits,
                "fake_probability": cls_logits.sigmoid(),
                "aux_forensic_logits": aux_forensic_logits,
                "segmentation_logits": seg_logits,
                "segmentation_prob": seg_prob,
                "stream_gates": stream_gates,
            }

        # veritas_v4_1
        srm_map = self.streams(x, valid_mask)
        srm_mid, srm_top = self.srm_encoder(srm_map)

        seg_logits = self.seg_head(high, mid, low, out_size=size, srm_mid=srm_mid)
        seg_prob = seg_logits.sigmoid()

        cls_logits, rgb_logits, aux_forensic_logits, stream_gates = self.fusion_head(
            feat, srm_top, seg_prob, valid_mask=valid_mask
        )
        return {
            "classification_logits": cls_logits,
            "fake_probability": cls_logits.sigmoid(),
            "rgb_logits": rgb_logits,
            "aux_forensic_logits": aux_forensic_logits,
            "segmentation_logits": seg_logits,
            "segmentation_prob": seg_prob,
            "stream_gates": stream_gates,
        }
