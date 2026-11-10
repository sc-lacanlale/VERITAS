"""OW v3 candidates and references, with an unchanged direct RGB B7 path.

Local FFT callers should pass a binary rectangular ``valid_mask`` (N,1,H,W)
covering the letterboxed content, including its exact offset after augmentation.
Absent masks mean the entire tensor is content; black pixels never imply padding.
This is a model candidate, not a claim of improved trained accuracy.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

from .architecture import ClassificationHead, DeepLabV3PlusHead, IMAGENET_MEAN, IMAGENET_STD, _srm_kernels
from .architecture_v2 import expand_rgb_stem


IMPLEMENTATION_VERSION = "ow_v3"


class ForensicStreamsV3(nn.Module):
    """Color residuals and optional padding-aware local spectral statistics.

Local FFT uses nonoverlapping 32px patches, patch-mean removal, a Hann taper,
and three radial bands [0,.125), [.125,.25), [.25,sqrt(.5)] cycles/pixel.
Replicate extension fills incomplete edge patches; no zero letterbox enters FFT.
Band powers retain their amplitude via log1p, without per-image min/max scaling.
"""
    def __init__(self, color_srm=False, frequency="global"):
        super().__init__()
        if frequency not in ("global", "local"):
            raise ValueError("frequency must be 'global' or 'local'")
        self.color_srm = bool(color_srm)
        self.frequency = frequency
        self.register_buffer("srm_weight", _srm_kernels())
        self.patch_size = 32
        self.grid_size = 32
        window = torch.hann_window(self.patch_size, periodic=False)
        self.register_buffer("fft_window", window[:, None] * window[None, :])
        fy = torch.fft.fftfreq(self.patch_size)[:, None]
        fx = torch.fft.rfftfreq(self.patch_size)[None, :]
        radius = torch.sqrt(fy.square() + fx.square())
        bands = torch.stack([(radius < 0.125), ((radius >= 0.125) & (radius < 0.25)), (radius >= 0.25)]).float()
        self.register_buffer("fft_bands", bands / bands.sum((-2, -1), keepdim=True))

    @staticmethod
    def content_bounds(rgb, valid_mask):
        if valid_mask is None:
            return [(0, rgb.shape[-2], 0, rgb.shape[-1])] * rgb.shape[0]
        if valid_mask.shape != (rgb.shape[0], 1, *rgb.shape[-2:]):
            raise ValueError("valid_mask must have shape (N,1,H,W) matching RGB")
        if not torch.isfinite(valid_mask).all():
            raise ValueError("valid_mask must be finite and binary")
        if not ((valid_mask == 0) | (valid_mask == 1)).all():
            raise ValueError("valid_mask must be binary")
        result = []
        for mask in valid_mask[:, 0]:
            positions = torch.nonzero(mask, as_tuple=False)
            if positions.numel() == 0:
                raise ValueError("valid_mask must contain nonempty rectangular content")
            y0, x0 = positions.amin(0).tolist()
            y1, x1 = (positions.amax(0) + 1).tolist()
            if int(mask.sum()) != (y1 - y0) * (x1 - x0):
                raise ValueError("valid_mask must describe one filled rectangle")
            result.append((y0, y1, x0, x1))
        return result

    def local_frequency(self, rgb, valid_mask):
        outputs = []
        patch = self.patch_size
        for sample, (y0, y1, x0, x1) in zip(rgb, self.content_bounds(rgb, valid_mask)):
            crop = sample[None, :, y0:y1, x0:x1]
            height, width = crop.shape[-2:]
            crop = F.pad(crop, (0, (-width) % patch, 0, (-height) % patch), mode="replicate")
            patches = crop.unfold(2, patch, patch).unfold(3, patch, patch)
            patches = patches - patches.mean((-2, -1), keepdim=True)
            power = torch.fft.rfft2(patches * self.fft_window, norm="ortho").abs().square()
            band_power = torch.einsum("bcijuv,kuv->bckij", power, self.fft_bands)
            stats = torch.log1p(band_power).flatten(1, 2)
            outputs.append(F.adaptive_avg_pool2d(stats, (self.grid_size, self.grid_size)))
        return torch.cat(outputs, dim=0)

    def forward(self, normalized_rgb, valid_mask=None):
        # CUDA FFT on 600px inputs and all spectral reductions stay float32.
        with torch.autocast(device_type=normalized_rgb.device.type, enabled=False):
            x = normalized_rgb.float()
            rgb = (x * IMAGENET_STD.to(x.device) + IMAGENET_MEAN.to(x.device)).clamp(0, 1)
            if self.color_srm:
                residual = F.conv2d(rgb, self.srm_weight.float().repeat(3, 1, 1, 1), padding=2, groups=3).tanh()
            else:
                gray = (rgb * x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
                residual = F.conv2d(gray, self.srm_weight.float(), padding=2).tanh()
            if self.frequency == "local":
                magnitude = self.local_frequency(rgb, valid_mask)
            else:
                spectrum = torch.fft.fftshift(torch.fft.fft2(rgb, norm="ortho"), dim=(-2, -1))
                magnitude = torch.log1p(spectrum.abs())
                lo = magnitude.amin((-2, -1), keepdim=True)
                span = magnitude.amax((-2, -1), keepdim=True) - lo
                magnitude = (magnitude - lo) / span.clamp_min(1e-6)
                magnitude = magnitude - magnitude.mean((-2, -1), keepdim=True)
        return residual, magnitude


class EvidencePoolV3(nn.Module):
    def __init__(self, mode="avg"):
        super().__init__()
        if mode not in ("avg", "avg_max"):
            raise ValueError("pooling must be 'avg' or 'avg_max'")
        self.mode = mode

    def forward(self, feature):
        average = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        if self.mode == "avg":
            return average
        maximum = F.adaptive_max_pool2d(feature, 1).flatten(1)
        return torch.cat([average, maximum], dim=1)


class AuxiliaryEncoderV3(nn.Sequential):
    def __init__(self, channels=3, pooling="avg"):
        layers = []
        for width in (32, 64, 128):
            layers.extend([nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, width), nn.SiLU()])
            channels = width
        layers.append(EvidencePoolV3(pooling))
        super().__init__(*layers)


class ResidualFusionV3(nn.Sequential):
    """Only the final layer starts at zero; hidden features can learn after step one."""
    def __init__(self, channels, hidden=256):
        super().__init__(nn.LayerNorm(channels), nn.Linear(channels, hidden), nn.SiLU(),
                         nn.Dropout(0.2), nn.Linear(hidden, 1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


def _gn_v3(channels):
    return nn.GroupNorm(math.gcd(32, channels), channels)


class SeparableConvV3(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _gn_v3(out_channels), nn.ReLU(inplace=True))


class SeparableASPPV3(nn.Module):
    def __init__(self, in_channels=224, channels=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False),
                                                    _gn_v3(channels), nn.ReLU(inplace=True))])
        self.branches.extend(SeparableConvV3(in_channels, channels, rate) for rate in rates)
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, channels, 1, bias=False),
                                 _gn_v3(channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(channels * 5, channels, 1, bias=False),
                                     _gn_v3(channels), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        features.append(F.interpolate(self.gap(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(features, dim=1))


class SeparableDeepLabHeadV3(nn.Module):
    """The two 150px decoder blocks cost 3.339G rather than 29.0304G MAC/image.

    These operation counts apply at 600px input and unchanged 304/256 widths,
    and do not predict the whole model's wall-clock speedup or trained accuracy.
    """
    def __init__(self, high_channels=224, low_channels=48):
        super().__init__()
        self.aspp = SeparableASPPV3(high_channels)
        self.low_proj = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                      _gn_v3(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(SeparableConvV3(304, 256), SeparableConvV3(256, 256), nn.Conv2d(256, 1, 1))

    def forward(self, high, low, out_size):
        high = F.interpolate(self.aspp(high), size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([high, self.low_proj(low)], dim=1))
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class VERITASV3(nn.Module):
    def __init__(self, multi_stream=False, segmentation=False, pretrained=True, fusion="rgb",
                 color_srm=False, frequency="global", pooling="avg", decoder="dense"):
        super().__init__()
        if fusion not in ("rgb", "early", "late_additive", "late_nonlinear") or bool(multi_stream) != (fusion != "rgb"):
            raise ValueError("RGB-only requires fusion='rgb'; multistream requires an early/late fusion mode")
        if frequency not in ("global", "local") or pooling not in ("avg", "avg_max") or decoder not in ("dense", "separable"):
            raise ValueError("Unsupported frequency, pooling, or decoder mode")
        if fusion == "early" and (color_srm or frequency != "global"):
            raise ValueError("The corrected early reference retains grayscale SRM/global FFT and 9 input channels")
        self.multi_stream = bool(multi_stream)
        self.segmentation = bool(segmentation)
        self.fusion, self.pooling = fusion, pooling
        self.color_srm, self.frequency, self.decoder = bool(color_srm), frequency, decoder
        self.requires_valid_mask = self.multi_stream and frequency == "local"
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = efficientnet_b7(weights=weights).features
        self.cls_head = ClassificationHead(2560)
        if self.segmentation:
            self.seg_head = DeepLabV3PlusHead(224, 48, 1) if decoder == "dense" else SeparableDeepLabHeadV3()
        else:
            self.seg_head = None
        self.streams = ForensicStreamsV3(color_srm, frequency) if self.multi_stream else None
        self.evidence_pool = EvidencePoolV3(pooling)
        multiplier = 2 if pooling == "avg_max" else 1
        if fusion == "early":
            self.encoder[0][0] = expand_rgb_stem(self.encoder[0][0])
        elif fusion.startswith("late"):
            self.srm_encoder = AuxiliaryEncoderV3(9 if color_srm else 3, pooling)
            self.fft_encoder = AuxiliaryEncoderV3(9 if frequency == "local" else 3, pooling)
            if fusion == "late_additive":
                self.aux_projection = nn.Linear(256 * multiplier, 2560, bias=False)
                nn.init.zeros_(self.aux_projection.weight)
            else:
                self.fusion_head = ResidualFusionV3((2560 + 256) * multiplier)
        if pooling == "avg_max" and fusion != "late_nonlinear":
            self.local_head = ResidualFusionV3(2560 * 2)

    def forward(self, x, valid_mask=None):
        size = x.shape[-2:]
        residual = magnitude = None
        if self.multi_stream:
            residual, magnitude = self.streams(x, valid_mask)
        if self.fusion == "early":
            x = torch.cat([x, residual, magnitude], dim=1)
        low = high = None
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if i == 2:
                low = x
            elif i == 5:
                high = x
        rgb_average = self.cls_head.pool(x).flatten(1)
        rgb_evidence = self.evidence_pool(x)
        if self.fusion == "late_additive":
            auxiliary = torch.cat([self.srm_encoder(residual), self.fft_encoder(magnitude)], dim=1)
            logits = self.cls_head.fc(rgb_average + self.aux_projection(auxiliary))
        else:
            logits = self.cls_head.fc(rgb_average)
            if self.fusion == "late_nonlinear":
                auxiliary = torch.cat([self.srm_encoder(residual), self.fft_encoder(magnitude)], dim=1)
                logits = logits + self.fusion_head(torch.cat([rgb_evidence, auxiliary], dim=1))
        if hasattr(self, "local_head"):
            logits = logits + self.local_head(rgb_evidence)
        out = {"classification_logits": logits, "fake_probability": logits.sigmoid()}
        if self.segmentation:
            seg_logits = self.seg_head(high, low, out_size=size)
            out.update(segmentation_logits=seg_logits, segmentation_prob=seg_logits.sigmoid())
        return out
