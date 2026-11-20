"""Versioned OW models: corrected early fusion and an experimental late fusion.

Legacy checkpoints continue to use architecture.py. These models require fresh
training and record implementation_version='ow_v2' in their checkpoints.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

from .architecture import ClassificationHead, DeepLabV3PlusHead, IMAGENET_MEAN, IMAGENET_STD, _srm_kernels


IMPLEMENTATION_VERSION = "ow_v2"


class ForensicStreams(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("srm_weight", _srm_kernels())

    def forward(self, normalized_rgb):
        # FFT on 600px images must stay float32 even under CUDA autocast.
        with torch.autocast(device_type=normalized_rgb.device.type, enabled=False):
            x = normalized_rgb.float()
            mean = IMAGENET_MEAN.to(x.device)
            std = IMAGENET_STD.to(x.device)
            rgb = (x * std + mean).clamp(0, 1)
            gray = (rgb * x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
            residual = F.conv2d(gray, self.srm_weight.float(), padding=2).tanh()
            spectrum = torch.fft.fftshift(torch.fft.fft2(rgb, norm="ortho"), dim=(-2, -1))
            magnitude = torch.log1p(spectrum.abs())
            lo = magnitude.amin(dim=(-2, -1), keepdim=True)
            span = magnitude.amax(dim=(-2, -1), keepdim=True) - lo
            magnitude = (magnitude - lo) / span.clamp_min(1e-6)
            magnitude = magnitude - magnitude.mean(dim=(-2, -1), keepdim=True)
        return residual, magnitude


def expand_rgb_stem(conv3):
    """Preserve the RGB response exactly; learn the auxiliary channels from zero."""
    conv9 = nn.Conv2d(9, conv3.out_channels, conv3.kernel_size, stride=conv3.stride,
                      padding=conv3.padding, dilation=conv3.dilation,
                      bias=conv3.bias is not None, device=conv3.weight.device, dtype=conv3.weight.dtype)
    with torch.no_grad():
        conv9.weight.zero_()
        conv9.weight[:, :3].copy_(conv3.weight)
        if conv3.bias is not None:
            conv9.bias.copy_(conv3.bias)
    return conv9


class AuxiliaryEncoder(nn.Sequential):
    def __init__(self):
        layers = []
        channels = 3
        for width in (32, 64, 128):
            layers.extend([nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, width), nn.SiLU()])
            channels = width
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(1)])
        super().__init__(*layers)


class VERITASV2(nn.Module):
    def __init__(self, multi_stream=True, segmentation=True, pretrained=True, fusion="early"):
        super().__init__()
        if fusion not in ("rgb", "early", "late") or multi_stream != (fusion != "rgb"):
            raise ValueError("Use fusion='rgb' for RGB-only models, otherwise 'early' or 'late'")
        self.multi_stream = bool(multi_stream)
        self.segmentation = bool(segmentation)
        self.fusion = fusion
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = efficientnet_b7(weights=weights).features
        # Create the common RGB backbone/head before variant-specific parameters.
        self.cls_head = ClassificationHead(2560)
        self.seg_head = DeepLabV3PlusHead(224, 48, 1) if segmentation else None
        self.streams = ForensicStreams() if multi_stream else None
        if fusion == "early":
            self.encoder[0][0] = expand_rgb_stem(self.encoder[0][0])
        elif fusion == "late":
            self.srm_encoder = AuxiliaryEncoder()
            self.fft_encoder = AuxiliaryEncoder()
            self.aux_projection = nn.Linear(256, 2560, bias=False)
            nn.init.zeros_(self.aux_projection.weight)

    def forward(self, x):
        size = x.shape[-2:]
        residual = magnitude = None
        if self.multi_stream:
            residual, magnitude = self.streams(x)
        if self.fusion == "early":
            # Keep the original ImageNet-normalized RGB, unlike the legacy stem.
            x = torch.cat([x, residual, magnitude], dim=1)
        low = high = None
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if i == 2:
                low = x
            elif i == 5:
                high = x
        if self.fusion == "late":
            # Pool independent domains before fusion: frequency coordinates are
            # never treated as image locations by the segmentation decoder.
            auxiliary = torch.cat([self.srm_encoder(residual), self.fft_encoder(magnitude)], dim=1)
            pooled = self.cls_head.pool(x).flatten(1) + self.aux_projection(auxiliary)
            logits = self.cls_head.fc(pooled)
        else:
            logits = self.cls_head(x)
        out = {"classification_logits": logits, "fake_probability": torch.sigmoid(logits)}
        if self.segmentation:
            seg_logits = self.seg_head(high, low, out_size=size)
            out.update(segmentation_logits=seg_logits, segmentation_prob=torch.sigmoid(seg_logits))
        return out
