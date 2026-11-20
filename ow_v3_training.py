"""OW v3 supervision and optimization helpers, embedded into Kaggle notebooks.

Real class labels establish zero manipulation targets. Fake localization remains
teacher supervision, not independent ground truth or calibrated confidence.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _binary_tensor(value, name):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if not torch.isfinite(value).all() or ((value != 0) & (value != 1)).any():
        raise ValueError(f"{name} must contain only finite binary values 0 and 1")


class VERITASLossV3(nn.Module):
    """Class BCE plus label-consistent, empty-aware auxiliary segmentation.

    BCE is averaged over valid canvas pixels separately for each trusted image.
    A Real image contributes its full BCE. A trusted nonempty Fake contributes
    (BCE + dice_weight * foreground_Dice) / (1 + dice_weight). Those per-image
    losses are averaged, keeping each trusted image's total weight equal.

    ``seg_valid`` can exclude unreliable Fake masks; it never overrides known
    Real negatives. Mask area determines target availability, not confidence.
    Empty Fake targets are unavailable localization supervision and are skipped.
    """

    def __init__(self, lambda_cls=1.0, lambda_seg=0.25, dice_weight=1.0, segmentation=False):
        super().__init__()
        values = (float(lambda_cls), float(lambda_seg), float(dice_weight))
        if not all(math.isfinite(value) for value in values) or values[0] <= 0 or min(values[1:]) < 0:
            raise ValueError("Loss weights must be finite and nonnegative, with positive classification weight")
        self.lambda_cls, self.lambda_seg, self.dice_weight = values
        self.segmentation = bool(segmentation)

    def forward(self, outputs, labels, masks=None, valid_mask=None, seg_valid=None):
        cls_logits = outputs["classification_logits"]
        if cls_logits.ndim != 2 or cls_logits.shape[1] != 1 or cls_logits.shape[0] == 0:
            raise ValueError("Classification logits must have nonempty shape [N, 1]")
        _binary_tensor(labels, "labels")
        if labels.shape != cls_logits.shape:
            raise ValueError("Labels must have the same [N, 1] shape as classification logits")
        labels = labels.to(device=cls_logits.device, dtype=torch.float32)
        cls = F.binary_cross_entropy_with_logits(cls_logits.float(), labels)
        seg = seg_bce = seg_dice = cls.new_zeros(())
        counts = {key: cls.new_zeros(()) for key in (
            "segmentation_valid_samples", "segmentation_positive_samples",
            "segmentation_ignored_fake_samples", "segmentation_valid_pixels",
        )}
        if self.segmentation:
            if masks is None:
                raise ValueError("Segmentation targets are required")
            logits = outputs["segmentation_logits"]
            if logits.ndim != 4 or logits.shape[:2] != (len(labels), 1):
                raise ValueError("Segmentation logits must have shape [N, 1, H, W]")
            if masks.shape != logits.shape:
                raise ValueError("Masks must have the same shape as segmentation logits")
            _binary_tensor(masks, "masks")
            targets = masks.to(device=logits.device, dtype=torch.float32)
            real = labels[:, 0].to(logits.device) == 0
            # Preserve the original teacher PNG; correct targets only in memory.
            targets = torch.where(real[:, None, None, None], torch.zeros_like(targets), targets)
            if valid_mask is None:
                valid = torch.ones_like(targets)
            else:
                if valid_mask.shape != targets.shape:
                    raise ValueError("valid_mask must have the same shape as segmentation targets")
                _binary_tensor(valid_mask, "valid_mask")
                valid = valid_mask.to(device=logits.device, dtype=torch.float32)
            if seg_valid is None:
                reliability = torch.ones(len(labels), dtype=torch.bool, device=logits.device)
            else:
                if seg_valid.shape not in ((len(labels),), (len(labels), 1)):
                    raise ValueError("seg_valid must have shape [N] or [N, 1]")
                _binary_tensor(seg_valid, "seg_valid")
                reliability = seg_valid.reshape(-1).to(device=logits.device, dtype=torch.bool)
            dims = (1, 2, 3)
            pixel_counts = valid.sum(dims)
            nonempty = (targets * valid).sum(dims) > 0
            trusted = (pixel_counts > 0) & (real | (reliability & nonempty))
            positive = trusted & ~real & nonempty
            trusted_weight, positive_weight = trusted.float(), positive.float()
            # Float32 also protects 600px reductions inside CUDA autocast.
            seg_logits = logits.float()
            per_bce = (F.binary_cross_entropy_with_logits(seg_logits, targets, reduction="none") * valid).sum(dims)
            per_bce = per_bce / pixel_counts.clamp_min(1)
            probs = seg_logits.sigmoid() * valid
            foreground = targets * valid
            per_dice = 1 - ((2 * (probs * foreground).sum(dims) + 1e-6) /
                            (probs.sum(dims) + foreground.sum(dims) + 1e-6))
            per_image = torch.where(positive, (per_bce + self.dice_weight * per_dice) /
                                    (1 + self.dice_weight), per_bce)
            # Weighted sums remain differentiable zero when no target is trusted.
            seg = (per_image * trusted_weight).sum() / trusted_weight.sum().clamp_min(1)
            seg_bce = (per_bce * trusted_weight).sum() / trusted_weight.sum().clamp_min(1)
            seg_dice = (per_dice * positive_weight).sum() / positive_weight.sum().clamp_min(1)
            counts = dict(segmentation_valid_samples=trusted_weight.sum(),
                          segmentation_positive_samples=positive_weight.sum(),
                          segmentation_ignored_fake_samples=(~real & ~trusted).float().sum(),
                          segmentation_valid_pixels=(pixel_counts * trusted_weight).sum())
        total = self.lambda_cls * cls + self.lambda_seg * seg
        return dict(total=total, classification=cls.detach(), segmentation=seg.detach(),
                    segmentation_bce=seg_bce.detach(), segmentation_dice=seg_dice.detach(),
                    **{key: value.detach() for key, value in counts.items()})


def validate_resume_checkpoint(checkpoint, experiment, exp_cfg, protocol_hash):
    if checkpoint.get("experiment_name") != experiment:
        raise ValueError("Checkpoint belongs to a different experiment")
    if checkpoint.get("exp_cfg") != exp_cfg:
        raise ValueError("Checkpoint architecture/version differs; start a fresh experiment")
    if checkpoint.get("protocol_hash") != protocol_hash:
        raise ValueError("Checkpoint protocol differs (including smoke/full or mask source)")


def build_optimizer(model, learning_rate, weight_decay, backbone_lr_multiplier=0.1):
    """Use a smaller LR for pretrained encoder parameters, with exact coverage."""
    learning_rate, weight_decay, multiplier = map(float, (learning_rate, weight_decay, backbone_lr_multiplier))
    if not all(math.isfinite(value) for value in (learning_rate, weight_decay, multiplier)):
        raise ValueError("Optimizer settings must be finite")
    if learning_rate <= 0 or weight_decay < 0 or multiplier <= 0:
        raise ValueError("Learning rates must be positive and weight decay nonnegative")
    if not isinstance(getattr(model, "encoder", None), nn.Module):
        raise ValueError("Model must expose its pretrained encoder as model.encoder")
    backbone = [p for p in model.encoder.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    groups = []
    if backbone:
        groups.append(dict(params=backbone, lr=learning_rate * multiplier, name="backbone"))
    if heads:
        groups.append(dict(params=heads, lr=learning_rate, name="heads"))
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(optimizer, total_updates, warmup_fraction=0.05):
    """Step after each successful optimizer update; save/load its state on resume."""
    if isinstance(total_updates, bool) or int(total_updates) != total_updates or total_updates <= 0:
        raise ValueError("total_updates must be a positive integer")
    total_updates = int(total_updates)
    fraction = float(warmup_fraction)
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("warmup_fraction must be finite and in [0, 1)")
    warmup_updates = min(max(int(round(total_updates * fraction)), 1), total_updates) if fraction else 0

    def factor(update):
        if update < warmup_updates:
            return (update + 1) / warmup_updates
        progress = min(max((update - warmup_updates) / max(total_updates - warmup_updates, 1), 0), 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def freeze_backbone_bn(model):
    """Freeze encoder BN statistics after model.train(); keep affine gradients."""
    if not isinstance(getattr(model, "encoder", None), nn.Module):
        raise ValueError("Model must expose its pretrained encoder as model.encoder")
    for module in model.encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
