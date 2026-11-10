"""Loss and checkpoint safeguards shared by the versioned OW notebooks."""
import torch
from torch import nn


class VERITASLossV2(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_seg=0.25, dice_weight=1.0, segmentation=False):
        super().__init__()
        if lambda_cls <= 0 or lambda_seg < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be nonnegative, with positive classification weight")
        self.lambda_cls = float(lambda_cls)
        self.lambda_seg = float(lambda_seg)
        self.dice_weight = float(dice_weight)
        self.segmentation = bool(segmentation)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels, masks=None):
        cls = self.bce(outputs["classification_logits"], labels)
        seg_bce = seg_dice = cls.new_zeros(())
        if self.segmentation:
            if masks is None:
                raise ValueError("Segmentation targets are required")
            logits = outputs["segmentation_logits"]
            seg_bce = self.bce(logits, masks)
            # 600x600 reductions exceed float16's range under CUDA autocast.
            probs = logits.float().sigmoid()
            masks = masks.float()
            dims = (1, 2, 3)
            intersection = (probs * masks).sum(dims)
            denominator = probs.sum(dims) + masks.sum(dims)
            seg_dice = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        # Average component losses; lambda_seg then controls the auxiliary task.
        seg = (seg_bce + self.dice_weight * seg_dice) / (1 + self.dice_weight)
        total = self.lambda_cls * cls + self.lambda_seg * seg
        return dict(total=total, classification=cls.detach(), segmentation=seg.detach(),
                    segmentation_bce=seg_bce.detach(), segmentation_dice=seg_dice.detach())


def validate_resume_checkpoint(checkpoint, experiment, exp_cfg, protocol_hash):
    if checkpoint.get("experiment_name") != experiment:
        raise ValueError("Checkpoint belongs to a different experiment")
    if checkpoint.get("exp_cfg") != exp_cfg:
        raise ValueError("Checkpoint architecture/version differs; start a fresh experiment")
    if checkpoint.get("protocol_hash") != protocol_hash:
        raise ValueError("Checkpoint protocol differs (including smoke/full or mask source)")
