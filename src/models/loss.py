"""
Multi-task loss function for VERITAS.

Requirements: 12
"""

import torch
import torch.nn as nn
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and segmentation losses.
    
    Loss formula:
    L_seg = 0.5 * BCE + 0.5 * Dice
    L_total = cls_weight * L_classification + seg_weight * L_segmentation
    
    Default weights: classification=0.4, segmentation=0.6
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Get loss weights from config
        self.cls_weight = config.get('loss.classification_weight', 0.4)
        self.seg_weight = config.get('loss.segmentation_weight', 0.6)
        self.seg_bce_weight = config.get('loss.segmentation_bce_weight', 0.5)
        self.seg_dice_weight = config.get('loss.segmentation_dice_weight', 0.5)
        
        # BCE loss
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-7
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred_logits: Prediction logits [B, 1, H, W]
            target: Target masks [B, 1, H, W]
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice loss (1 - Dice coefficient)
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred_logits)
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute Dice coefficient
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # Return 1 - Dice as loss
        return 1.0 - dice
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary with:
                - 'classification_logit': [B, 1]
                - 'segmentation_logits': [B, 1, H, W]
            targets: Dictionary with:
                - 'label': [B, 1]
                - 'mask': [B, 1, H, W]
                
        Returns:
            Dictionary with:
                - 'total_loss': Combined loss
                - 'classification_loss': Classification BCE loss
                - 'segmentation_loss': Combined segmentation loss
        """
        # Classification loss
        cls_loss = self.bce(
            predictions['classification_logit'],
            targets['label']
        )
        
        # Segmentation BCE loss
        seg_bce = self.bce(
            predictions['segmentation_logits'],
            targets['mask']
        )
        
        # Segmentation Dice loss
        seg_dice = self.dice_loss(
            predictions['segmentation_logits'],
            targets['mask']
        )
        
        # Combined segmentation loss
        seg_loss = self.seg_bce_weight * seg_bce + self.seg_dice_weight * seg_dice
        
        # Total loss
        total_loss = self.cls_weight * cls_loss + self.seg_weight * seg_loss
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'segmentation_loss': seg_loss,
            'segmentation_bce_loss': seg_bce,
            'segmentation_dice_loss': seg_dice
        }
