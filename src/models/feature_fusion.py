"""
Feature fusion module for VERITAS.

Fuses RGB, noise residual, and frequency-domain representations
into a unified feature representation for EfficientNet-B7.

Requirements: 7
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class FeatureFusion(nn.Module):
    """
    Feature fusion module that combines multi-representation features.
    
    Concatenates RGB, noise residual, and frequency representations along
    the channel dimension (9 channels total), then projects to 3 channels
    to match EfficientNet-B7 input requirements.
    
    Architecture:
        Input: 3 representations × 3 channels = 9 channels
        1×1 Conv: 9 → 3 channels
        Batch Normalization
        Output: 3-channel fused representation [B, 3, H, W]
    """
    
    def __init__(self, config=None):
        """
        Initialize feature fusion module.
        
        Args:
            config: Optional configuration object (for future extensibility)
        """
        super().__init__()
        
        self.config = config
        
        # 1×1 convolution to project 9 channels → 3 channels
        # This learns optimal weighting of the three representations
        self.conv1x1 = nn.Conv2d(
            in_channels=9,   # RGB (3) + Noise (3) + Frequency (3)
            out_channels=3,  # Match EfficientNet-B7 input
            kernel_size=1,
            bias=False
        )
        
        # Batch normalization for stable training
        self.bn = nn.BatchNorm2d(3)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info("Initialized FeatureFusion module (9 → 3 channels)")
    
    def _initialize_weights(self):
        """
        Initialize convolution weights.
        
        Start with equal weighting of all representations
        (each representation contributes 1/3).
        """
        # Initialize conv1x1 to give equal weight to each representation
        with torch.no_grad():
            # Each output channel receives equal contribution from
            # corresponding channels of all 3 representations
            weight = torch.zeros(3, 9, 1, 1)
            
            # For each output channel (0, 1, 2):
            for i in range(3):
                # Contribution from RGB channel i
                weight[i, i, 0, 0] = 1.0 / 3.0
                # Contribution from noise channel i
                weight[i, i + 3, 0, 0] = 1.0 / 3.0
                # Contribution from frequency channel i
                weight[i, i + 6, 0, 0] = 1.0 / 3.0
            
            self.conv1x1.weight.copy_(weight)
    
    def forward(self, representations: dict) -> torch.Tensor:
        """
        Fuse multi-representation features.
        
        Args:
            representations: Dictionary with keys:
                - 'rgb': RGB representation [B, 3, H, W]
                - 'noise': Noise residual representation [B, 3, H, W]
                - 'frequency': Frequency-domain representation [B, 3, H, W]
        
        Returns:
            Fused representation [B, 3, H, W]
        
        Raises:
            ValueError: If input shapes don't match or required keys are missing
        """
        # Validate inputs
        required_keys = ['rgb', 'noise', 'frequency']
        for key in required_keys:
            if key not in representations:
                raise ValueError(f"Missing required representation: {key}")
        
        rgb = representations['rgb']
        noise = representations['noise']
        frequency = representations['frequency']
        
        # Validate shapes
        if not (rgb.shape == noise.shape == frequency.shape):
            raise ValueError(
                f"Representation shapes must match. Got: "
                f"RGB={rgb.shape}, Noise={noise.shape}, Frequency={frequency.shape}"
            )
        
        if rgb.shape[1] != 3:
            raise ValueError(
                f"Each representation must have 3 channels. Got: {rgb.shape[1]}"
            )
        
        # Concatenate along channel dimension
        # [B, 3, H, W] + [B, 3, H, W] + [B, 3, H, W] → [B, 9, H, W]
        concatenated = torch.cat([rgb, noise, frequency], dim=1)
        
        # Project to 3 channels using 1×1 convolution
        # [B, 9, H, W] → [B, 3, H, W]
        fused = self.conv1x1(concatenated)
        
        # Apply batch normalization
        fused = self.bn(fused)
        
        return fused
    
    def get_fusion_weights(self) -> torch.Tensor:
        """
        Get current fusion weights for analysis.
        
        Returns:
            Convolution weights [3, 9, 1, 1]
        """
        return self.conv1x1.weight.data
    
    def visualize_weights(self) -> dict:
        """
        Visualize contribution of each representation to final output.
        
        Returns:
            Dictionary with contribution percentages for each representation
        """
        weights = self.get_fusion_weights().squeeze()  # [3, 9]
        
        # Sum absolute weights for each representation
        rgb_contribution = weights[:, 0:3].abs().sum().item()
        noise_contribution = weights[:, 3:6].abs().sum().item()
        frequency_contribution = weights[:, 6:9].abs().sum().item()
        
        total = rgb_contribution + noise_contribution + frequency_contribution
        
        return {
            'rgb_percent': 100 * rgb_contribution / total,
            'noise_percent': 100 * noise_contribution / total,
            'frequency_percent': 100 * frequency_contribution / total
        }


class AdaptiveFeatureFusion(nn.Module):
    """
    Advanced feature fusion with learnable attention weights.
    
    This is an alternative to simple concatenation-based fusion.
    Can be used for ablation studies or future improvements.
    """
    
    def __init__(self, config=None):
        """
        Initialize adaptive fusion module.
        
        Args:
            config: Optional configuration object
        """
        super().__init__()
        
        self.config = config
        
        # Learnable attention weights for each representation
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Per-representation projection
        self.rgb_proj = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self.noise_proj = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self.frequency_proj = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        
        # Final batch normalization
        self.bn = nn.BatchNorm2d(3)
        
        logger.info("Initialized AdaptiveFeatureFusion module with learnable attention")
    
    def forward(self, representations: dict) -> torch.Tensor:
        """
        Fuse features with adaptive attention weighting.
        
        Args:
            representations: Dictionary with 'rgb', 'noise', 'frequency' keys
        
        Returns:
            Fused representation [B, 3, H, W]
        """
        rgb = representations['rgb']
        noise = representations['noise']
        frequency = representations['frequency']
        
        # Project each representation
        rgb_proj = self.rgb_proj(rgb)
        noise_proj = self.noise_proj(noise)
        frequency_proj = self.frequency_proj(frequency)
        
        # Normalize attention weights using softmax
        weights = torch.softmax(self.attention_weights, dim=0)
        
        # Weighted combination
        fused = (
            weights[0] * rgb_proj +
            weights[1] * noise_proj +
            weights[2] * frequency_proj
        )
        
        # Batch normalization
        fused = self.bn(fused)
        
        return fused
    
    def get_attention_weights(self) -> dict:
        """
        Get current attention weights.
        
        Returns:
            Dictionary with attention weights for each representation
        """
        weights = torch.softmax(self.attention_weights, dim=0)
        
        return {
            'rgb_weight': weights[0].item(),
            'noise_weight': weights[1].item(),
            'frequency_weight': weights[2].item()
        }


# Export main classes
__all__ = ['FeatureFusion', 'AdaptiveFeatureFusion']
