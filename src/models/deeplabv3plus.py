"""
DeepLabV3+ decoder for VERITAS segmentation head.

Implements the DeepLabV3+ decoder with skip connections from low-level features
to recover spatial details lost in the encoder/backbone.

Requirements: 11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .aspp import ASPP

logger = logging.getLogger(__name__)


class DeepLabV3PlusDecoder(nn.Module):
    """
    DeepLabV3+ decoder for semantic segmentation.
    
    Architecture:
        Low-level features (from backbone block 2)
            ↓
        1×1 conv to reduce channels (48 → 48)
            ↓
        ┌───────────────────────────────────┐
        │                                   │
        │   High-level features (ASPP)      │
        │            ↓                      │
        │   Upsample 4× (bilinear)         │
        │            ↓                      │
        └──────► Concatenate ◄──────────────┘
                    ↓
            3×3 conv (256 channels)
                    ↓
            3×3 conv (256 channels)
                    ↓
            1×1 conv (num_classes)
                    ↓
        Upsample 4× to input resolution
                    ↓
            Segmentation logits
    
    Reference: DeepLabV3+ paper (Chen et al., 2018)
    """
    
    def __init__(
        self,
        low_level_channels: int = 48,
        low_level_out_channels: int = 48,
        aspp_out_channels: int = 256,
        decoder_channels: int = 256,
        num_classes: int = 1,
        output_size: int = 600
    ):
        """
        Initialize DeepLabV3+ decoder.
        
        Args:
            low_level_channels: Number of channels in low-level features
                               (48 for EfficientNet-B7 block 2)
            low_level_out_channels: Number of output channels after processing
                                   low-level features
            aspp_out_channels: Number of channels from ASPP module (256)
            decoder_channels: Number of channels in decoder convolutions
            num_classes: Number of output classes (1 for binary segmentation)
            output_size: Output spatial size (600×600)
        """
        super().__init__()
        
        self.output_size = output_size
        self.num_classes = num_classes
        
        # Process low-level features
        # Reduce channels to keep parameters reasonable
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(
                low_level_channels,
                low_level_out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(low_level_out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Decoder convolutions
        # Input: ASPP features (256) + low-level features (48) = 304 channels
        decoder_in_channels = aspp_out_channels + low_level_out_channels
        
        self.decoder = nn.Sequential(
            # First 3×3 conv
            nn.Conv2d(
                decoder_in_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            
            # Second 3×3 conv
            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            
            # Optional: Add dropout for regularization
            nn.Dropout(0.1)
        )
        
        # Final 1×1 conv to produce class predictions
        self.classifier = nn.Conv2d(
            decoder_channels,
            num_classes,
            kernel_size=1
        )
        
        logger.info(
            f"Initialized DeepLabV3PlusDecoder "
            f"(low_level={low_level_channels}, aspp={aspp_out_channels}, "
            f"decoder={decoder_channels}, num_classes={num_classes})"
        )
    
    def forward(
        self,
        low_level_features: torch.Tensor,
        high_level_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            low_level_features: Low-level features from backbone [B, 48, H/4, W/4]
            high_level_features: High-level features from ASPP [B, 256, H/32, W/32]
        
        Returns:
            Segmentation logits [B, num_classes, output_size, output_size]
        """
        # Process low-level features
        # [B, 48, H/4, W/4] → [B, 48, H/4, W/4]
        low_level = self.low_level_conv(low_level_features)
        
        # Upsample ASPP features to match low-level feature resolution
        # [B, 256, H/32, W/32] → [B, 256, H/4, W/4]
        high_level_upsampled = F.interpolate(
            high_level_features,
            size=low_level.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Concatenate along channel dimension
        # [B, 256, H/4, W/4] + [B, 48, H/4, W/4] → [B, 304, H/4, W/4]
        concatenated = torch.cat([high_level_upsampled, low_level], dim=1)
        
        # Apply decoder convolutions
        # [B, 304, H/4, W/4] → [B, 256, H/4, W/4]
        decoded = self.decoder(concatenated)
        
        # Apply classifier to get class logits
        # [B, 256, H/4, W/4] → [B, num_classes, H/4, W/4]
        logits = self.classifier(decoded)
        
        # Upsample to output resolution
        # [B, num_classes, H/4, W/4] → [B, num_classes, output_size, output_size]
        output = F.interpolate(
            logits,
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=False
        )
        
        return output


class DeepLabV3PlusHead(nn.Module):
    """
    Complete DeepLabV3+ segmentation head combining ASPP and decoder.
    
    This module integrates ASPP (multi-scale feature extraction) with
    the DeepLabV3+ decoder (skip connections and upsampling).
    """
    
    def __init__(
        self,
        backbone_high_channels: int = 2560,
        backbone_low_channels: int = 48,
        aspp_out_channels: int = 256,
        num_classes: int = 1,
        output_size: int = 600
    ):
        """
        Initialize complete segmentation head.
        
        Args:
            backbone_high_channels: Number of channels in backbone high-level
                                   features (2560 for EfficientNet-B7)
            backbone_low_channels: Number of channels in backbone low-level
                                  features (48 for EfficientNet-B7 block 2)
            aspp_out_channels: Number of output channels from ASPP (256)
            num_classes: Number of output classes (1 for binary segmentation)
            output_size: Output spatial size (600×600)
        """
        super().__init__()
        
        # ASPP module for multi-scale context
        self.aspp = ASPP(
            in_channels=backbone_high_channels,
            out_channels=aspp_out_channels
        )
        
        # DeepLabV3+ decoder with skip connections
        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=backbone_low_channels,
            low_level_out_channels=48,
            aspp_out_channels=aspp_out_channels,
            decoder_channels=256,
            num_classes=num_classes,
            output_size=output_size
        )
        
        logger.info("Initialized DeepLabV3PlusHead (ASPP + Decoder)")
    
    def forward(
        self,
        high_level_features: torch.Tensor,
        low_level_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through segmentation head.
        
        Args:
            high_level_features: High-level features from backbone [B, 2560, H/32, W/32]
            low_level_features: Low-level features from backbone [B, 48, H/4, W/4]
        
        Returns:
            Segmentation logits [B, num_classes, output_size, output_size]
        """
        # Apply ASPP to extract multi-scale features
        # [B, 2560, H/32, W/32] → [B, 256, H/32, W/32]
        aspp_features = self.aspp(high_level_features)
        
        # Apply decoder with skip connections
        # [B, 256, H/32, W/32] + [B, 48, H/4, W/4] → [B, 1, 600, 600]
        segmentation_logits = self.decoder(low_level_features, aspp_features)
        
        return segmentation_logits


class LightweightDecoderHead(nn.Module):
    """
    Lightweight segmentation decoder for faster inference.
    
    Uses fewer channels and simpler architecture compared to full DeepLabV3+.
    Can be used for ablation studies or deployment constraints.
    """
    
    def __init__(
        self,
        backbone_channels: int = 2560,
        decoder_channels: int = 128,
        num_classes: int = 1,
        output_size: int = 600
    ):
        """
        Initialize lightweight decoder.
        
        Args:
            backbone_channels: Number of channels from backbone
            decoder_channels: Number of channels in decoder
            num_classes: Number of output classes
            output_size: Output spatial size
        """
        super().__init__()
        
        self.output_size = output_size
        
        # Simple decoder without ASPP
        self.decoder = nn.Sequential(
            # Reduce channels
            nn.Conv2d(backbone_channels, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            
            # Refine features
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            
            # Classifier
            nn.Conv2d(decoder_channels, num_classes, 1)
        )
        
        logger.info("Initialized LightweightDecoderHead")
    
    def forward(
        self,
        high_level_features: torch.Tensor,
        low_level_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass (ignores low-level features).
        
        Args:
            high_level_features: High-level features from backbone
            low_level_features: Ignored (for interface compatibility)
        
        Returns:
            Segmentation logits
        """
        # Decode
        logits = self.decoder(high_level_features)
        
        # Upsample to output size
        output = F.interpolate(
            logits,
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=False
        )
        
        return output


# Export main classes
__all__ = [
    'DeepLabV3PlusDecoder',
    'DeepLabV3PlusHead',
    'LightweightDecoderHead'
]
