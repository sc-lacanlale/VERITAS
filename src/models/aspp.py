"""
ASPP (Atrous Spatial Pyramid Pooling) module for VERITAS.

Implements multi-scale contextual feature extraction using atrous convolutions
with multiple dilation rates as specified in the DeepLabV3+ paper.

Requirements: 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ASPPConv(nn.Module):
    """
    Atrous convolution with batch normalization and ReLU activation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        """
        Initialize ASPP convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dilation: Dilation rate for atrous convolution
        """
        super().__init__()
        
        # Atrous convolution with specified dilation
        # Padding = dilation to maintain spatial dimensions
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPooling(nn.Module):
    """
    Global average pooling branch of ASPP.
    
    Captures global context by pooling entire feature map to a single value
    per channel, then upsampling back to original spatial dimensions.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize ASPP pooling branch.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 1×1 convolution to adjust channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, C, H, W]
        
        Returns:
            Pooled features upsampled to input size [B, out_channels, H, W]
        """
        size = x.shape[-2:]  # Save spatial dimensions (H, W)
        
        # Global pooling: [B, C, H, W] → [B, C, 1, 1]
        x = self.global_pool(x)
        
        # 1×1 conv: [B, C, 1, 1] → [B, out_channels, 1, 1]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # Upsample back to original size: [B, out_channels, 1, 1] → [B, out_channels, H, W]
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    
    Extracts multi-scale contextual features using parallel atrous convolutions
    with different dilation rates (1, 6, 12, 18) plus a global pooling branch.
    
    Architecture:
        Input features
            ↓
        ┌──────┬──────┬──────┬──────┬────────┐
        │ 1×1  │ 3×3  │ 3×3  │ 3×3  │ Global │
        │ conv │ d=6  │ d=12 │ d=18 │  pool  │
        └──────┴──────┴──────┴──────┴────────┘
            ↓      ↓      ↓      ↓      ↓
        Concatenate (5 × out_channels)
            ↓
        1×1 projection conv
            ↓
        Output features (out_channels)
    
    Reference: DeepLabV3+ paper (Chen et al., 2018)
    """
    
    def __init__(self, in_channels: int = 2560, out_channels: int = 256,
                 dilation_rates: tuple = (1, 6, 12, 18)):
        """
        Initialize ASPP module.
        
        Args:
            in_channels: Number of input channels (default 2560 for EfficientNet-B7)
            out_channels: Number of output channels for each branch (default 256)
            dilation_rates: Tuple of dilation rates for atrous convolutions
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rates = dilation_rates
        
        # Parallel atrous convolution branches
        modules = []
        
        # Branch 1: 1×1 convolution (dilation=1, equivalent to standard conv)
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Branches 2-4: 3×3 atrous convolutions with dilation rates 6, 12, 18
        for dilation in dilation_rates[1:]:
            modules.append(ASPPConv(in_channels, out_channels, dilation))
        
        # Branch 5: Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Projection layer to fuse all branches
        # Input: 5 branches × out_channels = 5 * out_channels
        # Output: out_channels
        self.project = nn.Sequential(
            nn.Conv2d(
                len(self.convs) * out_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Dropout for regularization
        )
        
        logger.info(
            f"Initialized ASPP module "
            f"(in_channels={in_channels}, out_channels={out_channels}, "
            f"dilation_rates={dilation_rates})"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ASPP.
        
        Args:
            x: Input features [B, in_channels, H, W]
        
        Returns:
            Multi-scale features [B, out_channels, H, W]
        """
        # Apply all parallel branches
        branch_outputs = []
        for conv in self.convs:
            branch_outputs.append(conv(x))
        
        # Concatenate all branches along channel dimension
        # Each branch: [B, out_channels, H, W]
        # Concatenated: [B, 5 * out_channels, H, W]
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # Project to final output channels
        # [B, 5 * out_channels, H, W] → [B, out_channels, H, W]
        output = self.project(concatenated)
        
        return output
    
    def get_receptive_field_info(self) -> dict:
        """
        Get information about receptive field for each branch.
        
        Returns:
            Dictionary with receptive field information
        """
        # Effective receptive field for 3×3 convolution with dilation d:
        # RF = 1 + 2 * d
        
        receptive_fields = {}
        
        # 1×1 conv branch
        receptive_fields['1x1_conv'] = 1
        
        # Atrous conv branches
        for i, dilation in enumerate(self.dilation_rates[1:], start=2):
            rf = 1 + 2 * dilation
            receptive_fields[f'atrous_conv_{dilation}'] = rf
        
        # Global pooling branch
        receptive_fields['global_pool'] = 'entire_feature_map'
        
        return receptive_fields


class LightweightASPP(nn.Module):
    """
    Lightweight ASPP variant for faster inference or lower memory usage.
    
    Uses depthwise separable convolutions instead of standard convolutions.
    Can be used for ablation studies or deployment constraints.
    """
    
    def __init__(self, in_channels: int = 2560, out_channels: int = 256):
        """
        Initialize lightweight ASPP.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        # Depthwise separable convolutions for atrous branches
        # (Depthwise conv + pointwise conv)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.atrous_conv = nn.Sequential(
            # Depthwise conv with dilation
            nn.Conv2d(
                in_channels, in_channels, 3,
                padding=6, dilation=6,
                groups=in_channels, bias=False
            ),
            # Pointwise conv
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = ASPPPooling(in_channels, out_channels)
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        logger.info("Initialized LightweightASPP module")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        feat1 = self.conv1(x)
        feat2 = self.atrous_conv(x)
        feat3 = self.global_pool(x)
        
        concatenated = torch.cat([feat1, feat2, feat3], dim=1)
        output = self.project(concatenated)
        
        return output


# Export main classes
__all__ = ['ASPP', 'LightweightASPP']
