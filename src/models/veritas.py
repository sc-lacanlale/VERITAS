"""
VERITAS model architecture.

Full multi-task learning model with:
- Multi-representation feature extraction (RGB + noise + frequency)
- Feature fusion module
- EfficientNet-B7 backbone with intermediate feature extraction
- Classification head with global average pooling
- DeepLabV3+ segmentation head with ASPP (to be integrated)

Requirements: 8, 9, 10, 11
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EfficientNetB7Backbone(nn.Module):
    """
    EfficientNet-B7 backbone with intermediate feature extraction.
    
    Extracts features at multiple levels for skip connections in
    the segmentation decoder.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize EfficientNet-B7 backbone.
        
        Args:
            pretrained: Whether to load ImageNet pretrained weights
        """
        super().__init__()
        
        # Load EfficientNet-B7
        efficientnet = models.efficientnet_b7(pretrained=pretrained)
        
        # Extract feature layers (before classifier)
        self.features = efficientnet.features
        
        # Store original classifier input dimension
        self.classifier_in_features = efficientnet.classifier[1].in_features
        
        # Feature extraction points for skip connections
        # EfficientNet-B7 has 8 MBConv blocks
        # We extract features after block 2 (low-level) and final (high-level)
        self.low_level_idx = 2   # After 2nd block (~48 channels)
        self.high_level_idx = -1  # Final features (~2560 channels)
        
        logger.info(f"Initialized EfficientNet-B7 backbone (pretrained={pretrained})")
        logger.info(f"Classifier input features: {self.classifier_in_features}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with intermediate feature extraction.
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Dictionary with:
            - 'low_level': Low-level features for skip connections [B, 48, H/4, W/4]
            - 'high_level': High-level features [B, 2560, H/32, W/32]
        """
        low_level_features = None
        
        # Forward through feature layers
        for idx, layer in enumerate(self.features):
            x = layer(x)
            
            # Extract low-level features after block 2
            if idx == self.low_level_idx:
                low_level_features = x
        
        # x is now the high-level features
        high_level_features = x
        
        return {
            'low_level': low_level_features,
            'high_level': high_level_features
        }
    
    def get_low_level_channels(self) -> int:
        """Get number of channels in low-level features."""
        # EfficientNet-B7 block 2 output: 48 channels
        return 48
    
    def get_high_level_channels(self) -> int:
        """Get number of channels in high-level features."""
        # EfficientNet-B7 final features: 2560 channels
        return self.classifier_in_features


class ClassificationHead(nn.Module):
    """
    Classification head for manipulation detection.
    
    Applies global average pooling followed by a fully connected layer
    to produce a single manipulation probability.
    """
    
    def __init__(self, in_features: int):
        """
        Initialize classification head.
        
        Args:
            in_features: Number of input features from backbone
        """
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, 1)
        
        logger.info(f"Initialized ClassificationHead (in_features={in_features})")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: High-level features [B, C, H, W]
        
        Returns:
            Classification logit [B, 1] (pre-sigmoid)
        """
        # Global average pooling: [B, C, H, W] → [B, C, 1, 1]
        pooled = self.global_pool(features)
        
        # Flatten: [B, C, 1, 1] → [B, C]
        pooled = pooled.flatten(1)
        
        # Fully connected: [B, C] → [B, 1]
        logit = self.fc(pooled)
        
        return logit


# SimplifiedSegmentationHead removed - now using DeepLabV3+ with ASPP


class VERITASModel(nn.Module):
    """
    Complete VERITAS multi-task learning model.
    
    Performs facial manipulation detection (classification) and
    localization (segmentation) using a shared EfficientNet-B7 backbone.
    
    Full architecture with all components:
    - Multi-representation extraction (RGB + noise + frequency)
    - Feature fusion module
    - EfficientNet-B7 backbone with skip connections
    - Classification head with global average pooling
    - DeepLabV3+ segmentation head with ASPP module
    
    Architecture:
        Input Image (3×600×600)
            ↓
        [Multi-Representation Extraction] (optional)
            ├─ RGB stream (3 channels)
            ├─ Noise residual (SRM, 3 channels)
            └─ Frequency domain (DCT, 3 channels)
            ↓
        [Feature Fusion] (optional)
            Concatenate (9 channels) → 1×1 conv → 3 channels
            ↓
        EfficientNet-B7 Backbone
            ├─ Low-level features (block 2): [B, 48, H/4, W/4]
            └─ High-level features (final): [B, 2560, H/32, W/32]
            ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
    Classification Head                    Segmentation Head
    (Global pool + FC)                    (ASPP + DeepLabV3+ decoder)
        │                                           │
        ↓                                           ↓
    Manipulation                            Manipulation
    Probability (0-1)                       Mask (600×600)
    """
    
    def __init__(self, config, use_multi_representation: bool = False,
                 use_full_segmentation: bool = True):
        """
        Initialize complete VERITAS model.
        
        Args:
            config: Configuration object
            use_multi_representation: Whether to use multi-representation
                                     extraction and fusion (Task 10-11)
            use_full_segmentation: Whether to use full DeepLabV3+ with ASPP
                                  (True) or simplified head (False)
        """
        super().__init__()
        
        self.config = config
        self.input_resolution = config.get('model.input_resolution', 600)
        self.use_multi_representation = use_multi_representation
        self.use_full_segmentation = use_full_segmentation
        
        # Optional: Multi-representation extraction and fusion (Tasks 10-11)
        if use_multi_representation:
            from .representation_extractor import RepresentationExtractor
            from .feature_fusion import FeatureFusion
            
            self.representation_extractor = RepresentationExtractor(config)
            self.feature_fusion = FeatureFusion(config)
            logger.info("Using multi-representation feature extraction")
        else:
            self.representation_extractor = None
            self.feature_fusion = None
            logger.info("Using RGB-only input (multi-representation disabled)")
        
        # EfficientNet-B7 backbone with intermediate feature extraction (Task 12)
        self.backbone = EfficientNetB7Backbone(pretrained=True)
        
        # Classification head (Task 13)
        self.classification_head = ClassificationHead(
            in_features=self.backbone.get_high_level_channels()
        )
        
        # Segmentation head (Tasks 14-15)
        if use_full_segmentation:
            from .deeplabv3plus import DeepLabV3PlusHead
            
            self.segmentation_head = DeepLabV3PlusHead(
                backbone_high_channels=self.backbone.get_high_level_channels(),
                backbone_low_channels=self.backbone.get_low_level_channels(),
                aspp_out_channels=256,
                num_classes=1,
                output_size=self.input_resolution
            )
            logger.info("Using full DeepLabV3+ with ASPP for segmentation")
        else:
            # Simplified segmentation for ablation or faster inference
            from .deeplabv3plus import LightweightDecoderHead
            
            self.segmentation_head = LightweightDecoderHead(
                backbone_channels=self.backbone.get_high_level_channels(),
                decoder_channels=128,
                num_classes=1,
                output_size=self.input_resolution
            )
            logger.info("Using lightweight decoder for segmentation")
        
        logger.info("Initialized complete VERITAS model")
        logger.info(f"Total parameters: {self.get_num_parameters():,}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete VERITAS model.
        
        Args:
            x: Input tensor [B, 3, H, W] (H=W=600)
        
        Returns:
            Dictionary with:
            - 'classification_logit': [B, 1] (pre-sigmoid)
            - 'segmentation_logits': [B, 1, 600, 600] (pre-sigmoid)
        """
        # Step 1: Optional multi-representation extraction and fusion
        if self.use_multi_representation:
            # Extract multiple representations
            representations = self.representation_extractor(x)
            
            # Fuse representations into 3-channel input
            x = self.feature_fusion(representations)
        
        # Step 2: Extract features from backbone
        features = self.backbone(x)
        
        # Step 3: Classification branch
        cls_logit = self.classification_head(features['high_level'])
        
        # Step 4: Segmentation branch
        seg_logits = self.segmentation_head(
            features['high_level'],
            features['low_level']
        )
        
        return {
            'classification_logit': cls_logit,
            'segmentation_logits': seg_logits
        }
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Inference with sigmoid activation and thresholding.
        
        Args:
            x: Input tensor [B, 3, H, W]
            threshold: Decision threshold (default 0.5)
        
        Returns:
            Dictionary with:
            - 'classification_prob': [B, 1] (probability in [0, 1])
            - 'classification_pred': [B, 1] (binary: 0=authentic, 1=manipulated)
            - 'segmentation_prob': [B, 1, H, W] (probability map in [0, 1])
            - 'segmentation_mask': [B, 1, H, W] (binary mask)
        """
        # Forward pass
        outputs = self.forward(x)
        
        # Apply sigmoid to get probabilities
        cls_prob = torch.sigmoid(outputs['classification_logit'])
        seg_prob = torch.sigmoid(outputs['segmentation_logits'])
        
        # Apply threshold to get binary predictions
        cls_pred = (cls_prob >= threshold).float()
        seg_mask = (seg_prob >= threshold).float()
        
        return {
            'classification_prob': cls_prob,
            'classification_pred': cls_pred,
            'segmentation_prob': seg_prob,
            'segmentation_mask': seg_mask
        }
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get approximate model size in megabytes."""
        num_params = self.get_num_parameters()
        # Assume float32 (4 bytes per parameter)
        size_mb = (num_params * 4) / (1024 ** 2)
        return size_mb
    
    def freeze_backbone(self):
        """Freeze backbone weights (for fine-tuning experiments)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze EfficientNet-B7 backbone weights")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfroze EfficientNet-B7 backbone weights")
    
    def get_architecture_summary(self) -> dict:
        """
        Get summary of model architecture.
        
        Returns:
            Dictionary with architecture information
        """
        return {
            'multi_representation': self.use_multi_representation,
            'full_segmentation': self.use_full_segmentation,
            'backbone': 'EfficientNet-B7',
            'backbone_channels_low': self.backbone.get_low_level_channels(),
            'backbone_channels_high': self.backbone.get_high_level_channels(),
            'input_resolution': self.input_resolution,
            'total_parameters': self.get_num_parameters(),
            'model_size_mb': self.get_model_size_mb()
        }
