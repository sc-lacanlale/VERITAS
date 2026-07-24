"""
Test full VERITAS architecture components.

Tests the complete model with all components:
- Multi-representation extraction
- Feature fusion
- EfficientNet-B7 backbone
- Classification head
- ASPP module
- DeepLabV3+ decoder
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from src.config import Config
from src.models.representation_extractor import RepresentationExtractor
from src.models.feature_fusion import FeatureFusion
from src.models.aspp import ASPP
from src.models.deeplabv3plus import DeepLabV3PlusHead, DeepLabV3PlusDecoder
from src.models.veritas import VERITASModel, EfficientNetB7Backbone, ClassificationHead


@pytest.fixture
def config():
    """Create test configuration."""
    return Config({
        'model': {
            'input_resolution': 600
        },
        'representations': {
            'imagenet_mean': [0.485, 0.456, 0.406],
            'imagenet_std': [0.229, 0.224, 0.225],
            'srm_filter_count': 3,
            'noise_normalization': 'tanh',
            'dct_block_size': 8,
            'frequency_bands': ['low', 'mid', 'high']
        }
    })


def test_representation_extractor(config):
    """Test multi-representation extraction."""
    extractor = RepresentationExtractor(config)
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 600, 600)
    
    # Extract representations
    representations = extractor(x)
    
    # Check outputs
    assert 'rgb' in representations
    assert 'noise' in representations
    assert 'frequency' in representations
    
    # Check shapes
    assert representations['rgb'].shape == (batch_size, 3, 600, 600)
    assert representations['noise'].shape == (batch_size, 3, 600, 600)
    assert representations['frequency'].shape == (batch_size, 3, 600, 600)
    
    print("✓ RepresentationExtractor test passed")


def test_feature_fusion(config):
    """Test feature fusion module."""
    fusion = FeatureFusion(config)
    
    # Create dummy representations
    batch_size = 2
    representations = {
        'rgb': torch.randn(batch_size, 3, 600, 600),
        'noise': torch.randn(batch_size, 3, 600, 600),
        'frequency': torch.randn(batch_size, 3, 600, 600)
    }
    
    # Fuse
    fused = fusion(representations)
    
    # Check output
    assert fused.shape == (batch_size, 3, 600, 600)
    
    print("✓ FeatureFusion test passed")


def test_efficientnet_b7_backbone():
    """Test EfficientNet-B7 backbone."""
    backbone = EfficientNetB7Backbone(pretrained=False)
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 600, 600)
    
    # Forward pass
    features = backbone(x)
    
    # Check outputs
    assert 'low_level' in features
    assert 'high_level' in features
    
    # Check channels
    assert features['low_level'].shape[1] == 48
    assert features['high_level'].shape[1] == 2560
    
    print("✓ EfficientNet-B7 backbone test passed")


def test_classification_head():
    """Test classification head."""
    head = ClassificationHead(in_features=2560)
    
    # Create dummy high-level features
    batch_size = 2
    features = torch.randn(batch_size, 2560, 19, 19)
    
    # Forward pass
    logit = head(features)
    
    # Check output
    assert logit.shape == (batch_size, 1)
    
    print("✓ ClassificationHead test passed")


def test_aspp_module():
    """Test ASPP module."""
    aspp = ASPP(in_channels=2560, out_channels=256)
    
    # Create dummy high-level features
    batch_size = 2
    x = torch.randn(batch_size, 2560, 19, 19)
    
    # Forward pass
    output = aspp(x)
    
    # Check output
    assert output.shape == (batch_size, 256, 19, 19)
    
    print("✓ ASPP module test passed")


def test_deeplabv3plus_decoder():
    """Test DeepLabV3+ decoder."""
    decoder = DeepLabV3PlusDecoder(
        low_level_channels=48,
        low_level_out_channels=48,
        aspp_out_channels=256,
        decoder_channels=256,
        num_classes=1,
        output_size=600
    )
    
    # Create dummy features
    batch_size = 2
    low_level = torch.randn(batch_size, 48, 150, 150)
    high_level = torch.randn(batch_size, 256, 19, 19)
    
    # Forward pass
    output = decoder(low_level, high_level)
    
    # Check output
    assert output.shape == (batch_size, 1, 600, 600)
    
    print("✓ DeepLabV3+ decoder test passed")


def test_deeplabv3plus_head():
    """Test complete DeepLabV3+ head."""
    head = DeepLabV3PlusHead(
        backbone_high_channels=2560,
        backbone_low_channels=48,
        aspp_out_channels=256,
        num_classes=1,
        output_size=600
    )
    
    # Create dummy features
    batch_size = 2
    high_level = torch.randn(batch_size, 2560, 19, 19)
    low_level = torch.randn(batch_size, 48, 150, 150)
    
    # Forward pass
    output = head(high_level, low_level)
    
    # Check output
    assert output.shape == (batch_size, 1, 600, 600)
    
    print("✓ DeepLabV3+ head test passed")


def test_veritas_model_rgb_only(config):
    """Test complete VERITAS model (RGB-only)."""
    model = VERITASModel(
        config,
        use_multi_representation=False,
        use_full_segmentation=True
    )
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 600, 600)
    
    # Forward pass
    outputs = model(x)
    
    # Check outputs
    assert 'classification_logit' in outputs
    assert 'segmentation_logits' in outputs
    
    assert outputs['classification_logit'].shape == (batch_size, 1)
    assert outputs['segmentation_logits'].shape == (batch_size, 1, 600, 600)
    
    # Test predict method
    predictions = model.predict(x)
    assert 'classification_prob' in predictions
    assert 'classification_pred' in predictions
    assert 'segmentation_prob' in predictions
    assert 'segmentation_mask' in predictions
    
    print("✓ VERITAS model (RGB-only) test passed")


def test_veritas_model_multi_representation(config):
    """Test complete VERITAS model with multi-representation."""
    model = VERITASModel(
        config,
        use_multi_representation=True,
        use_full_segmentation=True
    )
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 600, 600)
    
    # Forward pass
    outputs = model(x)
    
    # Check outputs
    assert 'classification_logit' in outputs
    assert 'segmentation_logits' in outputs
    
    assert outputs['classification_logit'].shape == (batch_size, 1)
    assert outputs['segmentation_logits'].shape == (batch_size, 1, 600, 600)
    
    print("✓ VERITAS model (multi-representation) test passed")


def test_gradient_flow(config):
    """Test gradient flow through complete model."""
    model = VERITASModel(
        config,
        use_multi_representation=True,
        use_full_segmentation=True
    )
    
    # Create dummy input and targets
    batch_size = 2
    x = torch.randn(batch_size, 3, 600, 600)
    cls_target = torch.ones(batch_size, 1)
    seg_target = torch.ones(batch_size, 1, 600, 600)
    
    # Forward pass
    outputs = model(x)
    
    # Compute simple loss
    cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs['classification_logit'], cls_target
    )
    seg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs['segmentation_logits'], seg_target
    )
    total_loss = cls_loss + seg_loss
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    
    print("✓ Gradient flow test passed")


def test_model_parameter_count(config):
    """Test model parameter count."""
    model = VERITASModel(
        config,
        use_multi_representation=True,
        use_full_segmentation=True
    )
    
    num_params = model.get_num_parameters()
    model_size_mb = model.get_model_size_mb()
    
    print(f"✓ Model parameter count: {num_params:,}")
    print(f"✓ Model size: {model_size_mb:.2f} MB")
    
    # EfficientNet-B7 has ~66M parameters, total model should be 70-80M
    assert num_params > 60_000_000, "Model seems too small"
    assert num_params < 100_000_000, "Model seems too large"


def test_architecture_summary(config):
    """Test architecture summary."""
    model = VERITASModel(
        config,
        use_multi_representation=True,
        use_full_segmentation=True
    )
    
    summary = model.get_architecture_summary()
    
    assert summary['multi_representation'] == True
    assert summary['full_segmentation'] == True
    assert summary['backbone'] == 'EfficientNet-B7'
    assert summary['backbone_channels_low'] == 48
    assert summary['backbone_channels_high'] == 2560
    assert summary['input_resolution'] == 600
    
    print("✓ Architecture summary test passed")
    print(f"  Summary: {summary}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Testing Full VERITAS Architecture")
    print("="*60 + "\n")
    
    # Create config
    config = Config({
        'model': {
            'input_resolution': 600
        },
        'representations': {
            'imagenet_mean': [0.485, 0.456, 0.406],
            'imagenet_std': [0.229, 0.224, 0.225],
            'srm_filter_count': 3,
            'noise_normalization': 'tanh',
            'dct_block_size': 8,
            'frequency_bands': ['low', 'mid', 'high']
        }
    })
    
    try:
        test_representation_extractor(config)
        test_feature_fusion(config)
        test_efficientnet_b7_backbone()
        test_classification_head()
        test_aspp_module()
        test_deeplabv3plus_decoder()
        test_deeplabv3plus_head()
        test_veritas_model_rgb_only(config)
        test_veritas_model_multi_representation(config)
        test_gradient_flow(config)
        test_model_parameter_count(config)
        test_architecture_summary(config)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
