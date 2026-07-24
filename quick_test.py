#!/usr/bin/env python3
"""
Quick test script to verify VERITAS components work.

Run this locally (not in Colab) to test implementation without GPU.
"""

import sys
sys.path.insert(0, 'src')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.config import Config, create_default_config
        print("✓ Config module")
    except ImportError as e:
        print(f"✗ Config module: {e}")
        return False
    
    try:
        from src.utils.reproducibility import set_random_seed
        print("✓ Reproducibility module")
    except ImportError as e:
        print(f"✗ Reproducibility module: {e}")
        return False
    
    try:
        from src.data.annotation_parser import AnnotationParser, parse_openforensics_json
        print("✓ Annotation parser")
    except ImportError as e:
        print(f"✗ Annotation parser: {e}")
        return False
    
    try:
        from src.data.mask_converter import polygon_to_mask, create_mask_from_annotation
        print("✓ Mask converter")
    except ImportError as e:
        print(f"✗ Mask converter: {e}")
        return False
    
    try:
        from src.data.validator import DataValidator, validate_dataset
        print("✓ Data validator")
    except ImportError as e:
        print(f"✗ Data validator: {e}")
        return False
    
    try:
        from src.data.dataset import VERITASDataset, create_dataloader
        print("✓ Dataset and DataLoader")
    except ImportError as e:
        print(f"✗ Dataset and DataLoader: {e}")
        return False
    
    try:
        from src.models.veritas import VERITASModel
        print("✓ VERITAS model")
    except ImportError as e:
        print(f"✗ VERITAS model: {e}")
        return False
    
    try:
        from src.models.loss import MultiTaskLoss
        print("✓ Multi-task loss")
    except ImportError as e:
        print(f"✗ Multi-task loss: {e}")
        return False
    
    return True


def test_mask_converter():
    """Test mask converter with dummy data."""
    print("\nTesting mask converter...")
    
    from src.data.mask_converter import polygon_to_mask
    import numpy as np
    
    # Create square polygon
    polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
    mask = polygon_to_mask(polygon, 300, 300)
    
    assert mask.shape == (300, 300), "Mask shape incorrect"
    assert mask.dtype == np.uint8, "Mask dtype incorrect"
    assert mask.max() == 1, "Mask should contain 1s"
    assert mask.min() == 0, "Mask should contain 0s"
    assert mask.sum() > 0, "Mask should have non-zero pixels"
    
    print("✓ Mask converter works correctly")


def test_model_creation():
    """Test model can be created."""
    print("\nTesting model creation...")
    
    import torch
    from src.models.veritas import VERITASModel
    from src.config import create_default_config
    
    # Create dummy config
    config_dict = create_default_config("./")
    
    class DummyConfig:
        def __init__(self, d):
            self.d = d
        def get(self, key, default=None):
            keys = key.split('.')
            val = self.d
            for k in keys:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    return default
            return val
    
    config = DummyConfig(config_dict)
    
    # Create model
    model = VERITASModel(config)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 600, 600)
    with torch.no_grad():
        outputs = model(dummy_input)
    
    assert 'classification_logit' in outputs, "Missing classification output"
    assert 'segmentation_logits' in outputs, "Missing segmentation output"
    assert outputs['classification_logit'].shape == (2, 1), "Classification shape incorrect"
    assert outputs['segmentation_logits'].shape == (2, 1, 600, 600), "Segmentation shape incorrect"
    
    print("✓ Model creation and forward pass work correctly")


def test_loss_computation():
    """Test loss computation."""
    print("\nTesting loss computation...")
    
    import torch
    from src.models.loss import MultiTaskLoss
    from src.config import create_default_config
    
    # Create dummy config
    config_dict = create_default_config("./")
    
    class DummyConfig:
        def __init__(self, d):
            self.d = d
        def get(self, key, default=None):
            keys = key.split('.')
            val = self.d
            for k in keys:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    return default
            return val
    
    config = DummyConfig(config_dict)
    
    # Create loss function
    loss_fn = MultiTaskLoss(config)
    
    # Create dummy predictions and targets
    predictions = {
        'classification_logit': torch.randn(2, 1),
        'segmentation_logits': torch.randn(2, 1, 600, 600)
    }
    
    targets = {
        'label': torch.randint(0, 2, (2, 1)).float(),
        'mask': torch.randint(0, 2, (2, 1, 600, 600)).float()
    }
    
    # Compute loss
    losses = loss_fn(predictions, targets)
    
    assert 'total_loss' in losses, "Missing total loss"
    assert 'classification_loss' in losses, "Missing classification loss"
    assert 'segmentation_loss' in losses, "Missing segmentation loss"
    assert losses['total_loss'].numel() == 1, "Total loss should be scalar"
    
    print("✓ Loss computation works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("VERITAS QUICK TEST")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import test failed - fix import errors before proceeding")
        return False
    
    # Test mask converter
    try:
        test_mask_converter()
    except Exception as e:
        print(f"✗ Mask converter test failed: {e}")
        return False
    
    # Test model creation
    try:
        test_model_creation()
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss computation
    try:
        test_loss_computation()
    except Exception as e:
        print(f"✗ Loss computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nComponents ready:")
    print("  ✓ Data pipeline (parsing, masks, validation, dataset)")
    print("  ✓ Model architecture (simplified)")
    print("  ✓ Loss function (multi-task)")
    print("\nNext steps:")
    print("  1. Run smoke_test.ipynb in Google Colab")
    print("  2. Verify training works on 100-image subset")
    print("  3. Implement full architecture components")
    print("\nSee SMOKE_TEST_READY.md for details.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
