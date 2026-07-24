"""
Quick validation script for VERITAS full architecture.

Runs basic checks to ensure all components are working correctly.
Use this before proceeding to full training.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.models import VERITASModel


def validate_architecture():
    """Validate VERITAS architecture."""
    
    print("\n" + "="*70)
    print("VERITAS Architecture Validation")
    print("="*70 + "\n")
    
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
    
    # Test 1: RGB-only model
    print("Test 1: RGB-only model")
    print("-" * 70)
    try:
        model_rgb = VERITASModel(
            config,
            use_multi_representation=False,
            use_full_segmentation=True
        )
        
        # Get summary
        summary = model_rgb.get_architecture_summary()
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {summary['total_parameters']:,}")
        print(f"  - Model size: {summary['model_size_mb']:.2f} MB")
        print(f"  - Multi-representation: {summary['multi_representation']}")
        print(f"  - Full segmentation: {summary['full_segmentation']}")
        
        # Test forward pass
        x = torch.randn(1, 3, 600, 600)
        outputs = model_rgb(x)
        
        print(f"✓ Forward pass successful")
        print(f"  - Classification logit shape: {outputs['classification_logit'].shape}")
        print(f"  - Segmentation logits shape: {outputs['segmentation_logits'].shape}")
        
        # Test predict
        predictions = model_rgb.predict(x)
        print(f"✓ Predict method successful")
        print(f"  - Classification prob: {predictions['classification_prob'].item():.4f}")
        print(f"  - Segmentation mask unique values: {predictions['segmentation_mask'].unique().tolist()}")
        
    except Exception as e:
        print(f"✗ RGB-only model failed: {e}")
        return False
    
    print()
    
    # Test 2: Multi-representation model
    print("Test 2: Multi-representation model")
    print("-" * 70)
    try:
        model_multi = VERITASModel(
            config,
            use_multi_representation=True,
            use_full_segmentation=True
        )
        
        # Get summary
        summary = model_multi.get_architecture_summary()
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {summary['total_parameters']:,}")
        print(f"  - Model size: {summary['model_size_mb']:.2f} MB")
        print(f"  - Multi-representation: {summary['multi_representation']}")
        print(f"  - Full segmentation: {summary['full_segmentation']}")
        
        # Test forward pass
        x = torch.randn(1, 3, 600, 600)
        outputs = model_multi(x)
        
        print(f"✓ Forward pass successful")
        print(f"  - Classification logit shape: {outputs['classification_logit'].shape}")
        print(f"  - Segmentation logits shape: {outputs['segmentation_logits'].shape}")
        
    except Exception as e:
        print(f"✗ Multi-representation model failed: {e}")
        return False
    
    print()
    
    # Test 3: Gradient flow
    print("Test 3: Gradient flow")
    print("-" * 70)
    try:
        model = VERITASModel(
            config,
            use_multi_representation=True,
            use_full_segmentation=True
        )
        
        x = torch.randn(2, 3, 600, 600)
        cls_target = torch.ones(2, 1)
        seg_target = torch.ones(2, 1, 600, 600)
        
        # Forward
        outputs = model(x)
        
        # Loss
        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs['classification_logit'], cls_target
        )
        seg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs['segmentation_logits'], seg_target
        )
        total_loss = cls_loss + seg_loss
        
        # Backward
        total_loss.backward()
        
        # Check gradients
        has_grad = 0
        no_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    has_grad += 1
                else:
                    no_grad += 1
        
        print(f"✓ Gradient flow successful")
        print(f"  - Parameters with gradients: {has_grad}")
        print(f"  - Parameters without gradients: {no_grad}")
        print(f"  - Total loss: {total_loss.item():.4f}")
        
        if no_grad > 0:
            print(f"⚠ Warning: {no_grad} parameters have no gradients")
        
    except Exception as e:
        print(f"✗ Gradient flow failed: {e}")
        return False
    
    print()
    
    # Test 4: Lightweight model
    print("Test 4: Lightweight model")
    print("-" * 70)
    try:
        model_light = VERITASModel(
            config,
            use_multi_representation=False,
            use_full_segmentation=False
        )
        
        summary = model_light.get_architecture_summary()
        print(f"✓ Lightweight model created successfully")
        print(f"  - Total parameters: {summary['total_parameters']:,}")
        print(f"  - Model size: {summary['model_size_mb']:.2f} MB")
        
        x = torch.randn(1, 3, 600, 600)
        outputs = model_light(x)
        
        print(f"✓ Forward pass successful")
        
    except Exception as e:
        print(f"✗ Lightweight model failed: {e}")
        return False
    
    print()
    
    # Summary
    print("="*70)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("="*70)
    print("\nArchitecture is ready for:")
    print("  1. Integration with training pipeline")
    print("  2. Smoke test on real data")
    print("  3. Full 50-epoch training")
    print("\nNext steps:")
    print("  - Run tests: python tests/test_full_architecture.py")
    print("  - See ARCHITECTURE_COMPLETE.md for details")
    print("  - See WHATS_NEXT.md for implementation roadmap")
    print()
    
    return True


if __name__ == '__main__':
    try:
        success = validate_architecture()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
