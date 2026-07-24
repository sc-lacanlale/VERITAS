"""
Test script to verify VERITAS environment setup.

Run this after completing the setup notebook to ensure all components
are working correctly.
"""

import sys
import os
import tempfile
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"  ✓ torchvision {torchvision.__version__}")
        
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
        
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
        
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
        
        import scipy
        print(f"  ✓ scipy {scipy.__version__}")
        
        from PIL import Image
        print(f"  ✓ Pillow {Image.__version__}")
        
        print("\n✓ All required packages imported successfully\n")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}\n")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")
    
    import torch
    
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print("\n✓ CUDA test passed\n")
        return True
    else:
        print("\n⚠ CUDA not available - will run on CPU\n")
        return True


def test_config_module():
    """Test configuration management module."""
    print("Testing configuration module...")
    
    try:
        from src.config import Config, create_default_config, setup_directories
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test directory creation
            dirs = setup_directories(tmpdir)
            assert 'dataset' in dirs
            assert 'checkpoints' in dirs
            assert 'logs' in dirs
            assert 'results' in dirs
            print("  ✓ Directory creation works")
            
            # Test config creation
            config_dict = create_default_config(tmpdir)
            assert 'project_name' in config_dict
            assert config_dict['project_name'] == 'VERITAS'
            print("  ✓ Config creation works")
            
            # Test config save/load
            config_path = os.path.join(tmpdir, 'test_config.json')
            with open(config_path, 'w') as f:
                json.dump(config_dict, f)
            
            config = Config(config_path)
            assert config.get('project_name') == 'VERITAS'
            print("  ✓ Config load works")
            
            # Test config update
            config.set('training.batch_size', 16)
            assert config.get('training.batch_size') == 16
            print("  ✓ Config update works")
        
        print("\n✓ Configuration module test passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration module test failed: {e}\n")
        return False


def test_reproducibility_module():
    """Test reproducibility utilities."""
    print("Testing reproducibility module...")
    
    try:
        from src.utils.reproducibility import set_random_seed, get_random_state, restore_random_state
        import numpy as np
        import torch
        
        # Test seed setting
        set_random_seed(42)
        print("  ✓ Seed setting works")
        
        # Test state save/restore
        state = get_random_state()
        assert 'python' in state
        assert 'numpy' in state
        assert 'torch' in state
        print("  ✓ State capture works")
        
        # Generate some random numbers
        np_val1 = np.random.rand()
        torch_val1 = torch.rand(1).item()
        
        # Restore state and verify we get same numbers
        restore_random_state(state)
        np_val2 = np.random.rand()
        torch_val2 = torch.rand(1).item()
        
        assert np_val1 == np_val2, "NumPy state restore failed"
        assert torch_val1 == torch_val2, "PyTorch state restore failed"
        print("  ✓ State restore works")
        
        print("\n✓ Reproducibility module test passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Reproducibility module test failed: {e}\n")
        return False


def test_environment_module():
    """Test environment verification utilities."""
    print("Testing environment module...")
    
    try:
        from src.utils.environment import (
            verify_environment, 
            print_environment_info,
            test_gpu,
            get_gpu_memory_info
        )
        
        # Test environment verification
        env_info = verify_environment()
        assert 'python_version' in env_info
        assert 'cuda_available' in env_info
        print("  ✓ Environment verification works")
        
        # Test GPU memory info
        mem_info = get_gpu_memory_info()
        assert 'total' in mem_info
        print("  ✓ Memory info retrieval works")
        
        # Test GPU if available
        import torch
        if torch.cuda.is_available():
            gpu_ok = test_gpu()
            assert gpu_ok, "GPU test failed"
            print("  ✓ GPU test works")
        
        print("\n✓ Environment module test passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Environment module test failed: {e}\n")
        return False


def run_all_tests():
    """Run all setup tests."""
    print("=" * 60)
    print("VERITAS SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Config Module", test_config_module()))
    results.append(("Reproducibility Module", test_reproducibility_module()))
    results.append(("Environment Module", test_environment_module()))
    
    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print()
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All tests passed! Setup is complete.")
    else:
        print("✗ Some tests failed. Please review errors above.")
    
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
