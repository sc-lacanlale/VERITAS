"""
Environment verification and logging utilities.

This module provides functions to verify the runtime environment,
check CUDA/GPU availability, and log environment details for reproducibility.
"""

import sys
import torch
import subprocess
from typing import Dict, Any


def verify_environment() -> Dict[str, Any]:
    """
    Verify runtime environment and collect system information.
    
    Checks:
    - Python version
    - PyTorch and related library versions
    - CUDA availability and GPU information
    - cuDNN version
    
    Returns:
        Dictionary containing environment information
        
    Example:
        >>> env_info = verify_environment()
        >>> print(f"GPU: {env_info['gpu_name']}")
        >>> print(f"VRAM: {env_info['total_vram_gb']:.2f} GB")
    """
    import torch
    import torchvision
    import cv2
    import numpy as np
    import sklearn
    import scipy
    from PIL import Image
    
    env_info = {
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'opencv_version': cv2.__version__,
        'numpy_version': np.__version__,
        'sklearn_version': sklearn.__version__,
        'scipy_version': scipy.__version__,
        'pillow_version': Image.__version__,
    }
    
    # GPU information
    cuda_available = torch.cuda.is_available()
    env_info['cuda_available'] = cuda_available
    
    if cuda_available:
        env_info['gpu_count'] = torch.cuda.device_count()
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['total_vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        env_info['cuda_version'] = torch.version.cuda
        env_info['cudnn_version'] = torch.backends.cudnn.version()
    else:
        env_info['gpu_count'] = 0
        env_info['gpu_name'] = 'CPU'
        env_info['total_vram_gb'] = 0
        env_info['cuda_version'] = 'N/A'
        env_info['cudnn_version'] = 'N/A'
    
    return env_info


def print_environment_info(env_info: Dict[str, Any] = None) -> None:
    """
    Print formatted environment information.
    
    Args:
        env_info: Environment information dictionary (from verify_environment())
                 If None, will call verify_environment() automatically
    """
    if env_info is None:
        env_info = verify_environment()
    
    print("=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    
    print(f"\nPython version: {env_info['python_version']}")
    
    print(f"\nDependency Versions:")
    print(f"  PyTorch: {env_info['torch_version']}")
    print(f"  torchvision: {env_info['torchvision_version']}")
    print(f"  OpenCV: {env_info['opencv_version']}")
    print(f"  NumPy: {env_info['numpy_version']}")
    print(f"  scikit-learn: {env_info['sklearn_version']}")
    print(f"  scipy: {env_info['scipy_version']}")
    print(f"  Pillow: {env_info['pillow_version']}")
    
    print(f"\nCUDA Available: {env_info['cuda_available']}")
    
    if env_info['cuda_available']:
        print(f"GPU Count: {env_info['gpu_count']}")
        print(f"\nGPU 0:")
        print(f"  Model: {env_info['gpu_name']}")
        print(f"  Total VRAM: {env_info['total_vram_gb']:.2f} GB")
        print(f"\nCUDA Version: {env_info['cuda_version']}")
        print(f"cuDNN Version: {env_info['cudnn_version']}")
    else:
        print("\n⚠ WARNING: CUDA is not available. Running on CPU.")
        print("  For GPU acceleration, ensure you're using a GPU runtime:")
        print("  Runtime > Change runtime type > Hardware accelerator > GPU")
    
    print("\n" + "=" * 60)


def test_gpu() -> bool:
    """
    Test GPU functionality with a simple matrix multiplication.
    
    Returns:
        True if GPU test successful, False otherwise
        
    Example:
        >>> if test_gpu():
        ...     print("GPU is working correctly")
        ... else:
        ...     print("GPU test failed")
    """
    if not torch.cuda.is_available():
        print("⚠ Cannot test GPU: CUDA not available")
        return False
    
    try:
        # Create test tensors on GPU
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.matmul(test_tensor, test_tensor)
        
        # Synchronize to ensure operation completed
        torch.cuda.synchronize()
        
        print("✓ GPU test successful - device is functioning correctly")
        return True
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage information.
    
    Returns:
        Dictionary with memory information in GB:
        - allocated: Memory currently allocated by tensors
        - cached: Memory cached by PyTorch allocator
        - free: Free memory available
        - total: Total GPU memory
        
    Example:
        >>> mem = get_gpu_memory_info()
        >>> print(f"Allocated: {mem['allocated']:.2f} GB")
        >>> print(f"Free: {mem['free']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0.0,
            'cached': 0.0,
            'free': 0.0,
            'total': 0.0
        }
    
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    cached = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - allocated
    
    return {
        'allocated': allocated,
        'cached': cached,
        'free': free,
        'total': total
    }


def print_gpu_memory_info() -> None:
    """
    Print formatted GPU memory information.
    
    Example:
        >>> print_gpu_memory_info()
        GPU Memory Usage:
          Allocated: 2.34 GB
          Cached: 3.12 GB
          Free: 12.54 GB
          Total: 15.00 GB
    """
    mem = get_gpu_memory_info()
    
    if mem['total'] == 0:
        print("No GPU available")
        return
    
    print("GPU Memory Usage:")
    print(f"  Allocated: {mem['allocated']:.2f} GB")
    print(f"  Cached: {mem['cached']:.2f} GB")
    print(f"  Free: {mem['free']:.2f} GB")
    print(f"  Total: {mem['total']:.2f} GB")
