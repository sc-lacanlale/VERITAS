"""
Reproducibility utilities for ensuring deterministic behavior.

This module provides functions to set random seeds across all libraries
(Python, NumPy, PyTorch) to ensure reproducible experiments.
"""

import random
import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    This function sets random seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generators (CPU and CUDA)
    - cuDNN deterministic behavior
    
    Note: Setting deterministic behavior may impact performance due to
    disabling cuDNN's auto-tuning and benchmarking features.
    
    Args:
        seed: Random seed value (default: 42)
        
    Example:
        >>> set_random_seed(42)
        ✓ Random seed set to 42 for reproducibility
          - Python random: 42
          - NumPy: 42
          - PyTorch: 42
          - cuDNN deterministic: True
          - cuDNN benchmark: False
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set deterministic behavior for cuDNN
    # Note: This may reduce performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed} for reproducibility")
    print(f"  - Python random: {seed}")
    print(f"  - NumPy: {seed}")
    print(f"  - PyTorch: {seed}")
    if torch.cuda.is_available():
        print(f"  - PyTorch CUDA: {seed}")
    print(f"  - cuDNN deterministic: True")
    print(f"  - cuDNN benchmark: False")


def get_random_state() -> dict:
    """
    Get current random state from all random number generators.
    
    Useful for debugging reproducibility issues or saving/restoring
    random state during experiments.
    
    Returns:
        Dictionary containing random states from Python, NumPy, and PyTorch
        
    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def restore_random_state(state: dict) -> None:
    """
    Restore random state from a previously saved state.
    
    Args:
        state: Dictionary containing random states (from get_random_state())
        
    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    
    print("✓ Random state restored")
