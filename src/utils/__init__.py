"""
Utility functions for VERITAS project.
"""

# Import only if torch is available (for non-Colab testing)
try:
    from .reproducibility import set_random_seed
    __all__ = ['set_random_seed']
except ImportError:
    __all__ = []
