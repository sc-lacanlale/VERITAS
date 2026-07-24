"""
Configuration management system for VERITAS project.

This module provides utilities for loading, saving, and updating configuration files.
All configurations are stored as JSON files on Google Drive for persistence across
Colab sessions.
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime


class Config:
    """
    Configuration management class for VERITAS project.
    
    Handles loading, saving, and updating configuration files stored on Google Drive.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = config_path
        self.config = self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is malformed
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please run the setup notebook to create initial configuration."
            )
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Configuration file is malformed: {self.config_path}",
                e.doc,
                e.pos
            )
    
    def save(self) -> None:
        """Save current configuration to JSON file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"✓ Configuration saved to {self.config_path}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Performs deep update to merge nested dictionaries.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._deep_update(self.config, updates)
        self.save()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Recursively update nested dictionaries.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if (isinstance(value, dict) and 
                key in base_dict and 
                isinstance(base_dict[key], dict)):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports dot notation for nested keys (e.g., 'model.backbone').
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Supports dot notation for nested keys (e.g., 'model.backbone').
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to nested location
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
        self.save()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()


def create_default_config(base_path: str, gpu_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create default VERITAS configuration.
    
    Args:
        base_path: Base path for VERITAS project on Google Drive
        gpu_info: GPU information dictionary (optional)
        
    Returns:
        Default configuration dictionary
    """
    if gpu_info is None:
        import torch
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'total_vram_gb': (torch.cuda.get_device_properties(0).total_memory / (1024**3) 
                            if torch.cuda.is_available() else 0),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
        }
    
    config = {
        "project_name": "VERITAS",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        
        # Paths
        "paths": {
            "base_path": base_path,
            "dataset_path": os.path.join(base_path, "dataset"),
            "checkpoints_path": os.path.join(base_path, "checkpoints"),
            "logs_path": os.path.join(base_path, "logs"),
            "results_path": os.path.join(base_path, "results")
        },
        
        # Model configuration
        "model": {
            "input_resolution": 600,
            "backbone": "efficientnet_b7",
            "pretrained": True,
            "num_classes": 1
        },
        
        # Training configuration
        "training": {
            "batch_size": 8,
            "num_epochs": 50,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "gradient_clip_max_norm": 1.0
        },
        
        # Loss weights
        "loss": {
            "classification_weight": 0.4,
            "segmentation_weight": 0.6,
            "segmentation_bce_weight": 0.5,
            "segmentation_dice_weight": 0.5
        },
        
        # Data split
        "data": {
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "supported_categories": [
                "authentic",
                "face_swap",
                "face_reenact",
                "inpainting"
            ]
        },
        
        # Augmentation configuration
        "augmentation": {
            "enabled": True,
            "rotation_degrees": 2.0,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "shear_range": 0.1,
            "zoom_range": 0.05,
            "horizontal_flip": True
        },
        
        # Representation configuration
        "representations": {
            "rgb_normalization": "imagenet",
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "srm_filter_count": 3,
            "noise_normalization": "tanh",
            "dct_block_size": 8,
            "frequency_bands": ["low", "mid", "high"]
        },
        
        # ASPP configuration
        "aspp": {
            "dilation_rates": [1, 6, 12, 18],
            "output_channels": 256
        },
        
        # Reproducibility
        "random_seed": 42,
        
        # Device
        "device": "cuda" if gpu_info['cuda_available'] else "cpu",
        "num_workers": 4,
        
        # Environment information
        "environment": gpu_info
    }
    
    return config


def setup_directories(base_path: str) -> Dict[str, str]:
    """
    Create VERITAS directory structure on Google Drive.
    
    Args:
        base_path: Base path for VERITAS project
        
    Returns:
        Dictionary mapping directory names to their paths
    """
    directories = {
        'dataset': 'Dataset storage (images and annotations)',
        'checkpoints': 'Model checkpoint storage',
        'logs': 'Training logs and metrics',
        'results': 'Evaluation results and visualizations'
    }
    
    created_dirs = {}
    
    for dir_name, description in directories.items():
        dir_path = os.path.join(base_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        created_dirs[dir_name] = dir_path
        print(f"✓ Created: {dir_path}")
        print(f"  Purpose: {description}")
    
    return created_dirs
