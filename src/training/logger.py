"""
Reproducibility logging for VERITAS training.

Logs all information needed to reproduce experiments:
- Random seeds
- Environment details (GPU, Python, PyTorch versions)
- Training configuration
- Model architecture
- Dataset statistics

Requirements: 13.11, 21
"""

import torch
import sys
import platform
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import subprocess

logger = logging.getLogger(__name__)


class ReproducibilityLogger:
    """
    Logs all information needed to reproduce experiments.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize reproducibility logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.info = {
            'timestamp': datetime.now().isoformat(),
            'environment': {},
            'seeds': {},
            'configuration': {},
            'model': {},
            'dataset': {}
        }
    
    def log_environment(self):
        """Log runtime environment details."""
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            env_info['cuda_version'] = torch.version.cuda
            env_info['cudnn_version'] = torch.backends.cudnn.version()
            env_info['gpu_count'] = torch.cuda.device_count()
            env_info['gpu_devices'] = [
                {
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'compute_capability': f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
                }
                for i in range(torch.cuda.device_count())
            ]
        
        # Try to get library versions
        try:
            import numpy as np
            env_info['numpy_version'] = np.__version__
        except ImportError:
            pass
        
        try:
            import PIL
            env_info['pillow_version'] = PIL.__version__
        except ImportError:
            pass
        
        try:
            import cv2
            env_info['opencv_version'] = cv2.__version__
        except ImportError:
            pass
        
        try:
            import torchvision
            env_info['torchvision_version'] = torchvision.__version__
        except ImportError:
            pass
        
        try:
            import scipy
            env_info['scipy_version'] = scipy.__version__
        except ImportError:
            pass
        
        self.info['environment'] = env_info
        
        logger.info("Logged environment information")
        return env_info
    
    def log_seeds(self, seeds: Dict[str, int]):
        """
        Log random seeds.
        
        Args:
            seeds: Dictionary of seeds (python, numpy, pytorch, etc.)
        """
        self.info['seeds'] = seeds
        logger.info(f"Logged random seeds: {seeds}")
    
    def log_configuration(self, config: Dict):
        """
        Log training configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.info['configuration'] = config
        logger.info("Logged training configuration")
    
    def log_model_info(
        self,
        model_name: str,
        num_parameters: int,
        model_size_mb: float,
        architecture_summary: Optional[Dict] = None
    ):
        """
        Log model information.
        
        Args:
            model_name: Name of the model
            num_parameters: Total number of parameters
            model_size_mb: Model size in MB
            architecture_summary: Optional architecture summary
        """
        model_info = {
            'model_name': model_name,
            'num_parameters': num_parameters,
            'model_size_mb': model_size_mb
        }
        
        if architecture_summary:
            model_info['architecture_summary'] = architecture_summary
        
        self.info['model'] = model_info
        logger.info(f"Logged model info: {num_parameters:,} parameters, {model_size_mb:.2f} MB")
    
    def log_dataset_info(
        self,
        dataset_name: str,
        dataset_path: str,
        split_info: Dict,
        statistics: Optional[Dict] = None
    ):
        """
        Log dataset information.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset
            split_info: Train/val/test split information
            statistics: Optional dataset statistics
        """
        dataset_info = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'split_info': split_info
        }
        
        if statistics:
            dataset_info['statistics'] = statistics
        
        self.info['dataset'] = dataset_info
        logger.info(f"Logged dataset info: {dataset_name}")
    
    def save(self, filename: str = "reproducibility_info.json"):
        """
        Save reproducibility information to JSON.
        
        Args:
            filename: Name of the file to save
        """
        save_path = self.log_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(self.info, f, indent=2)
        
        logger.info(f"Saved reproducibility info: {save_path}")
        
        # Also save a human-readable version
        readme_path = self.log_dir / "REPRODUCIBILITY.md"
        self._save_markdown(readme_path)
    
    def _save_markdown(self, path: Path):
        """Save human-readable markdown version."""
        with open(path, 'w') as f:
            f.write("# Reproducibility Information\n\n")
            
            f.write(f"**Experiment Date**: {self.info['timestamp']}\n\n")
            
            # Environment
            f.write("## Environment\n\n")
            env = self.info['environment']
            f.write(f"- **Python**: {env.get('python_version', 'N/A')}\n")
            f.write(f"- **PyTorch**: {env.get('pytorch_version', 'N/A')}\n")
            f.write(f"- **CUDA**: {env.get('cuda_version', 'N/A')}\n")
            f.write(f"- **Platform**: {env.get('platform', 'N/A')}\n")
            
            if 'gpu_devices' in env:
                f.write(f"\n### GPU Information\n\n")
                for i, gpu in enumerate(env['gpu_devices']):
                    f.write(f"**GPU {i}**: {gpu['name']}\n")
                    f.write(f"- Memory: {gpu['memory_total_gb']:.2f} GB\n")
                    f.write(f"- Compute Capability: {gpu['compute_capability']}\n\n")
            
            # Random Seeds
            if self.info['seeds']:
                f.write("\n## Random Seeds\n\n")
                for key, value in self.info['seeds'].items():
                    f.write(f"- **{key}**: {value}\n")
            
            # Model
            if self.info['model']:
                f.write("\n## Model\n\n")
                model = self.info['model']
                f.write(f"- **Name**: {model.get('model_name', 'N/A')}\n")
                f.write(f"- **Parameters**: {model.get('num_parameters', 0):,}\n")
                f.write(f"- **Size**: {model.get('model_size_mb', 0):.2f} MB\n")
            
            # Dataset
            if self.info['dataset']:
                f.write("\n## Dataset\n\n")
                dataset = self.info['dataset']
                f.write(f"- **Name**: {dataset.get('dataset_name', 'N/A')}\n")
                f.write(f"- **Path**: {dataset.get('dataset_path', 'N/A')}\n")
                
                if 'split_info' in dataset:
                    f.write("\n### Split Information\n\n")
                    for split, info in dataset['split_info'].items():
                        f.write(f"- **{split}**: {info}\n")
            
            # Configuration
            if self.info['configuration']:
                f.write("\n## Configuration\n\n")
                f.write("```json\n")
                f.write(json.dumps(self.info['configuration'], indent=2))
                f.write("\n```\n")
        
        logger.info(f"Saved markdown summary: {path}")


def log_experiment(
    log_dir: str,
    model,
    config: Dict,
    seeds: Dict[str, int],
    dataset_info: Optional[Dict] = None
) -> ReproducibilityLogger:
    """
    Convenience function to log all experiment information.
    
    Args:
        log_dir: Directory to save logs
        model: Model to log
        config: Training configuration
        seeds: Random seeds used
        dataset_info: Optional dataset information
    
    Returns:
        ReproducibilityLogger instance
    """
    repro_logger = ReproducibilityLogger(log_dir)
    
    # Log environment
    repro_logger.log_environment()
    
    # Log seeds
    repro_logger.log_seeds(seeds)
    
    # Log configuration
    repro_logger.log_configuration(config)
    
    # Log model info
    if hasattr(model, 'get_num_parameters'):
        num_params = model.get_num_parameters()
        model_size = model.get_model_size_mb() if hasattr(model, 'get_model_size_mb') else 0
        arch_summary = model.get_architecture_summary() if hasattr(model, 'get_architecture_summary') else None
        
        repro_logger.log_model_info(
            model_name=model.__class__.__name__,
            num_parameters=num_params,
            model_size_mb=model_size,
            architecture_summary=arch_summary
        )
    
    # Log dataset info
    if dataset_info:
        repro_logger.log_dataset_info(**dataset_info)
    
    # Save
    repro_logger.save()
    
    return repro_logger
