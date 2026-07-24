"""
PyTorch Dataset and DataLoader for VERITAS.

Requirements: 4, 5, 22.4
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from src.data.mask_converter import create_mask_from_annotation

logger = logging.getLogger(__name__)


class VERITASDataset(Dataset):
    """
    PyTorch Dataset for VERITAS training.
    
    Returns (image, mask, label) tuples with preprocessing applied.
    """
    
    def __init__(
        self,
        annotations: List,
        dataset_root: str,
        config,
        split: str = 'train'
    ):
        """
        Initialize dataset.
        
        Args:
            annotations: List of ImageAnnotation objects
            dataset_root: Root directory containing images
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
        """
        self.annotations = annotations
        self.dataset_root = dataset_root
        self.config = config
        self.split = split
        
        # Get configuration
        self.input_size = config.get('model.input_resolution', 600)
        self.imagenet_mean = config.get('representations.imagenet_mean')
        self.imagenet_std = config.get('representations.imagenet_std')
        
        logger.info(f"Created {split} dataset with {len(annotations)} samples")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'image', 'mask', 'label' tensors
        """
        try:
            annotation = self.annotations[idx]
            
            # Load image
            image_path = os.path.join(self.dataset_root, annotation.image_path)
            image = Image.open(image_path).convert('RGB')
            
            # Resize image to target size
            image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
            
            # Convert to numpy
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # Normalize
            for i in range(3):
                image_np[:, :, i] = (image_np[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
            
            # Convert to tensor [C, H, W]
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
            
            # Create mask
            mask = create_mask_from_annotation(
                annotation,
                target_size=(self.input_size, self.input_size)
            )
            
            # Convert mask to tensor [1, H, W]
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            
            # Label tensor [1]
            label_tensor = torch.tensor([annotation.label], dtype=torch.float32)
            
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'label': label_tensor
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy data to avoid breaking training
            return {
                'image': torch.zeros(3, self.input_size, self.input_size),
                'mask': torch.zeros(1, self.input_size, self.input_size),
                'label': torch.tensor([0.0])
            }


def create_dataloader(
    annotations: List,
    dataset_root: str,
    config,
    split: str = 'train',
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for dataset split.
    
    Args:
        annotations: List of ImageAnnotation objects
        dataset_root: Root directory containing images
        config: Configuration object
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size (uses config if None)
        shuffle: Whether to shuffle (True for train, False for val/test if None)
        num_workers: Number of worker processes
        
    Returns:
        PyTorch DataLoader
    """
    dataset = VERITASDataset(
        annotations=annotations,
        dataset_root=dataset_root,
        config=config,
        split=split
    )
    
    # Default batch size from config
    if batch_size is None:
        batch_size = config.get('training.batch_size', 8)
    
    # Default shuffle based on split
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader
