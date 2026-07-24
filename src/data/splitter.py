"""
Dataset splitting with leakage prevention for VERITAS.

Creates train/validation/test splits while preventing data leakage.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 24
"""

import json
import os
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Create train/validation/test splits with leakage prevention.
    
    Implements:
    - 70% training, 15% validation, 15% test split
    - Ensures no image appears in multiple splits
    - Near-duplicate detection using perceptual hashing
    - Optional identity-based splitting
    - Persistent split assignments for reproducibility
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        """
        Initialize splitter.
        
        Args:
            train_ratio: Training set ratio (default: 0.7)
            val_ratio: Validation set ratio (default: 0.15)
            test_ratio: Test set ratio (default: 0.15)
            random_seed: Random seed for reproducibility
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'duplicates_found': 0,
            'train_category_dist': Counter(),
            'val_category_dist': Counter(),
            'test_category_dist': Counter()
        }
    
    def _compute_image_hash(self, image_path: str) -> str:
        """
        Compute perceptual hash for near-duplicate detection.
        
        For now, uses file path hash. Full implementation would use
        actual image content hashing (e.g., pHash, dHash).
        
        Args:
            image_path: Path to image
            
        Returns:
            Hash string
        """
        # Simple approach: hash the filename
        # TODO: Implement actual perceptual hashing with PIL/cv2
        filename = Path(image_path).stem
        return hashlib.md5(filename.encode()).hexdigest()[:16]
    
    def check_duplicates(
        self,
        annotations: List
    ) -> List[Tuple[int, int]]:
        """
        Check for potential duplicate images.
        
        Args:
            annotations: List of ImageAnnotation objects
            
        Returns:
            List of (idx1, idx2) tuples for potential duplicates
        """
        logger.info("Checking for near-duplicate images...")
        
        hash_to_indices = {}
        duplicates = []
        
        for idx, ann in enumerate(annotations):
            img_hash = self._compute_image_hash(ann.image_path)
            
            if img_hash in hash_to_indices:
                # Potential duplicate
                for prev_idx in hash_to_indices[img_hash]:
                    duplicates.append((prev_idx, idx))
                    logger.warning(
                        f"Potential duplicate: {annotations[prev_idx].image_path} "
                        f"<-> {ann.image_path}"
                    )
                hash_to_indices[img_hash].append(idx)
            else:
                hash_to_indices[img_hash] = [idx]
        
        self.stats['duplicates_found'] = len(duplicates)
        
        if duplicates:
            logger.warning(f"Found {len(duplicates)} potential duplicate pairs")
        else:
            logger.info("No duplicates found")
        
        return duplicates
    
    def split_by_identity(
        self,
        annotations: List,
        identity_key: str = 'image_id'
    ) -> Tuple[List, List, List]:
        """
        Split dataset ensuring no identity overlap between splits.
        
        This is a placeholder - full implementation would require
        identity metadata from the dataset.
        
        Args:
            annotations: List of ImageAnnotation objects
            identity_key: Key to extract identity from annotations
            
        Returns:
            Tuple of (train_annotations, val_annotations, test_annotations)
        """
        logger.info("Splitting by identity (preventing identity overlap)...")
        
        # Group by identity
        identity_to_annotations = {}
        for ann in annotations:
            identity = getattr(ann, identity_key, ann.image_id)
            if identity not in identity_to_annotations:
                identity_to_annotations[identity] = []
            identity_to_annotations[identity].append(ann)
        
        # Get unique identities
        identities = list(identity_to_annotations.keys())
        n_identities = len(identities)
        
        logger.info(f"Found {n_identities} unique identities")
        
        # Shuffle identities
        np.random.seed(self.random_seed)
        np.random.shuffle(identities)
        
        # Split identities
        n_train = int(n_identities * self.train_ratio)
        n_val = int(n_identities * self.val_ratio)
        
        train_identities = identities[:n_train]
        val_identities = identities[n_train:n_train + n_val]
        test_identities = identities[n_train + n_val:]
        
        # Assign annotations
        train_anns = []
        val_anns = []
        test_anns = []
        
        for identity in train_identities:
            train_anns.extend(identity_to_annotations[identity])
        
        for identity in val_identities:
            val_anns.extend(identity_to_annotations[identity])
        
        for identity in test_identities:
            test_anns.extend(identity_to_annotations[identity])
        
        logger.info(
            f"Split by identity: {len(train_anns)} train, "
            f"{len(val_anns)} val, {len(test_anns)} test"
        )
        
        return train_anns, val_anns, test_anns
    
    def split_random(
        self,
        annotations: List
    ) -> Tuple[List, List, List]:
        """
        Random split ensuring no image appears in multiple splits.
        
        Args:
            annotations: List of ImageAnnotation objects
            
        Returns:
            Tuple of (train_annotations, val_annotations, test_annotations)
        """
        logger.info("Performing random split...")
        
        # Shuffle indices
        n_samples = len(annotations)
        indices = np.arange(n_samples)
        
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        
        # Calculate split points
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create annotation lists
        train_anns = [annotations[i] for i in train_indices]
        val_anns = [annotations[i] for i in val_indices]
        test_anns = [annotations[i] for i in test_indices]
        
        logger.info(
            f"Random split: {len(train_anns)} train, "
            f"{len(val_anns)} val, {len(test_anns)} test"
        )
        
        return train_anns, val_anns, test_anns
    
    def split(
        self,
        annotations: List,
        check_duplicates: bool = True,
        split_by_identity: bool = False
    ) -> Dict[str, List]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            annotations: List of ImageAnnotation objects
            check_duplicates: Whether to check for near-duplicates
            split_by_identity: Whether to prevent identity overlap
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        self.stats['total_samples'] = len(annotations)
        
        logger.info(f"Splitting {len(annotations)} samples...")
        logger.info(
            f"Target ratios: train={self.train_ratio}, "
            f"val={self.val_ratio}, test={self.test_ratio}"
        )
        
        # Check for duplicates
        if check_duplicates:
            duplicates = self.check_duplicates(annotations)
            if duplicates:
                logger.warning(
                    "Duplicates detected - consider removing before training"
                )
        
        # Perform split
        if split_by_identity:
            train_anns, val_anns, test_anns = self.split_by_identity(annotations)
        else:
            train_anns, val_anns, test_anns = self.split_random(annotations)
        
        # Update statistics
        self.stats['train_samples'] = len(train_anns)
        self.stats['val_samples'] = len(val_anns)
        self.stats['test_samples'] = len(test_anns)
        
        # Category distribution
        for ann in train_anns:
            self.stats['train_category_dist'][ann.category] += 1
        
        for ann in val_anns:
            self.stats['val_category_dist'][ann.category] += 1
        
        for ann in test_anns:
            self.stats['test_category_dist'][ann.category] += 1
        
        # Log summary
        self._log_split_summary()
        
        return {
            'train': train_anns,
            'val': val_anns,
            'test': test_anns
        }
    
    def _log_split_summary(self):
        """Log split summary statistics."""
        logger.info("=" * 60)
        logger.info("DATASET SPLIT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples: {self.stats['total_samples']}")
        logger.info(f"Train: {self.stats['train_samples']} "
                   f"({self.stats['train_samples']/self.stats['total_samples']*100:.1f}%)")
        logger.info(f"Val: {self.stats['val_samples']} "
                   f"({self.stats['val_samples']/self.stats['total_samples']*100:.1f}%)")
        logger.info(f"Test: {self.stats['test_samples']} "
                   f"({self.stats['test_samples']/self.stats['total_samples']*100:.1f}%)")
        
        if self.stats['duplicates_found'] > 0:
            logger.info(f"\nDuplicates found: {self.stats['duplicates_found']}")
        
        logger.info("\nCategory distribution per split:")
        
        logger.info("  Train:")
        for cat, count in sorted(self.stats['train_category_dist'].items()):
            pct = count / self.stats['train_samples'] * 100
            logger.info(f"    {cat}: {count} ({pct:.1f}%)")
        
        logger.info("  Val:")
        for cat, count in sorted(self.stats['val_category_dist'].items()):
            pct = count / self.stats['val_samples'] * 100
            logger.info(f"    {cat}: {count} ({pct:.1f}%)")
        
        logger.info("  Test:")
        for cat, count in sorted(self.stats['test_category_dist'].items()):
            pct = count / self.stats['test_samples'] * 100
            logger.info(f"    {cat}: {count} ({pct:.1f}%)")
        
        logger.info("=" * 60)
    
    def save_split(
        self,
        splits: Dict[str, List],
        output_path: str
    ):
        """
        Save split assignments to JSON file for reproducibility.
        
        Args:
            splits: Dictionary with 'train', 'val', 'test' keys
            output_path: Path to save JSON file
        """
        # Create serializable format
        split_data = {
            'metadata': {
                'total_samples': self.stats['total_samples'],
                'train_samples': self.stats['train_samples'],
                'val_samples': self.stats['val_samples'],
                'test_samples': self.stats['test_samples'],
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'random_seed': self.random_seed,
                'duplicates_found': self.stats['duplicates_found'],
                'train_category_dist': dict(self.stats['train_category_dist']),
                'val_category_dist': dict(self.stats['val_category_dist']),
                'test_category_dist': dict(self.stats['test_category_dist'])
            },
            'splits': {
                'train': [ann.image_path for ann in splits['train']],
                'val': [ann.image_path for ann in splits['val']],
                'test': [ann.image_path for ann in splits['test']]
            }
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        logger.info(f"✓ Split assignments saved to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get split statistics."""
        return self.stats.copy()


def create_splits(
    annotations: List,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    check_duplicates: bool = True,
    split_by_identity: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, List]:
    """
    Convenience function to create dataset splits.
    
    Args:
        annotations: List of ImageAnnotation objects
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed
        check_duplicates: Whether to check for duplicates
        split_by_identity: Whether to split by identity
        save_path: Optional path to save split assignments
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    splitter = DatasetSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    splits = splitter.split(
        annotations,
        check_duplicates=check_duplicates,
        split_by_identity=split_by_identity
    )
    
    if save_path:
        splitter.save_split(splits, save_path)
    
    return splits
