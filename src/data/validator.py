"""
Data validator for VERITAS dataset.

Validates dataset integrity, filters supported categories, and generates statistics.

Requirements: 1.1, 1.2, 1.6, 1.7, 1.10, 1.11
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from collections import Counter

logger = logging.getLogger(__name__)

# Supported categories from requirements
SUPPORTED_CATEGORIES = [
    "authentic",
    "face_swap",
    "face_reenact",
    "inpainting"
]


class DataValidator:
    """
    Validates dataset annotations and filters to supported categories.
    
    Performs integrity checks:
    - Verify image files exist
    - Verify annotations are valid
    - Filter to supported manipulation categories
    - Generate dataset statistics
    """
    
    def __init__(self, dataset_root: str):
        """
        Initialize validator.
        
        Args:
            dataset_root: Root directory containing dataset images
        """
        self.dataset_root = Path(dataset_root)
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'excluded_samples': 0,
            'missing_images': 0,
            'unsupported_categories': 0,
            'category_distribution': Counter()
        }
    
    def validate_image_exists(self, image_path: str) -> bool:
        """
        Check if image file exists and is readable.
        
        Args:
            image_path: Relative path to image
            
        Returns:
            True if image exists, False otherwise
        """
        full_path = self.dataset_root / image_path
        
        if not full_path.exists():
            logger.warning(f"Image not found: {full_path}")
            return False
        
        if not full_path.is_file():
            logger.warning(f"Path is not a file: {full_path}")
            return False
        
        # Try to check if readable
        if not os.access(full_path, os.R_OK):
            logger.warning(f"Image not readable: {full_path}")
            return False
        
        return True
    
    def is_supported_category(self, category: str) -> bool:
        """
        Check if category is supported.
        
        Args:
            category: Category string
            
        Returns:
            True if supported, False otherwise
        """
        return category in SUPPORTED_CATEGORIES
    
    def validate_annotation(self, annotation) -> tuple[bool, str]:
        """
        Validate a single annotation.
        
        Args:
            annotation: ImageAnnotation object
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if image exists
        if not self.validate_image_exists(annotation.image_path):
            return False, "Image file not found"
        
        # Check if category is supported
        if not self.is_supported_category(annotation.category):
            return False, f"Unsupported category: {annotation.category}"
        
        # Check label is valid
        if annotation.label not in [0, 1]:
            return False, f"Invalid label: {annotation.label}"
        
        # Check dimensions are positive
        if annotation.image_width <= 0 or annotation.image_height <= 0:
            return False, "Invalid image dimensions"
        
        return True, "Valid"
    
    def validate_and_filter(
        self,
        annotations: List
    ) -> tuple[List, Dict[str, Any]]:
        """
        Validate annotations and filter to supported categories.
        
        Args:
            annotations: List of ImageAnnotation objects
            
        Returns:
            Tuple of (filtered_annotations, statistics)
        """
        logger.info(f"Validating {len(annotations)} annotations...")
        
        filtered_annotations = []
        exclusion_reasons = Counter()
        
        for ann in annotations:
            self.stats['total_samples'] += 1
            
            is_valid, reason = self.validate_annotation(ann)
            
            if is_valid:
                filtered_annotations.append(ann)
                self.stats['valid_samples'] += 1
                self.stats['category_distribution'][ann.category] += 1
            else:
                self.stats['excluded_samples'] += 1
                exclusion_reasons[reason] += 1
                
                # Update specific error counters
                if "not found" in reason:
                    self.stats['missing_images'] += 1
                elif "Unsupported category" in reason:
                    self.stats['unsupported_categories'] += 1
                
                logger.debug(f"Excluded {ann.image_path}: {reason}")
        
        # Log summary
        self._log_validation_summary(exclusion_reasons)
        
        return filtered_annotations, self.get_statistics()
    
    def _log_validation_summary(self, exclusion_reasons: Counter):
        """Log validation summary."""
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples: {self.stats['total_samples']}")
        logger.info(f"Valid samples: {self.stats['valid_samples']}")
        logger.info(f"Excluded samples: {self.stats['excluded_samples']}")
        
        if self.stats['excluded_samples'] > 0:
            logger.info(f"\nExclusion reasons:")
            for reason, count in exclusion_reasons.most_common():
                logger.info(f"  {reason}: {count}")
        
        logger.info(f"\nCategory distribution (valid samples):")
        for category, count in sorted(self.stats['category_distribution'].items()):
            percentage = (count / self.stats['valid_samples'] * 100) if self.stats['valid_samples'] > 0 else 0
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        logger.info("=" * 60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_samples': self.stats['total_samples'],
            'valid_samples': self.stats['valid_samples'],
            'excluded_samples': self.stats['excluded_samples'],
            'missing_images': self.stats['missing_images'],
            'unsupported_categories': self.stats['unsupported_categories'],
            'category_distribution': dict(self.stats['category_distribution'])
        }
    
    def generate_report(self) -> str:
        """
        Generate a formatted validation report.
        
        Returns:
            Formatted report string
        """
        stats = self.get_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("DATASET VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal samples processed: {stats['total_samples']}")
        report.append(f"Valid samples: {stats['valid_samples']}")
        report.append(f"Excluded samples: {stats['excluded_samples']}")
        
        if stats['excluded_samples'] > 0:
            report.append(f"\nExclusion breakdown:")
            report.append(f"  Missing images: {stats['missing_images']}")
            report.append(f"  Unsupported categories: {stats['unsupported_categories']}")
        
        report.append(f"\nCategory distribution:")
        total_valid = stats['valid_samples']
        for category in sorted(stats['category_distribution'].keys()):
            count = stats['category_distribution'][category]
            percentage = (count / total_valid * 100) if total_valid > 0 else 0
            report.append(f"  {category}: {count} ({percentage:.1f}%)")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def validate_dataset(
    annotations: List,
    dataset_root: str
) -> tuple[List, Dict[str, Any]]:
    """
    Convenience function to validate dataset.
    
    Args:
        annotations: List of ImageAnnotation objects
        dataset_root: Root directory containing images
        
    Returns:
        Tuple of (filtered_annotations, statistics)
    """
    validator = DataValidator(dataset_root)
    return validator.validate_and_filter(annotations)
