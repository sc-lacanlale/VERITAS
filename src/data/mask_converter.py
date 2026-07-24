"""
Polygon to binary mask converter for VERITAS.

Converts polygon coordinates from OpenForensics annotations to binary 
segmentation masks for training.

Requirements: 1.9
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def polygon_to_mask(
    polygon: Optional[List[List[float]]],
    image_width: int,
    image_height: int,
    validate: bool = True
) -> np.ndarray:
    """
    Convert polygon coordinates to binary mask.
    
    Args:
        polygon: List of [x, y] coordinate pairs, or None for authentic images
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        validate: If True, validate polygon coordinates are within bounds
        
    Returns:
        Binary mask as numpy array (H, W) with values 0 or 1
        - 0 = authentic/background
        - 1 = manipulated region
        
    Example:
        >>> polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
        >>> mask = polygon_to_mask(polygon, 1024, 768)
        >>> assert mask.shape == (768, 1024)
        >>> assert mask.dtype == np.uint8
    """
    # Create empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # If no polygon (authentic image), return all-zero mask
    if polygon is None or len(polygon) == 0:
        return mask
    
    # Convert polygon to numpy array
    polygon_array = np.array(polygon, dtype=np.int32)
    
    # Validate coordinates are within bounds
    if validate:
        if np.any(polygon_array[:, 0] < 0) or np.any(polygon_array[:, 0] >= image_width):
            logger.warning(
                f"Polygon x-coordinates out of bounds [0, {image_width}): "
                f"min={polygon_array[:, 0].min()}, max={polygon_array[:, 0].max()}"
            )
            # Clip to valid range
            polygon_array[:, 0] = np.clip(polygon_array[:, 0], 0, image_width - 1)
        
        if np.any(polygon_array[:, 1] < 0) or np.any(polygon_array[:, 1] >= image_height):
            logger.warning(
                f"Polygon y-coordinates out of bounds [0, {image_height}): "
                f"min={polygon_array[:, 1].min()}, max={polygon_array[:, 1].max()}"
            )
            # Clip to valid range
            polygon_array[:, 1] = np.clip(polygon_array[:, 1], 0, image_height - 1)
    
    # Fill polygon using OpenCV
    cv2.fillPoly(mask, [polygon_array], color=1)
    
    return mask


def create_mask_from_annotation(
    annotation,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Create binary mask from ImageAnnotation object.
    
    Args:
        annotation: ImageAnnotation object from annotation parser
        target_size: Optional (width, height) to resize mask. If None, uses original size.
        
    Returns:
        Binary mask as numpy array (H, W) with values 0 or 1
    """
    # Get original image dimensions
    orig_width = annotation.image_width
    orig_height = annotation.image_height
    
    # Create mask at original resolution
    mask = polygon_to_mask(
        annotation.polygon,
        orig_width,
        orig_height,
        validate=True
    )
    
    # Resize if target size specified
    if target_size is not None:
        target_width, target_height = target_size
        if (orig_width, orig_height) != (target_width, target_height):
            # Use nearest neighbor interpolation to preserve binary values
            mask = cv2.resize(
                mask,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST
            )
    
    return mask


def batch_create_masks(
    annotations: List,
    target_size: Optional[Tuple[int, int]] = None
) -> List[np.ndarray]:
    """
    Create masks for a batch of annotations.
    
    Args:
        annotations: List of ImageAnnotation objects
        target_size: Optional (width, height) to resize masks
        
    Returns:
        List of binary masks
    """
    masks = []
    
    for ann in annotations:
        mask = create_mask_from_annotation(ann, target_size)
        masks.append(mask)
    
    return masks


def validate_mask(mask: np.ndarray) -> bool:
    """
    Validate that mask contains only binary values (0 or 1).
    
    Args:
        mask: Mask array to validate
        
    Returns:
        True if valid, False otherwise
    """
    unique_values = np.unique(mask)
    valid = np.all(np.isin(unique_values, [0, 1]))
    
    if not valid:
        logger.error(f"Mask contains invalid values: {unique_values}")
    
    return valid
