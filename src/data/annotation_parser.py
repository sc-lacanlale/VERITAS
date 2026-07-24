"""
Annotation parser for OpenForensics JSON format.

This module provides utilities to parse OpenForensics dataset annotation files
and extract image metadata, labels, bounding boxes, and polygon coordinates.

Requirements: 1.3, 1.4, 1.5
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


# Configure logger
logger = logging.getLogger(__name__)


# Category mapping from OpenForensics
CATEGORY_MAPPING = {
    0: "authentic",
    1: "manipulated"
}

# Supported manipulation categories (from requirements)
SUPPORTED_CATEGORIES = [
    "authentic",
    "face_swap",
    "face_reenact",
    "inpainting"
]


@dataclass
class ImageAnnotation:
    """
    Container for parsed image annotation data.
    
    Attributes:
        image_path: Relative path to image file
        label: Binary label (0=authentic, 1=manipulated)
        category: Manipulation category (authentic, face_swap, face_reenact, inpainting)
        bbox: Bounding box coordinates [x, y, width, height] or None if authentic
        polygon: List of polygon coordinates [[x1,y1], [x2,y2], ...] or None if authentic
        image_width: Original image width in pixels
        image_height: Original image height in pixels
        image_id: Unique image identifier
        annotation_id: Unique annotation identifier (None if authentic)
    """
    image_path: str
    label: int
    category: str
    bbox: Optional[List[float]]
    polygon: Optional[List[List[float]]]
    image_width: int
    image_height: int
    image_id: int
    annotation_id: Optional[int]
    
    def __post_init__(self):
        """Validate annotation data after initialization."""
        if self.label not in [0, 1]:
            raise ValueError(f"Invalid label: {self.label}. Must be 0 or 1.")
        
        if self.category not in SUPPORTED_CATEGORIES:
            logger.warning(
                f"Category '{self.category}' not in supported categories: {SUPPORTED_CATEGORIES}"
            )
        
        if self.label == 1 and self.bbox is None:
            logger.warning(
                f"Manipulated image (id={self.image_id}) missing bounding box"
            )
        
        if self.label == 1 and self.polygon is None:
            logger.warning(
                f"Manipulated image (id={self.image_id}) missing polygon annotation"
            )


class AnnotationParser:
    """
    Parser for OpenForensics JSON annotation files.
    
    The OpenForensics dataset uses COCO-style JSON format with:
    - "categories": List of category definitions
    - "images": List of image metadata
    - "annotations": List of annotations with bounding boxes and polygons
    
    Example:
        >>> parser = AnnotationParser('dataset/Train_poly.json')
        >>> annotations = parser.parse()
        >>> print(f"Loaded {len(annotations)} annotations")
    """
    
    def __init__(self, json_path: str):
        """
        Initialize annotation parser.
        
        Args:
            json_path: Path to OpenForensics JSON annotation file
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
        """
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
        
        self.annotations_data = None
        self.images_lookup = {}
        self.categories_lookup = {}
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'authentic_count': 0,
            'manipulated_count': 0,
            'missing_annotations': 0,
            'parsing_errors': 0,
            'category_distribution': {}
        }
    
    def load_json(self) -> Dict[str, Any]:
        """
        Load and parse JSON file.
        
        Returns:
            Parsed JSON data as dictionary
            
        Raises:
            json.JSONDecodeError: If JSON file is malformed
        """
        logger.info(f"Loading annotation file: {self.json_path}")
        
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Successfully loaded JSON with {len(data.get('images', []))} images")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file: {e}")
            raise
    
    def _build_lookups(self, data: Dict[str, Any]) -> None:
        """
        Build lookup dictionaries for fast access.
        
        Args:
            data: Parsed JSON data
        """
        # Build categories lookup
        for category in data.get('categories', []):
            cat_id = category['id']
            cat_name = category.get('name', 'unknown')
            self.categories_lookup[cat_id] = cat_name
        
        logger.info(f"Loaded {len(self.categories_lookup)} categories: {self.categories_lookup}")
        
        # Build images lookup
        for image_info in data.get('images', []):
            img_id = image_info['id']
            self.images_lookup[img_id] = image_info
        
        logger.info(f"Built lookup for {len(self.images_lookup)} images")
    
    def _map_category(self, category_id: int) -> str:
        """
        Map category ID to manipulation category name.
        
        Args:
            category_id: Category ID from annotation
            
        Returns:
            Category name (authentic, face_swap, face_reenact, inpainting)
        """
        # For now, we map based on binary categories (Real=0, Fake=1)
        # Future enhancement: extract specific manipulation types from metadata
        if category_id == 0:
            return "authentic"
        else:
            # Default to generic "manipulated" - would need metadata for specific types
            # In a real implementation, we'd check image metadata for manipulation type
            return "face_swap"  # Placeholder - should be determined from metadata
    
    def _parse_polygon(self, segmentation: List[List[float]]) -> List[List[float]]:
        """
        Parse polygon segmentation data.
        
        OpenForensics uses COCO format where segmentation is a list of lists.
        Each inner list contains [x1, y1, x2, y2, x3, y3, ...] coordinates.
        
        Args:
            segmentation: Segmentation data from annotation
            
        Returns:
            List of [x, y] coordinate pairs
        """
        if not segmentation or len(segmentation) == 0:
            return None
        
        # Take first polygon (some annotations may have multiple)
        flat_coords = segmentation[0]
        
        # Convert flat list [x1, y1, x2, y2, ...] to [[x1, y1], [x2, y2], ...]
        polygon = []
        for i in range(0, len(flat_coords), 2):
            if i + 1 < len(flat_coords):
                polygon.append([flat_coords[i], flat_coords[i + 1]])
        
        return polygon if polygon else None
    
    def _create_annotation(
        self,
        image_info: Dict[str, Any],
        annotation_info: Optional[Dict[str, Any]] = None
    ) -> Optional[ImageAnnotation]:
        """
        Create ImageAnnotation object from image and annotation data.
        
        Args:
            image_info: Image metadata dictionary
            annotation_info: Annotation dictionary (None for authentic images)
            
        Returns:
            ImageAnnotation object or None if parsing fails
        """
        try:
            image_id = image_info['id']
            image_path = image_info['file_name']
            
            # Fix path: Remove "Images/" prefix if present
            # JSON has: "Images/Train/xxx.jpg"
            # Actual structure: "Train/xxx.jpg"
            if image_path.startswith("Images/"):
                image_path = image_path.replace("Images/", "", 1)
            
            image_width = image_info['width']
            image_height = image_info['height']
            
            # If no annotation, this is an authentic image
            if annotation_info is None:
                return ImageAnnotation(
                    image_path=image_path,
                    label=0,
                    category="authentic",
                    bbox=None,
                    polygon=None,
                    image_width=image_width,
                    image_height=image_height,
                    image_id=image_id,
                    annotation_id=None
                )
            
            # Parse manipulated image annotation
            annotation_id = annotation_info['id']
            category_id = annotation_info['category_id']
            bbox = annotation_info.get('bbox', None)
            segmentation = annotation_info.get('segmentation', [])
            
            # Map category
            category = self._map_category(category_id)
            
            # Parse polygon
            polygon = self._parse_polygon(segmentation)
            
            # Determine label
            label = 0 if category_id == 0 else 1
            
            return ImageAnnotation(
                image_path=image_path,
                label=label,
                category=category,
                bbox=bbox,
                polygon=polygon,
                image_width=image_width,
                image_height=image_height,
                image_id=image_id,
                annotation_id=annotation_id
            )
            
        except KeyError as e:
            logger.error(f"Missing required field in annotation: {e}")
            self.stats['parsing_errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Error creating annotation for image {image_info.get('id', 'unknown')}: {e}")
            self.stats['parsing_errors'] += 1
            return None
    
    def parse(self) -> List[ImageAnnotation]:
        """
        Parse all annotations from JSON file.
        
        Returns:
            List of ImageAnnotation objects
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON file is malformed
        """
        # Load JSON data
        data = self.load_json()
        self.annotations_data = data
        
        # Build lookup dictionaries
        self._build_lookups(data)
        
        # Parse annotations
        parsed_annotations = []
        
        # Create lookup of image_id -> list of annotations
        annotations_by_image = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Process each image
        for image_id, image_info in self.images_lookup.items():
            self.stats['total_images'] += 1
            
            # Check if image has annotations
            if image_id in annotations_by_image:
                # Process each annotation for this image
                for ann_info in annotations_by_image[image_id]:
                    self.stats['total_annotations'] += 1
                    
                    annotation = self._create_annotation(image_info, ann_info)
                    if annotation is not None:
                        parsed_annotations.append(annotation)
                        
                        # Update statistics
                        if annotation.label == 0:
                            self.stats['authentic_count'] += 1
                        else:
                            self.stats['manipulated_count'] += 1
                        
                        # Update category distribution
                        category = annotation.category
                        self.stats['category_distribution'][category] = \
                            self.stats['category_distribution'].get(category, 0) + 1
            else:
                # No annotation means authentic image
                self.stats['missing_annotations'] += 1
                annotation = self._create_annotation(image_info, None)
                if annotation is not None:
                    parsed_annotations.append(annotation)
                    self.stats['authentic_count'] += 1
                    self.stats['category_distribution']['authentic'] = \
                        self.stats['category_distribution'].get('authentic', 0) + 1
        
        logger.info(f"Successfully parsed {len(parsed_annotations)} annotations")
        self._log_statistics()
        
        return parsed_annotations
    
    def _log_statistics(self) -> None:
        """Log parsing statistics."""
        logger.info("=" * 60)
        logger.info("ANNOTATION PARSING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {self.stats['total_images']}")
        logger.info(f"Total annotations: {self.stats['total_annotations']}")
        logger.info(f"Authentic images: {self.stats['authentic_count']}")
        logger.info(f"Manipulated images: {self.stats['manipulated_count']}")
        logger.info(f"Images without annotations: {self.stats['missing_annotations']}")
        logger.info(f"Parsing errors: {self.stats['parsing_errors']}")
        logger.info(f"\nCategory distribution:")
        for category, count in sorted(self.stats['category_distribution'].items()):
            logger.info(f"  {category}: {count}")
        logger.info("=" * 60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get parsing statistics.
        
        Returns:
            Dictionary with parsing statistics
        """
        return self.stats.copy()


def parse_openforensics_json(json_path: str) -> List[ImageAnnotation]:
    """
    Convenience function to parse OpenForensics JSON file.
    
    Args:
        json_path: Path to JSON annotation file
        
    Returns:
        List of ImageAnnotation objects
        
    Example:
        >>> annotations = parse_openforensics_json('dataset/Train_poly.json')
        >>> print(f"Loaded {len(annotations)} annotations")
    """
    parser = AnnotationParser(json_path)
    return parser.parse()
