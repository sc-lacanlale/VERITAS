"""
Unit tests for OpenForensics annotation parser.

Tests Requirements: 1.3, 1.4, 1.5
"""

import sys
import os
import json
import tempfile
import pytest
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.annotation_parser import (
    AnnotationParser,
    ImageAnnotation,
    parse_openforensics_json,
    CATEGORY_MAPPING,
    SUPPORTED_CATEGORIES
)


class TestImageAnnotation:
    """Test ImageAnnotation dataclass."""
    
    def test_valid_authentic_annotation(self):
        """Test creating valid authentic image annotation."""
        ann = ImageAnnotation(
            image_path="Images/Train/test.jpg",
            label=0,
            category="authentic",
            bbox=None,
            polygon=None,
            image_width=1024,
            image_height=768,
            image_id=1,
            annotation_id=None
        )
        
        assert ann.label == 0
        assert ann.category == "authentic"
        assert ann.bbox is None
        assert ann.polygon is None
    
    def test_valid_manipulated_annotation(self):
        """Test creating valid manipulated image annotation."""
        ann = ImageAnnotation(
            image_path="Images/Train/test.jpg",
            label=1,
            category="face_swap",
            bbox=[100, 200, 50, 60],
            polygon=[[100, 200], [150, 200], [150, 260], [100, 260]],
            image_width=1024,
            image_height=768,
            image_id=1,
            annotation_id=100
        )
        
        assert ann.label == 1
        assert ann.category == "face_swap"
        assert ann.bbox == [100, 200, 50, 60]
        assert len(ann.polygon) == 4
    
    def test_invalid_label(self):
        """Test that invalid label raises ValueError."""
        with pytest.raises(ValueError, match="Invalid label"):
            ImageAnnotation(
                image_path="test.jpg",
                label=2,  # Invalid
                category="authentic",
                bbox=None,
                polygon=None,
                image_width=1024,
                image_height=768,
                image_id=1,
                annotation_id=None
            )


class TestAnnotationParser:
    """Test AnnotationParser class."""
    
    @pytest.fixture
    def sample_json_data(self):
        """Create sample OpenForensics JSON data."""
        return {
            "categories": [
                {"id": 0, "name": "Real"},
                {"id": 1, "name": "Fake"}
            ],
            "images": [
                {
                    "id": 0,
                    "file_name": "Images/Train/image1.jpg",
                    "width": 1024,
                    "height": 768
                },
                {
                    "id": 1,
                    "file_name": "Images/Train/image2.jpg",
                    "width": 800,
                    "height": 600
                },
                {
                    "id": 2,
                    "file_name": "Images/Train/image3.jpg",
                    "width": 1024,
                    "height": 1024
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 150, 200, 250],
                    "segmentation": [[
                        100, 150,
                        300, 150,
                        300, 400,
                        100, 400
                    ]],
                    "area": 50000,
                    "iscrowd": 0
                }
            ]
        }
    
    @pytest.fixture
    def temp_json_file(self, sample_json_data):
        """Create temporary JSON file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_parser_initialization(self, temp_json_file):
        """Test parser initialization with valid file."""
        parser = AnnotationParser(temp_json_file)
        assert parser.json_path.exists()
        assert parser.annotations_data is None
    
    def test_parser_initialization_missing_file(self):
        """Test parser initialization with missing file."""
        with pytest.raises(FileNotFoundError):
            AnnotationParser("nonexistent_file.json")
    
    def test_load_json(self, temp_json_file, sample_json_data):
        """Test JSON loading."""
        parser = AnnotationParser(temp_json_file)
        data = parser.load_json()
        
        assert 'categories' in data
        assert 'images' in data
        assert 'annotations' in data
        assert len(data['images']) == 3
        assert len(data['annotations']) == 1
    
    def test_build_lookups(self, temp_json_file, sample_json_data):
        """Test building lookup dictionaries."""
        parser = AnnotationParser(temp_json_file)
        data = parser.load_json()
        parser._build_lookups(data)
        
        # Check categories lookup
        assert len(parser.categories_lookup) == 2
        assert parser.categories_lookup[0] == "Real"
        assert parser.categories_lookup[1] == "Fake"
        
        # Check images lookup
        assert len(parser.images_lookup) == 3
        assert 0 in parser.images_lookup
        assert 1 in parser.images_lookup
        assert 2 in parser.images_lookup
    
    def test_parse_polygon(self, temp_json_file):
        """Test polygon parsing."""
        parser = AnnotationParser(temp_json_file)
        
        # Test valid polygon
        segmentation = [[100, 150, 200, 150, 200, 250, 100, 250]]
        polygon = parser._parse_polygon(segmentation)
        
        assert polygon is not None
        assert len(polygon) == 4
        assert polygon[0] == [100, 150]
        assert polygon[1] == [200, 150]
        assert polygon[2] == [200, 250]
        assert polygon[3] == [100, 250]
    
    def test_parse_polygon_empty(self, temp_json_file):
        """Test parsing empty polygon."""
        parser = AnnotationParser(temp_json_file)
        
        # Test empty segmentation
        polygon = parser._parse_polygon([])
        assert polygon is None
        
        # Test None segmentation
        polygon = parser._parse_polygon(None)
        assert polygon is None
    
    def test_create_authentic_annotation(self, temp_json_file):
        """Test creating annotation for authentic image."""
        parser = AnnotationParser(temp_json_file)
        
        image_info = {
            'id': 0,
            'file_name': 'test.jpg',
            'width': 1024,
            'height': 768
        }
        
        ann = parser._create_annotation(image_info, None)
        
        assert ann is not None
        assert ann.label == 0
        assert ann.category == "authentic"
        assert ann.bbox is None
        assert ann.polygon is None
        assert ann.image_id == 0
        assert ann.annotation_id is None
    
    def test_create_manipulated_annotation(self, temp_json_file):
        """Test creating annotation for manipulated image."""
        parser = AnnotationParser(temp_json_file)
        
        image_info = {
            'id': 1,
            'file_name': 'test.jpg',
            'width': 1024,
            'height': 768
        }
        
        annotation_info = {
            'id': 10,
            'image_id': 1,
            'category_id': 1,
            'bbox': [100, 200, 50, 60],
            'segmentation': [[100, 200, 150, 200, 150, 260, 100, 260]]
        }
        
        ann = parser._create_annotation(image_info, annotation_info)
        
        assert ann is not None
        assert ann.label == 1
        assert ann.bbox == [100, 200, 50, 60]
        assert ann.polygon is not None
        assert len(ann.polygon) == 4
        assert ann.image_id == 1
        assert ann.annotation_id == 10
    
    def test_create_annotation_missing_fields(self, temp_json_file):
        """Test annotation creation with missing fields."""
        parser = AnnotationParser(temp_json_file)
        
        # Missing required 'id' field
        image_info = {
            'file_name': 'test.jpg',
            'width': 1024,
            'height': 768
        }
        
        ann = parser._create_annotation(image_info, None)
        
        assert ann is None
        assert parser.stats['parsing_errors'] == 1
    
    def test_parse_full(self, temp_json_file):
        """Test full parsing of JSON file."""
        parser = AnnotationParser(temp_json_file)
        annotations = parser.parse()
        
        # Should have 3 annotations total:
        # - 1 manipulated (image_id=1, with annotation)
        # - 2 authentic (image_id=0,2, without annotations)
        assert len(annotations) == 3
        
        # Check statistics
        stats = parser.get_statistics()
        assert stats['total_images'] == 3
        assert stats['total_annotations'] == 1
        assert stats['authentic_count'] == 2
        assert stats['manipulated_count'] == 1
        assert stats['missing_annotations'] == 2
    
    def test_parse_with_malformed_json(self):
        """Test parsing with malformed JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json")
            temp_path = f.name
        
        try:
            parser = AnnotationParser(temp_path)
            with pytest.raises(json.JSONDecodeError):
                parser.parse()
        finally:
            os.unlink(temp_path)
    
    def test_get_statistics(self, temp_json_file):
        """Test statistics retrieval."""
        parser = AnnotationParser(temp_json_file)
        annotations = parser.parse()
        
        stats = parser.get_statistics()
        
        assert 'total_images' in stats
        assert 'authentic_count' in stats
        assert 'manipulated_count' in stats
        assert 'category_distribution' in stats
        assert stats['total_images'] == 3


class TestConvenienceFunction:
    """Test parse_openforensics_json convenience function."""
    
    @pytest.fixture
    def sample_json_file(self):
        """Create sample JSON file."""
        data = {
            "categories": [
                {"id": 0, "name": "Real"},
                {"id": 1, "name": "Fake"}
            ],
            "images": [
                {
                    "id": 0,
                    "file_name": "test.jpg",
                    "width": 1024,
                    "height": 768
                }
            ],
            "annotations": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_convenience_function(self, sample_json_file):
        """Test convenience function."""
        annotations = parse_openforensics_json(sample_json_file)
        
        assert isinstance(annotations, list)
        assert len(annotations) == 1
        assert isinstance(annotations[0], ImageAnnotation)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_json(self):
        """Test parsing empty JSON structure."""
        data = {
            "categories": [],
            "images": [],
            "annotations": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            parser = AnnotationParser(temp_path)
            annotations = parser.parse()
            
            assert len(annotations) == 0
            assert parser.stats['total_images'] == 0
        finally:
            os.unlink(temp_path)
    
    def test_multiple_annotations_per_image(self):
        """Test handling multiple annotations for single image."""
        data = {
            "categories": [
                {"id": 0, "name": "Real"},
                {"id": 1, "name": "Fake"}
            ],
            "images": [
                {
                    "id": 0,
                    "file_name": "test.jpg",
                    "width": 1024,
                    "height": 768
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [100, 100, 50, 50],
                    "segmentation": [[100, 100, 150, 100, 150, 150, 100, 150]]
                },
                {
                    "id": 1,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [200, 200, 50, 50],
                    "segmentation": [[200, 200, 250, 200, 250, 250, 200, 250]]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            parser = AnnotationParser(temp_path)
            annotations = parser.parse()
            
            # Should have 2 annotations for the same image
            assert len(annotations) == 2
            assert annotations[0].image_id == 0
            assert annotations[1].image_id == 0
        finally:
            os.unlink(temp_path)
    
    def test_missing_bbox(self):
        """Test annotation with missing bounding box."""
        data = {
            "categories": [{"id": 1, "name": "Fake"}],
            "images": [
                {
                    "id": 0,
                    "file_name": "test.jpg",
                    "width": 1024,
                    "height": 768
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            parser = AnnotationParser(temp_path)
            annotations = parser.parse()
            
            # Should still parse successfully
            assert len(annotations) == 1
            assert annotations[0].bbox is None
        finally:
            os.unlink(temp_path)
    
    def test_missing_polygon(self):
        """Test annotation with missing polygon."""
        data = {
            "categories": [{"id": 1, "name": "Fake"}],
            "images": [
                {
                    "id": 0,
                    "file_name": "test.jpg",
                    "width": 1024,
                    "height": 768
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [100, 100, 50, 50]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            parser = AnnotationParser(temp_path)
            annotations = parser.parse()
            
            # Should still parse successfully
            assert len(annotations) == 1
            assert annotations[0].polygon is None
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
