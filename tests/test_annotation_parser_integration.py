"""
Integration tests for annotation parser with real OpenForensics dataset.

These tests verify the parser works correctly with actual dataset files.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.annotation_parser import AnnotationParser, parse_openforensics_json


def test_parse_train_annotations():
    """Test parsing Train_poly.json from actual dataset."""
    json_path = "dataset/Train_poly.json"
    
    if not os.path.exists(json_path):
        print(f"⚠ Skipping test: {json_path} not found")
        return
    
    print(f"\nTesting {json_path}...")
    
    parser = AnnotationParser(json_path)
    annotations = parser.parse()
    
    print(f"✓ Successfully parsed {len(annotations)} annotations")
    
    # Verify annotations structure
    assert len(annotations) > 0, "Should have at least one annotation"
    
    # Check first annotation
    first = annotations[0]
    assert hasattr(first, 'image_path')
    assert hasattr(first, 'label')
    assert hasattr(first, 'category')
    assert hasattr(first, 'bbox')
    assert hasattr(first, 'polygon')
    
    # Print statistics
    stats = parser.get_statistics()
    print("\nStatistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Authentic: {stats['authentic_count']}")
    print(f"  Manipulated: {stats['manipulated_count']}")
    print(f"  Parsing errors: {stats['parsing_errors']}")
    print(f"\n  Category distribution:")
    for category, count in sorted(stats['category_distribution'].items()):
        print(f"    {category}: {count}")
    
    # Sample annotations
    print(f"\nSample annotations:")
    for i, ann in enumerate(annotations[:3]):
        print(f"\n  Annotation {i+1}:")
        print(f"    Image: {ann.image_path}")
        print(f"    Label: {ann.label} ({ann.category})")
        print(f"    Size: {ann.image_width}x{ann.image_height}")
        print(f"    BBox: {ann.bbox}")
        if ann.polygon:
            print(f"    Polygon points: {len(ann.polygon)}")


def test_parse_val_annotations():
    """Test parsing Val_poly.json from actual dataset."""
    json_path = "dataset/Val_poly.json"
    
    if not os.path.exists(json_path):
        print(f"⚠ Skipping test: {json_path} not found")
        return
    
    print(f"\nTesting {json_path}...")
    
    parser = AnnotationParser(json_path)
    annotations = parser.parse()
    
    print(f"✓ Successfully parsed {len(annotations)} annotations")
    
    stats = parser.get_statistics()
    print(f"  Total images: {stats['total_images']}")
    print(f"  Authentic: {stats['authentic_count']}")
    print(f"  Manipulated: {stats['manipulated_count']}")


def test_parse_test_annotations():
    """Test parsing Test-Dev_poly.json from actual dataset."""
    json_path = "dataset/Test-Dev_poly.json"
    
    if not os.path.exists(json_path):
        print(f"⚠ Skipping test: {json_path} not found")
        return
    
    print(f"\nTesting {json_path}...")
    
    parser = AnnotationParser(json_path)
    annotations = parser.parse()
    
    print(f"✓ Successfully parsed {len(annotations)} annotations")
    
    stats = parser.get_statistics()
    print(f"  Total images: {stats['total_images']}")
    print(f"  Authentic: {stats['authentic_count']}")
    print(f"  Manipulated: {stats['manipulated_count']}")


def test_convenience_function_with_real_data():
    """Test convenience function with real dataset."""
    json_path = "dataset/Train_poly.json"
    
    if not os.path.exists(json_path):
        print(f"⚠ Skipping test: {json_path} not found")
        return
    
    print(f"\nTesting convenience function with {json_path}...")
    
    annotations = parse_openforensics_json(json_path)
    
    assert len(annotations) > 0
    print(f"✓ Convenience function parsed {len(annotations)} annotations")


def run_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("ANNOTATION PARSER INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Train annotations", test_parse_train_annotations),
        ("Val annotations", test_parse_val_annotations),
        ("Test annotations", test_parse_test_annotations),
        ("Convenience function", test_convenience_function_with_real_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, True))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("✓ All integration tests passed!" if all_passed else "✗ Some tests failed"))
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
