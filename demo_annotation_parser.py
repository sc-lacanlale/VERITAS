"""
Demo script showing annotation parser capabilities.

This demonstrates how to use the OpenForensics annotation parser
to extract image metadata, labels, bounding boxes, and polygon coordinates.
"""

import logging
from src.data.annotation_parser import AnnotationParser, parse_openforensics_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def demo_basic_usage():
    """Demonstrate basic usage of the annotation parser."""
    print("=" * 70)
    print("DEMO: OpenForensics Annotation Parser")
    print("=" * 70)
    print()
    
    # Parse annotations
    print("Parsing Train_poly.json...")
    parser = AnnotationParser('dataset/Train_poly.json')
    annotations = parser.parse()
    
    print(f"\n[OK] Loaded {len(annotations)} total annotations\n")
    
    # Get statistics
    stats = parser.get_statistics()
    
    print("Dataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Authentic images: {stats['authentic_count']}")
    print(f"  Manipulated images: {stats['manipulated_count']}")
    print(f"  Parsing errors: {stats['parsing_errors']}")
    
    print(f"\nCategory distribution:")
    for category, count in sorted(stats['category_distribution'].items()):
        percentage = (count / len(annotations)) * 100
        print(f"  {category:15} {count:6} ({percentage:5.2f}%)")
    
    # Show sample annotations
    print("\n" + "=" * 70)
    print("Sample Annotations")
    print("=" * 70)
    
    # Find one authentic and one manipulated
    authentic = next((a for a in annotations if a.label == 0), None)
    manipulated = next((a for a in annotations if a.label == 1), None)
    
    if authentic:
        print("\n1. Authentic Image Example:")
        print(f"   Path: {authentic.image_path}")
        print(f"   Label: {authentic.label} ({authentic.category})")
        print(f"   Dimensions: {authentic.image_width}x{authentic.image_height}")
        print(f"   Bounding box: {authentic.bbox}")
        print(f"   Polygon: {authentic.polygon}")
    
    if manipulated:
        print("\n2. Manipulated Image Example:")
        print(f"   Path: {manipulated.image_path}")
        print(f"   Label: {manipulated.label} ({manipulated.category})")
        print(f"   Dimensions: {manipulated.image_width}x{manipulated.image_height}")
        print(f"   Bounding box: {manipulated.bbox}")
        if manipulated.polygon:
            print(f"   Polygon: {len(manipulated.polygon)} coordinate pairs")
            print(f"   First 3 points: {manipulated.polygon[:3]}")
    
    # Access individual annotations
    print("\n" + "=" * 70)
    print("Accessing Annotation Data")
    print("=" * 70)
    
    ann = annotations[0]
    print(f"\nAnnotation fields:")
    print(f"  ann.image_path    : {ann.image_path}")
    print(f"  ann.label         : {ann.label}")
    print(f"  ann.category      : {ann.category}")
    print(f"  ann.bbox          : {ann.bbox}")
    print(f"  ann.image_width   : {ann.image_width}")
    print(f"  ann.image_height  : {ann.image_height}")
    print(f"  ann.image_id      : {ann.image_id}")
    print(f"  ann.annotation_id : {ann.annotation_id}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


def demo_convenience_function():
    """Demonstrate convenience function usage."""
    print("\n\nUsing convenience function:")
    print("-" * 70)
    
    # Simple one-liner to parse annotations
    annotations = parse_openforensics_json('dataset/Val_poly.json')
    
    print(f"[OK] Parsed {len(annotations)} annotations from Val set")
    
    # Count by category
    authentic = sum(1 for a in annotations if a.label == 0)
    manipulated = sum(1 for a in annotations if a.label == 1)
    
    print(f"  Authentic: {authentic}")
    print(f"  Manipulated: {manipulated}")


def demo_filtering():
    """Demonstrate filtering annotations."""
    print("\n\nFiltering annotations:")
    print("-" * 70)
    
    annotations = parse_openforensics_json('dataset/Train_poly.json')
    
    # Filter only manipulated images
    manipulated = [a for a in annotations if a.label == 1]
    print(f"Manipulated images: {len(manipulated)}")
    
    # Filter by image size
    large_images = [a for a in annotations if a.image_width >= 1024 and a.image_height >= 1024]
    print(f"Images >= 1024x1024: {len(large_images)}")
    
    # Filter images with large manipulation regions (area >= 50000)
    large_manip = [a for a in manipulated if a.bbox and a.bbox[2] * a.bbox[3] >= 50000]
    print(f"Large manipulation regions (area >= 50k): {len(large_manip)}")


if __name__ == "__main__":
    try:
        demo_basic_usage()
        demo_convenience_function()
        demo_filtering()
        
        print("\n\n[OK] All demos completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
