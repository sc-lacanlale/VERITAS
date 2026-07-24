"""
Quick verification script for annotation parser.
"""

import sys
import logging
from src.data.annotation_parser import parse_openforensics_json

# Configure logging to show progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("Testing annotation parser with Train_poly.json...")
    print("This may take a moment for large files...\n")
    
    try:
        annotations = parse_openforensics_json('dataset/Train_poly.json')
        
        print(f"\n[OK] Successfully parsed {len(annotations)} annotations!")
        
        # Show sample
        if len(annotations) > 0:
            print("\nSample annotation:")
            ann = annotations[0]
            print(f"  Image: {ann.image_path}")
            print(f"  Label: {ann.label} ({ann.category})")
            print(f"  Size: {ann.image_width}x{ann.image_height}")
            print(f"  BBox: {ann.bbox}")
            if ann.polygon:
                print(f"  Polygon: {len(ann.polygon)} points")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
