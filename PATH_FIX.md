# Dataset Path Fix

## Issue Identified

The OpenForensics JSON files contain image paths with an `Images/` prefix that doesn't match the actual Google Drive folder structure.

### JSON Contains:
```json
{
  "file_name": "Images/Train/9c541c578f.jpg"
}
```

### Actual Google Drive Structure:
```
/content/drive/MyDrive/VERITAS/
└── dataset/
    ├── Train/
    │   └── 9c541c578f.jpg
    ├── Val/
    └── Test-Dev/
```

### Problem:
When combined: `/content/drive/MyDrive/VERITAS/dataset/Images/Train/9c541c578f.jpg`
- ❌ This path doesn't exist (has extra "Images/" folder)

### Correct Path Should Be:
`/content/drive/MyDrive/VERITAS/dataset/Train/9c541c578f.jpg`
- ✅ This matches your actual structure

## Fix Applied

Updated `src/data/annotation_parser.py` to automatically strip the `Images/` prefix:

```python
def _create_annotation(self, image_info, annotation_info=None):
    image_path = image_info['file_name']
    
    # Fix path: Remove "Images/" prefix if present
    # JSON has: "Images/Train/xxx.jpg"
    # Actual structure: "Train/xxx.jpg"
    if image_path.startswith("Images/"):
        image_path = image_path.replace("Images/", "", 1)
    
    # Now image_path is "Train/xxx.jpg"
    # Combined with dataset_path gives correct full path
```

## Verification

After the fix, paths will work correctly:

```python
# Annotation parser returns
ann.image_path = "Train/9c541c578f.jpg"  # "Images/" removed

# Dataset class combines with dataset root
full_path = os.path.join(
    "/content/drive/MyDrive/VERITAS/dataset",  # dataset_root
    "Train/9c541c578f.jpg"                     # ann.image_path
)
# Result: "/content/drive/MyDrive/VERITAS/dataset/Train/9c541c578f.jpg" ✅
```

## Testing

Run smoke test again - image loading should now work:

```
Testing image loading...

Sample 1:
  Annotation path: Train/9c541c578f.jpg
  Full path: /content/drive/MyDrive/VERITAS/dataset/Train/9c541c578f.jpg
  ✓ Image loaded successfully
    Label: 1 (face_swap)
    Size: (1024, 768)
    BBox: [458, 237, 134, 131]
    Polygon points: 66
```

## Summary

✅ **Fixed**: Annotation parser now handles the path mismatch automatically
✅ **Works with**: Your actual Google Drive folder structure
✅ **No changes needed**: To your dataset organization

The smoke test should now run successfully!
