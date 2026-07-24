"""Quick test to verify parser works with all dataset splits."""

from src.data import parse_openforensics_json

splits = ['Train', 'Val', 'Test-Dev']

print("\nTesting all dataset splits:")
print("-" * 50)

for split in splits:
    json_path = f'dataset/{split}_poly.json'
    annotations = parse_openforensics_json(json_path)
    print(f"{split:10} {len(annotations):6} annotations")

print("-" * 50)
print("All splits parsed successfully!\n")
