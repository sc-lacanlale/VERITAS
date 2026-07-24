"""
Test script to verify VERITAS project structure without requiring dependencies.

This test can run in any Python environment to verify the project structure.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_directory_structure():
    """Verify that all required directories exist."""
    print("Testing directory structure...")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    required_dirs = [
        'src',
        'src/utils',
        'tests',
        'dataset'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        exists = os.path.isdir(full_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✓ Directory structure test passed\n")
    else:
        print("\n✗ Some directories missing\n")
    
    return all_exist


def test_module_files():
    """Verify that all required module files exist."""
    print("Testing module files...")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/utils/__init__.py',
        'src/utils/reproducibility.py',
        'src/utils/environment.py',
        'veritas_setup.ipynb',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        exists = os.path.isfile(full_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✓ Module files test passed\n")
    else:
        print("\n✗ Some module files missing\n")
    
    return all_exist


def test_module_imports():
    """Test that modules can be imported (without external dependencies)."""
    print("Testing module imports...")
    
    try:
        # Test importing config module structure
        import src
        print("  ✓ src package")
        
        import src.utils
        print("  ✓ src.utils package")
        
        # Check that functions are defined (without calling them)
        from src.config import Config, create_default_config, setup_directories
        print("  ✓ src.config exports")
        
        # Try to import set_random_seed (may fail if torch not installed)
        try:
            from src.utils import set_random_seed
            print("  ✓ src.utils.set_random_seed")
        except ImportError:
            print("  ⚠ src.utils.set_random_seed (requires torch - expected in Colab)")
        
        print("\n✓ Module imports test passed\n")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}\n")
        return False


def test_notebook_format():
    """Verify notebook is valid JSON."""
    print("Testing notebook format...")
    
    import json
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    notebook_path = os.path.join(base_dir, 'veritas_setup.ipynb')
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Check basic notebook structure
        assert 'cells' in notebook, "Notebook missing 'cells'"
        assert 'metadata' in notebook, "Notebook missing 'metadata'"
        assert 'nbformat' in notebook, "Notebook missing 'nbformat'"
        
        print(f"  ✓ Notebook is valid JSON")
        print(f"  ✓ Contains {len(notebook['cells'])} cells")
        print(f"  ✓ Format version: {notebook['nbformat']}")
        
        print("\n✓ Notebook format test passed\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Notebook test failed: {e}\n")
        return False


def run_structure_tests():
    """Run all structure tests."""
    print("=" * 60)
    print("VERITAS PROJECT STRUCTURE VERIFICATION")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Module Files", test_module_files()))
    results.append(("Module Imports", test_module_imports()))
    results.append(("Notebook Format", test_notebook_format()))
    
    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print()
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All structure tests passed!")
        print("\nNext steps:")
        print("  1. Open veritas_setup.ipynb in Google Colab")
        print("  2. Run all cells to complete environment setup")
        print("  3. Install dependencies and verify GPU access")
    else:
        print("✗ Some tests failed. Please review errors above.")
    
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_structure_tests()
    sys.exit(0 if success else 1)
