"""Kaggle can bundle a large notebook output into _output_.zip."""
import ast
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile


class MaskArchiveTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root / "notebooks-ow/OW_02_multistream.ipynb").read_text(encoding="utf-8"))
        code = next("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code" and "def unpack_mask_outputs(" in "".join(c["source"]))
        fn = next(n for n in ast.parse(code).body
                  if isinstance(n, ast.FunctionDef) and n.name == "unpack_mask_outputs")
        namespace = dict(Path=Path, shutil=shutil)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "unpack_mask_outputs", "exec"), namespace)
        self.unpack = namespace["unpack_mask_outputs"]

    def test_extracts_masks_and_index_from_wrapped_output(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            inputs = base / "inputs"
            inputs.mkdir()
            with zipfile.ZipFile(inputs / "_output_.zip", "w") as archive:
                archive.writestr("output/ow_masks/index.csv", "image_id,mask_file\n1,masks/one.png\n")
                archive.writestr("output/ow_masks/masks/one.png", b"mask bytes")
                archive.writestr("checkpoints/unrelated.pt", b"unused")
            make_temp = tempfile.mkdtemp
            with patch("tempfile.mkdtemp", side_effect=lambda **kwargs: make_temp(dir=base, **kwargs)):
                roots = self.unpack(inputs)
            self.assertEqual(len(roots), 1)
            self.assertTrue((roots[0] / "ow_masks/index.csv").is_file())
            self.assertEqual((roots[0] / "ow_masks/masks/one.png").read_bytes(), b"mask bytes")
            self.assertFalse((roots[0] / "checkpoints").exists())

    def test_rejects_paths_outside_the_archive_root(self):
        for name in ("../ow_masks/evil.png", "/ow_masks/evil.png", "..\\ow_masks\\evil.png"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                inputs = base / "inputs"
                inputs.mkdir()
                with zipfile.ZipFile(inputs / "_output_.zip", "w") as archive:
                    archive.writestr(name, b"bad")
                make_temp = tempfile.mkdtemp
                with patch("tempfile.mkdtemp", side_effect=lambda **kwargs: make_temp(dir=base, **kwargs)):
                    with self.assertRaises(ValueError):
                        self.unpack(inputs)


if __name__ == "__main__":
    unittest.main()
