"""Mask index metadata must not retain the image tensors from past batches."""
import ast
from contextlib import nullcontext
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import weakref

import numpy as np


class Tensor:
    def to(self, device):
        return self


class Predictions:
    def detach(self):
        return self

    float = cpu = detach

    def numpy(self):
        return np.ones((2, 1, 2, 2), dtype=np.float32)


class MaskMemoryTest(unittest.TestCase):
    def test_flushed_rows_release_tensors_on_success_and_write_failure(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks-ow/OW_00_generate_masks.ipynb").read_text())
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and "def flush_batch():" in "".join(c["source"]))
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "flush_batch")
        for write_ok in (True, False):
            with self.subTest(write_ok=write_ok), TemporaryDirectory() as output:
                env = dict(np=np, torch=SimpleNamespace(stack=lambda values: Tensor(), no_grad=nullcontext),
                           DEVICE=SimpleNamespace(type="cpu"), model=lambda batch: {"segmentation_prob": Predictions()},
                           cv2=SimpleNamespace(imwrite=lambda path, pred: write_ok), MASK_ROOT=Path(output),
                           thresh=0.5, rows=[], pending=[], n_new=0, TEACHER_NAME="teacher.pt")
                exec(compile(ast.Module(body=[function], type_ignores=[]), "flush_batch", "exec"), env)
                refs = []
                for batch in range(3):
                    env["pending"].extend({"tensor": Tensor(), "mask_file": f"{batch}-{i}.png"} for i in range(2))
                    refs.extend(weakref.ref(item["tensor"]) for item in env["pending"])
                    env["flush_batch"]()
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs), "Past image tensors are still retained")
                self.assertEqual(env["n_new"], 6)
                self.assertEqual(len(env["rows"]), 6)
                self.assertTrue(all("tensor" not in row for row in env["rows"]))
                self.assertTrue(all(row["status"] == ("ok" if write_ok else "error") for row in env["rows"]))


if __name__ == "__main__":
    unittest.main()
