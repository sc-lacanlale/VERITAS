"""Continue from tested pipeline: one dataset item + one model forward."""
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import matplotlib
matplotlib.use("Agg")
import traceback
from pathlib import Path

import nbformat
import torch

nb = nbformat.read("VERITAS_Kaggle.ipynb", as_version=4)
ns = {"__name__": "__main__"}
stop_after = "TEST_RESULTS = run_unit_tests()"

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = "\n".join(ln for ln in cell.source.splitlines() if not ln.strip().startswith("%"))
    exec(compile(src, f"nb_cell_{i}", "exec"), ns, ns)
    if stop_after in cell.source:
        break

print("getitem...")
item = ns["TRAIN_LOADER"].dataset[0]
print("image", tuple(item["image"].shape), "mask", tuple(item["mask"].shape), "label", float(item["label"]))
assert item["image"].shape[0] == 3
assert item["mask"].shape[0] == 1

print("build baseline...")
m = ns["build_model"](ns["EXPERIMENTS"]["baseline_effb7"])
m.eval()
x = item["image"].unsqueeze(0).to(ns["DEVICE"])
with torch.no_grad():
    out = m(x)
print("baseline logits", out["classification_logits"].shape, "prob", float(out["fake_probability"]))

print("build full veritas...")
m2 = ns["build_model"](ns["EXPERIMENTS"]["full_veritas"])
m2.eval()
with torch.no_grad():
    out2 = m2(x)
print("full logits", out2["classification_logits"].shape)
print("full seg", tuple(out2["segmentation_logits"].shape))
assert out2["segmentation_logits"].shape[-2:] == x.shape[-2:]
print("MODEL SMOKE OK")
