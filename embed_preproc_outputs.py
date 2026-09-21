"""Rebuild notebook, generate preprocessing figures, embed them as cell outputs."""
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import base64
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import nbformat
from nbformat.v4 import new_output

import assemble_notebook  # rebuilds VERITAS_Kaggle.ipynb  # noqa: F401

nb = nbformat.read("VERITAS_Kaggle.ipynb", as_version=4)
ns = {"__name__": "__main__"}
stop = "PREPROC_FIGS = show_preprocessing_gallery"

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = "\n".join(ln for ln in cell.source.splitlines() if not ln.strip().startswith("%"))
    exec(compile(src, f"nb_cell_{i}", "exec"), ns, ns)
    if stop in cell.source:
        viz_idx = i
        break
else:
    raise RuntimeError("preprocessing gallery cell not found")

debug_dir = ns["PATHS"]["working"] / "debug"
paths = list(sorted(debug_dir.glob("preproc_image_*.png"))) + list(sorted(debug_dir.glob("preproc_face_*.png")))
print("figures", len(paths))
for p in paths:
    print(" ", p.name, p.stat().st_size)

outputs = []
outputs.append(
    new_output(
        "stream",
        name="stdout",
        text="Preprocessing before/after samples (embedded from a local smoke run on the compiled OpenForensics files).\n"
        + "\n".join(p.name for p in paths)
        + "\n",
    )
)
for p in paths:
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    outputs.append(
        new_output(
            "display_data",
            data={
                "image/png": b64,
                "text/plain": [p.name],
            },
            metadata={"veritas_figure": p.name},
        )
    )

nb.cells[viz_idx]["outputs"] = outputs
nb.cells[viz_idx]["execution_count"] = 1
nbformat.write(nb, "VERITAS_Kaggle.ipynb")
print("embedded", len(paths), "figures into cell", viz_idx)
