"""Assemble VERITAS_Kaggle.ipynb from the three cell-definition modules."""
from pathlib import Path

import nbformat as nbf

import build_notebook  # noqa: F401  — defines cells, md, code and part-1 cells
import build_notebook_part2  # noqa: F401
import build_notebook_part3  # noqa: F401

from build_notebook import cells

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    "accelerator": "GPU",
}
nb["cells"] = cells

out = Path("VERITAS_Kaggle.ipynb")
nbf.write(nb, out)
print("Wrote", out.resolve(), "cells=", len(cells))
print("markdown", sum(1 for c in cells if c.cell_type == "markdown"))
print("code", sum(1 for c in cells if c.cell_type == "code"))
