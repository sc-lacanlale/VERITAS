"""Execute notebook code cells up to (and including) unit tests. Skip experiment training."""
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import matplotlib
matplotlib.use("Agg")
import traceback
from pathlib import Path

import nbformat

nb = nbformat.read("VERITAS_Kaggle.ipynb", as_version=4)
ns = {"__name__": "__main__"}
stop_after_substring = "TEST_RESULTS = run_unit_tests()"
log = []

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source
    lines = []
    for line in src.splitlines():
        if line.strip().startswith("%"):
            continue
        lines.append(line)
    src = "\n".join(lines)
    header = f"\n===== CELL {i} =====\n"
    print(header)
    log.append(header)
    try:
        exec(compile(src, f"nb_cell_{i}", "exec"), ns, ns)
        log.append("OK\n")
        print("OK")
    except Exception:
        tb = traceback.format_exc()
        log.append(tb + "\n")
        print(tb)
        Path("nb_exec_log.txt").write_text("".join(log), encoding="utf-8")
        raise
    if stop_after_substring in cell.source:
        log.append("STOPPED AFTER UNIT TESTS\n")
        break

Path("nb_exec_log.txt").write_text("".join(log), encoding="utf-8")
print("DONE through unit tests")
