"""Prepare authenticated-CLI upload folders; this script never publishes or runs a job."""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent


def prepare(destination, only=(), run_mode="smoke", time_budget_hours=None, resume_source=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in sorted((ROOT / "notebooks-ow-v3").glob("OWV3_*.ipynb")):
        suffix = source.stem.removeprefix("OWV3_")
        if only and suffix[:2] not in only:
            continue
        cpu = suffix.startswith(("00_", "11_"))
        notebook = nbformat.read(source, as_version=4)
        slug = "veritas-ow-v3-" + suffix.replace("_", "-")
        folder = destination / slug
        folder.mkdir(exist_ok=True)
        if not cpu:
            cell = next(c for c in notebook.cells if c.cell_type == "code" and c.source.startswith("CONFIG ="))
            assert cell.source.count("'run_mode': 'smoke'") == 1
            cell.source = cell.source.replace("'run_mode': 'smoke'", f"'run_mode': {run_mode!r}", 1)
            if time_budget_hours is not None:
                if not 0 < time_budget_hours <= 11:
                    raise ValueError("Time budget must be positive and at most 11 hours")
                cell.source = cell.source.replace("'time_budget_hours': 11.0", f"'time_budget_hours': {float(time_budget_hours)!r}", 1)
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{source.name}:cell{i}", "exec")
        nbformat.validate(notebook)
        code = folder / source.name
        nbformat.write(notebook, code)
        metadata = dict(id="clarenceshanley/"+slug, title="VERITAS OW v3 "+suffix.replace("_", " ").title(),
                        code_file=source.name, language="python", kernel_type="notebook", is_private=True,
                        enable_gpu=not cpu, enable_internet=not cpu,
                        dataset_sources=[] if suffix.startswith("11_") else ["ahmedgamal/openforensics-wild-dataset"],
                        kernel_sources=[] if suffix.startswith("11_") else ["clarenceshanley/veritas-ow-00-generate-masks/7"],
                        competition_sources=[], model_sources=[])
        if not cpu:
            metadata["machine_shape"] = "NvidiaTeslaT4"
            if resume_source:
                metadata["kernel_sources"].append(resume_source)
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        manifest.append(dict(source=str(source.relative_to(ROOT)), folder=str(folder), id=metadata["id"],
                             sha256=hashlib.sha256(code.read_bytes()).hexdigest(), cpu=cpu,
                             run_mode="audit" if suffix.startswith("00_") else "compare" if cpu else run_mode))
    if not manifest:
        raise ValueError("No notebooks matched; generate them first and use two-digit --only values")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "kaggle_working/ow-v3-publish")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--run-mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--time-budget-hours", type=float)
    parser.add_argument("--resume-source", help="This same experiment's full output, owner/slug/version")
    args = parser.parse_args()
    for item in prepare(args.output, args.only, args.run_mode, args.time_budget_hours, args.resume_source):
        print(item["id"], item["run_mode"], item["sha256"][:12])
    print("Prepared only. Review metadata, then use kaggle kernels push -p FOLDER --no-run to publish.")
