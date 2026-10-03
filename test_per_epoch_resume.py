"""Simulates several Kaggle sessions of notebooks/VERITAS_04_full.ipynb training with a tiny model.

Checks: epoch count advances by one per run and stops at `epochs`, the most advanced attached
checkpoint is picked, and a run cut mid-epoch resumes to the same weights as an uninterrupted run.
"""
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import nbformat
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

nb = nbformat.read("notebooks/VERITAS_04_full.ipynb", as_version=4)
cells = [c.source for c in nb.cells if c.cell_type == "code"]
train_src = next(s for s in cells if "def train_experiment" in s)
loader_src = next(s for s in cells if "class EpochSeededSampler" in s)
config_src = next(s for s in cells if '"epochs_per_run"' in s)
assert '"epochs_per_run": 1,' in config_src and '"test_eval_when": "final",' in config_src
sampler_src = loader_src.split("def make_loader")[0]

ROOT = Path(tempfile.mkdtemp())
INPUT = ROOT / "input"


class TinyData(Dataset):
    def __init__(self, n, delay=0.0):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, 8, 8, generator=g)
        self.y = (torch.rand(n, generator=g) > 0.5).float()
        self.delay = delay

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.delay:
            time.sleep(self.delay)
        return {"image": self.x[i], "label": self.y[i], "mask": torch.zeros(1, 8, 8)}


class TinyModel(nn.Module):
    segmentation = False

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3 * 8 * 8, 1)

    def forward(self, x):
        return {"classification_logits": self.fc(x.flatten(1)).squeeze(1)}


class TinyLoss(nn.Module):
    def __init__(self, **kw):
        super().__init__()

    def forward(self, out, y, m=None):
        l = nn.functional.binary_cross_entropy_with_logits(out["classification_logits"], y)
        return {"total": l, "classification": l.detach(), "segmentation": torch.tensor(0.0)}


def fake_eval(model, loader, criterion, split):
    return {"loss": 0.0, "f1": float(model.fc.weight.abs().sum()), "accuracy": 0.5, "miou": None}, []


def new_session(epochs_per_run=1, mid_min=None, budget_h=11.0, delay=0.0):
    """Fresh namespace + empty working dir, like a new Kaggle commit."""
    work = ROOT / "working"
    shutil.rmtree(work, ignore_errors=True)
    (work / "logs").mkdir(parents=True)
    ns = {
        "__name__": "__main__",
        "torch": torch, "nn": nn, "np": np, "math": math, "os": os, "time": time, "shutil": shutil,
        "Path": Path, "json": __import__("json"), "gc": __import__("gc"), "warnings": __import__("warnings"),
        "tqdm": __import__("tqdm").tqdm, "traceback": __import__("traceback"),
        "DEVICE": torch.device("cpu"),
        "PATHS": {"working": work},
        "CONFIG": {
            "epochs": 3, "epochs_per_run": epochs_per_run, "mid_epoch_checkpoint_minutes": mid_min,
            "test_eval_when": "final", "batch_size": 4, "seed": 42, "learning_rate": 1e-2,
            "weight_decay": 1e-4, "use_amp": False, "gradient_accumulation": 1, "resume": True,
            "resume_from_input": True, "skip_training": False, "time_budget_hours": budget_h,
            "mem_log_every_steps": 0, "lambda_cls": 1.0, "lambda_seg": 1.0, "dice_weight": 1.0,
        },
        "NOTEBOOK_START_TIME": time.time(),
        "build_model": lambda cfg: TinyModel(),
        "unwrap_model": lambda m: m,
        "VERITASLoss": TinyLoss,
        "evaluate_loader": fake_eval,
    }
    exec(sampler_src, ns)
    exec(train_src.replace('"/kaggle/input"', repr(str(INPUT))), ns)
    ns["evaluate_loader"] = fake_eval
    ds = TinyData(40, delay)
    sampler = ns["EpochSeededSampler"](len(ds), 42)
    loader = DataLoader(ds, batch_size=4, sampler=sampler, drop_last=True, num_workers=0)
    val = DataLoader(TinyData(8), batch_size=4)
    return ns, loader, val, work


def publish_output(work, name):
    """Kaggle: attach the finished version's output as an input for the next run."""
    dst = INPUT / name / "checkpoints"
    dst.mkdir(parents=True, exist_ok=True)
    for p in (work / "checkpoints").glob("*.pt"):
        shutil.copy2(p, dst / p.name)


cfg = {"multi_stream": True, "segmentation": False}

# 1) One epoch per run, 4 runs, older outputs stay attached as well.
for run in range(1, 5):
    ns, tl, vl, work = new_session()
    model, info = ns["train_experiment"]("full_veritas", cfg, tl, vl)
    expected = min(run, 3)
    print(f"RUN {run}: epochs_completed={info['epochs_completed']} this_run={info['epochs_trained_this_run']}")
    assert info["epochs_completed"] == expected, info
    assert info["epochs_trained_this_run"] == (1 if run <= 3 else 0)
    assert len(info["history"]) == expected
    assert [h["epoch"] for h in info["history"]] == list(range(1, expected + 1))
    publish_output(work, f"run{run}")
uninterrupted = {k: v.clone() for k, v in torch.load(work / "checkpoints" / "full_veritas_last.pt",
                                                     weights_only=False)["model_state_dict"].items()}

# 2) Mid-epoch stop: tiny budget cuts epochs; runs continue from the saved step until 3/3.
shutil.rmtree(INPUT)
runs = 0
while True:
    runs += 1
    ns, tl, vl, work = new_session(epochs_per_run=1, mid_min=1e-6, budget_h=0.6 / 3600, delay=0.01)
    model, info = ns["train_experiment"]("full_veritas", cfg, tl, vl)
    print(f"MID RUN {runs}: completed={info['epochs_completed']} partial={info['partial_epoch']}")
    publish_output(work, f"mid{runs}")
    if info["epochs_completed"] == 3:
        break
    assert runs < 40, "no progress"
assert runs > 3, "budget never cut an epoch; test did not exercise mid-epoch resume"
resumed = torch.load(work / "checkpoints" / "full_veritas_last.pt", weights_only=False)["model_state_dict"]
diff = max(float((resumed[k] - uninterrupted[k]).abs().max()) for k in resumed)
print("max weight diff interrupted vs uninterrupted:", diff)
assert diff < 1e-6

shutil.rmtree(ROOT)
print("PER-EPOCH RESUME TESTS OK")
