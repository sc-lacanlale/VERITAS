"""Build a separate, reproducible OW v2 study without changing legacy notebooks."""
import ast
import hashlib
from pathlib import Path

import nbformat as nbf

import make_ow_notebooks as legacy

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "notebooks-ow-v2"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Notebook template changed: {old[:90]!r}")
    return source.replace(old, new, 1)


def cell_starting(nb, prefix):
    return next(c for c in nb.cells if c.cell_type == "code" and c.source.startswith(prefix))


def model_source():
    names = {"_srm_kernels", "ASPP", "DeepLabV3PlusHead", "ClassificationHead"}
    shared = [ast.get_source_segment(legacy.MODEL_SRC, n) for n in ast.parse(legacy.MODEL_SRC).body
              if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    src = (ROOT / "veritas-app/backend/inference/architecture_v2.py").read_text(encoding="utf-8")
    nodes = [ast.get_source_segment(src, n) for n in ast.parse(src).body
             if not (isinstance(n, ast.ImportFrom) and n.level)]
    return "\n\n".join(shared + nodes) + '''

def build_model():
    return VERITASV2(multi_stream=EXP_CFG["multi_stream"], segmentation=EXP_CFG["segmentation"],
                     pretrained=CONFIG["use_pretrained"], fusion=EXP_CFG["fusion"]).to(DEVICE)
'''


def build_training_nb(variant, fusion):
    nb = legacy.build_training_nb(variant)
    nb.cells[0].source = replace_once(nb.cells[0].source,
        'Each Save & Run All trains one more epoch, then scores Test.',
        'Each Save & Run All trains one more epoch and scores Validation. Test runs only after epoch three.')
    nb.cells[0].source += '''
## OW v2 protocol

This is a fresh, separately named experiment. Do not attach legacy or smoke
checkpoints to a full run. Attach the same completed OW_00 masks for every variant.
The B7 backbone and RGB classifier share their initialization procedure across
variants. The late-fusion candidate is a new architecture; corrected early fusion
retains the original layer topology. Loss weights are starting settings, not a
claim of optimality: tune them on Validation, with equal tuning budgets.
Full runs select checkpoints on Validation accuracy and evaluate Test only after
all three epochs. Dice is agreement with teacher-generated masks, not independent GT.
'''
    cfg = cell_starting(nb, "CONFIG = {")
    cfg.source = cfg.source[:cfg.source.index("_protocol =")]
    cfg.source = replace_once(cfg.source, '    "lambda_seg": 1.0,', '    "lambda_seg": 0.25,')
    cfg.source = replace_once(cfg.source, '    "test_eval_when": "always",', '    "test_eval_when": "final",')
    cfg.source = replace_once(cfg.source, '    "seed": 42,', '    "seed": 42,\n    "implementation_version": "ow_v2",')
    marker = f'    "legacy_baseline": {variant["legacy_baseline"]},'
    cfg.source = replace_once(cfg.source, marker, marker + f'\n    "implementation_version": "ow_v2",\n    "fusion": "{fusion}",')

    masks = cell_starting(nb, 'EXTS = {')
    masks.source += '''

# Hash stable mask metadata, excluding machine-specific mounted paths.
mask_columns = ["rel_key", "label", "mask_file", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "positive_frac", "teacher"]
MASK_SOURCE_HASH = hashlib.sha256(MASKS[mask_columns].sort_values("rel_key").to_csv(index=False).encode()).hexdigest()
_protocol = {k: CONFIG[k] for k in [
    "implementation_version", "run_mode", "seed", "use_pretrained", "image_size",
    "epochs", "batch_size", "learning_rate", "weight_decay", "classification_threshold",
    "mask_threshold", "lambda_cls", "lambda_seg", "dice_weight", "use_amp", "gradient_accumulation",
]}
_protocol.update(mask_source_hash=MASK_SOURCE_HASH, augmentation="aligned_epoch_flip_v2", selection_metric="val_accuracy")
_protocol["data_index_hash"] = hashlib.sha256(pd.concat([
    INDEX[s][["rel_key", "label"]].sort_values("rel_key").assign(split=s)
    for s in ("train", "val", "test")
]).to_csv(index=False).encode()).hexdigest()
PROTOCOL_HASH = hashlib.sha256(json.dumps(_protocol, sort_keys=True).encode()).hexdigest()
print("PROTOCOL_HASH:", PROTOCOL_HASH)
(WORKING / "logs" / "protocol.json").write_text(json.dumps({"hash": PROTOCOL_HASH, "protocol": _protocol,
    "experiment": CONFIG["experiment"], "exp_cfg": EXP_CFG}, indent=2), encoding="utf-8")
'''
    data = cell_starting(nb, "IMAGENET_MEAN_NP =")
    data.source = replace_once(data.source, '        self.augment = bool(augment)',
        '        self.augment = bool(augment)\n        self.epoch = 0\n        self.seed = int(CONFIG["seed"])')
    data.source = replace_once(data.source, '        row = self.df.iloc[idx]',
        '        row = self.df.iloc[idx]\n        rng = random.Random(self.seed + self.epoch * len(self.df) + int(idx))')
    data.source = replace_once(data.source, '''            rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            x0 = y0 = 0
            x1 = y1 = self.size''', '''            raise FileNotFoundError(f"Unreadable training image: {row.path}")''')
    data.source = replace_once(data.source, '            mask = np.zeros((self.size, self.size), dtype=np.uint8)',
                               '            raise FileNotFoundError(f"Unreadable generated mask: {row.mask_path}")')
    data.source = replace_once(data.source, '''        if self.augment and random.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
''', '')
    data.source = replace_once(data.source, '        canvas = letterbox_rgb(crop, self.size)', '''        canvas = letterbox_rgb(crop, self.size)
        if self.augment and random.random() < 0.5:
            canvas = np.ascontiguousarray(canvas[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])''')
    data.source = data.source.replace('random.random()', 'rng.random()').replace('random.uniform(', 'rng.uniform(')
    data.source = replace_once(data.source, 'def make_loader(df, shuffle, augment):', '''def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df, shuffle, augment):''')
    data.source = replace_once(data.source, '        num_workers=CONFIG["num_workers"],', '''        num_workers=CONFIG["num_workers"],
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(CONFIG["seed"]),''')

    cell_starting(nb, "def _srm_kernels()").source = model_source()
    cell_starting(nb, "class DiceLoss").source = (ROOT / "ow_v2_training.py").read_text(encoding="utf-8")
    checkpoint = cell_starting(nb, 'EXP = CONFIG["experiment"]')
    checkpoint.source = replace_once(checkpoint.source, '            "epoch": epoch,', '''            "protocol_hash": PROTOCOL_HASH,
            "selection_metric": "val_accuracy",
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if DEVICE.type == "cuda" else None,
            "epoch": epoch,''')
    checkpoint.source = replace_once(checkpoint.source, '    ck = _torch_load(path)\n    model.load_state_dict',
        '    ck = _torch_load(path)\n    validate_resume_checkpoint(ck, EXP, EXP_CFG, PROTOCOL_HASH)\n    model.load_state_dict')
    checkpoint.source = replace_once(checkpoint.source,
        '    if not path.exists() or not CONFIG.get("resume", True):', '    if not path.exists():')
    training = cell_starting(nb, "def mask_metrics_from_tensors")
    training.source = replace_once(training.source,
        '    start_epoch, best_metric, resumed = load_if_any(resume_path, model, optimizer)',
        '''    start_epoch, best_metric, resumed = 0, -1.0, {}
    if CONFIG.get("resume", True):
        start_epoch, best_metric, resumed = load_if_any(resume_path, model, optimizer)''')
    training.source = replace_once(training.source, 'criterion = VERITASLoss(', 'criterion = VERITASLossV2(')
    training.source = replace_once(training.source, '        partial = None',
        '        epoch_partial = partial if start_step else None\n        partial = None')
    training.source = replace_once(training.source, '        model.train()', '''        set_seed(CONFIG["seed"] + epoch)
        TRAIN_LOADER.dataset.epoch = epoch
        TRAIN_LOADER.generator.manual_seed(CONFIG["seed"] + epoch)
        if start_step and resumed.get("torch_rng_state") is not None:
            torch.set_rng_state(resumed["torch_rng_state"])
            if DEVICE.type == "cuda" and resumed.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(resumed["cuda_rng_state"], DEVICE)
        total_steps = start_step + len(TRAIN_LOADER)
        model.train()''')
    training.source = replace_once(training.source, 'loss = losses["total"] / accum',
        'loss = losses["total"] / min(accum, total_steps - (step // accum) * accum)')
    training.source = replace_once(training.source, '            if (step + 1) % accum == 0:',
        '            update_boundary = (step + 1) % accum == 0 or step + 1 == total_steps\n            if update_boundary:')
    training.source = replace_once(training.source, '            if (time.time() - last_save) >= mid_s:',
        '            if update_boundary and (time.time() - last_save) >= mid_s:')
    training.source = replace_once(training.source, '            if elapsed() > budget_s:',
        '            if update_boundary and elapsed() > budget_s:')
    training.source = replace_once(training.source, 'running += float(loss.item()) * accum * images.size(0)',
        'running += float(losses["total"].detach()) * images.size(0)')
    training.source = replace_once(training.source, '        running, seen, t0, last_save =',
        '        running_cls = running_seg_bce = running_seg_dice = 0.0\n        running, seen, t0, last_save =')
    training.source = replace_once(training.source, '        it = tqdm(TRAIN_LOADER,', '''        if epoch_partial:
            running, running_cls, running_seg_bce, running_seg_dice = epoch_partial["loss_sums"]
            seen = int(epoch_partial["seen"])
            t0 -= float(epoch_partial["elapsed_seconds"])
        it = tqdm(TRAIN_LOADER,''')
    training.source = replace_once(training.source, '            seen += images.size(0)', '''            running_cls += float(losses["classification"]) * images.size(0)
            running_seg_bce += float(losses["segmentation_bce"]) * images.size(0)
            running_seg_dice += float(losses["segmentation_dice"]) * images.size(0)
            seen += images.size(0)''')
    partial = 'partial={"epoch": epoch, "step": step + 1}'
    assert training.source.count(partial) == 2
    training.source = training.source.replace(partial, '''partial={"epoch": epoch, "step": step + 1,
                    "loss_sums": [running, running_cls, running_seg_bce, running_seg_dice],
                    "seen": seen, "elapsed_seconds": time.time() - t0}''')
    training.source = replace_once(training.source, '        history.append(hist)', '''        hist.update(train_classification=running_cls / max(seen, 1),
                    train_segmentation_bce=running_seg_bce / max(seen, 1),
                    train_segmentation_dice=running_seg_dice / max(seen, 1), lambda_seg=CONFIG["lambda_seg"])
        history.append(hist)''')
    training.source = replace_once(training.source, 'score = float(val_metrics["f1"])', 'score = float(val_metrics["accuracy"])')
    training.source = training.source.replace('new best val f1', 'new best val accuracy').replace('best val f1', 'best val accuracy')
    training.source = training.source.replace('"best_val_f1": best_metric', '"best_val_accuracy": best_metric')
    training.source = replace_once(training.source, '"epochs_completed": EPOCHS_DONE, **{k: v for k, v in metrics.items()',
        '"epochs_completed": EPOCHS_DONE, "split": "test", "protocol_hash": PROTOCOL_HASH, "implementation_version": "ow_v2", **{k: v for k, v in metrics.items()')
    training.source += '''

if HISTORY:
    pd.DataFrame(HISTORY).to_csv(WORKING / "logs" / "validation_history.csv", index=False)
print("RUN_STATE", json.dumps({"experiment": EXP, "epochs_completed": EPOCHS_DONE,
    "epochs_total": CONFIG["epochs"], "protocol_hash": PROTOCOL_HASH, "exp_cfg": EXP_CFG}))
if CONFIG["run_mode"] == "smoke":
    assert EPOCHS_DONE == 1 and len(INDEX["test"]) == metrics["n"] == 128
    assert len(MASKS) == 190335
    assert all(math.isfinite(metrics[k]) for k in ("accuracy", "f1", "precision", "recall"))
    assert (WORKING / "checkpoints" / f"{EXP}_last.pt").is_file()
    print("TRAIN_SMOKE_VALIDATION_PASSED", EXP)
'''
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace('All four `OW_01`–`OW_04`', 'All five `OWV2_01`–`OWV2_05`')
            cell.source = cell.source.replace('`OW_05_compare.ipynb`', '`OWV2_06_compare.ipynb`')
    return nb


def variants():
    result = []
    for i, v in enumerate(legacy.VARIANTS, start=1):
        v = dict(v)
        v.update(exp=v["exp"] + "_v2", file=v["file"].replace("OW_", "OWV2_"), n=f"{i} of 5")
        result.append((v, "early" if v["multi_stream"] else "rgb"))
    v = dict(result[-1][0])
    v.update(exp="late_fusion_veritas_ow_v2", file="OWV2_05_late_fusion.ipynb", title="Late-fusion VERITAS candidate",
             who="Late-fusion experiment", n="5 of 5",
             blurb="RGB B7 + separate SRM/FFT encoders, pooled late fusion, and an RGB DeepLabV3+ decoder.")
    result.append((v, "late"))
    return result


COMPARE_SOURCE = '''import json
from pathlib import Path
import pandas as pd

EXPECTED = ["baseline_effb7_ow_v2", "multistream_no_seg_ow_v2", "seg_no_multistream_ow_v2", "full_veritas_ow_v2", "late_fusion_veritas_ow_v2"]
root = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()
frames = []
for path in root.rglob("experiment_results.csv"):
    df = pd.read_csv(path)
    if "experiment" not in df:
        continue
    df = df[df.experiment.isin(EXPECTED)]
    if not df.empty:
        frames.append(df.assign(source=str(path)))
if not frames:
    raise FileNotFoundError("Attach completed v2 full-training outputs")
results = pd.concat(frames, ignore_index=True)
required = {"protocol_hash", "split", "implementation_version", "epochs_completed", "n", "accuracy", "f1", "precision", "recall"}
if not required <= set(results.columns):
    raise ValueError("Incomplete result provenance")
if results.experiment.duplicated().any():
    raise ValueError("Attach exactly one final result per experiment")
if set(results.experiment) != set(EXPECTED):
    raise ValueError("All five final experiments, including B7 baseline, are required")
if not ((results.split == "test") & (results.implementation_version == "ow_v2") &
        (results.epochs_completed == 3) & (results.n == 10905)).all():
    raise ValueError("Only final full Test results are comparable; exclude smoke and partial runs")
if results.protocol_hash.nunique() != 1 or results.protocol_hash.isna().any():
    raise ValueError("Protocol/seed/mask sources differ; rerun a matched comparison")
scores = results[["accuracy", "f1", "precision", "recall"]]
if not ((scores >= 0) & (scores <= 1)).all().all():
    raise ValueError("Nonfinite or invalid classification metrics")
baseline = float(results.loc[results.experiment == EXPECTED[0], "accuracy"].iloc[0])
results["accuracy_delta_pp_vs_b7"] = 100 * (results.accuracy - baseline)
out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working"
out.mkdir(parents=True, exist_ok=True)
results.to_csv(out / "ow_v2_compare.csv", index=False)
print(results[["experiment", "accuracy", "accuracy_delta_pp_vs_b7", "f1", "precision", "recall"]].sort_values("accuracy", ascending=False).to_string(index=False))
print("Positive delta means higher accuracy in this run; repeat matched seeds before a superiority claim.")
print("Dice measures teacher-mask agreement, not independently annotated localization.")
'''


def write_nb(nb, name):
    for i, cell in enumerate(nb.cells):
        cell.id = hashlib.sha1(f"{name}:{i}".encode()).hexdigest()[:8]
        if cell.cell_type == "code":
            compile(cell.source, f"{name}:cell{i}", "exec")
    nbf.validate(nb)
    nbf.write(nb, DEST / name)


def main():
    DEST.mkdir(exist_ok=True)
    for v, fusion in variants():
        write_nb(build_training_nb(v, fusion), v["file"])
    compare = legacy._nb([legacy.md("# OW v2 matched final-Test comparison\nAttach all five completed v2 outputs. No training runs here."), legacy.cd(COMPARE_SOURCE)])
    write_nb(compare, "OWV2_06_compare.ipynb")
    print("Wrote", DEST)


if __name__ == "__main__":
    main()
