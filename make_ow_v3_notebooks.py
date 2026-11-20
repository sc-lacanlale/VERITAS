"""Build self-contained OW v3 notebooks and explicit architectural ablations."""
import ast
import hashlib
import json
from pathlib import Path

import nbformat as nbf

import make_ow_notebooks as legacy
import make_ow_v2_notebooks as v2

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "notebooks-ow-v3"


def variants():
    base = dict(multi_stream=True, segmentation=True, fusion="late_nonlinear",
                color_srm=True, frequency="local", pooling="avg_max", decoder="separable")
    specifications = [
        ("01_baseline", "baseline_effb7_ow_v3", dict(multi_stream=False, segmentation=False, fusion="rgb", color_srm=False, frequency="global", pooling="avg", decoder="dense")),
        ("02_multistream", "multistream_no_seg_ow_v3", dict(segmentation=False)),
        ("03_segmentation", "seg_no_multistream_ow_v3", dict(multi_stream=False, fusion="rgb", color_srm=False, frequency="global", pooling="avg")),
        ("04_full", "full_veritas_ow_v3", {}),
        ("05_early_reference", "early_reference_ow_v3", dict(fusion="early", color_srm=False, frequency="global", pooling="avg", decoder="dense")),
        ("06_additive_reference", "additive_reference_ow_v3", dict(fusion="late_additive", color_srm=False, frequency="global", pooling="avg", decoder="dense")),
        ("07_nonlinear_ablation", "nonlinear_ablation_ow_v3", dict(color_srm=False, frequency="global", pooling="avg", decoder="dense")),
        ("08_pooling_ablation", "pooling_ablation_ow_v3", dict(color_srm=False, frequency="global", decoder="dense")),
        ("09_color_ablation", "color_ablation_ow_v3", dict(frequency="global", decoder="dense")),
        ("10_frequency_ablation", "frequency_ablation_ow_v3", dict(decoder="dense")),
    ]
    return [dict(file="OWV3_"+name+".ipynb", exp=exp,
                 exp_cfg={**base, **overrides, "implementation_version": "ow_v3"})
            for name, exp, overrides in specifications]


def model_source():
    prior = v2.model_source()
    blocks = [ast.get_source_segment(prior, n) for n in ast.parse(prior).body
              if not (isinstance(n, ast.FunctionDef) and n.name == "build_model")]
    src = (ROOT / "veritas-app/backend/inference/architecture_v3.py").read_text(encoding="utf-8")
    blocks += [ast.get_source_segment(src, n) for n in ast.parse(src).body
               if not (isinstance(n, ast.ImportFrom) and n.level)]
    return "\n\n".join(blocks) + '''

def build_model():
    args = {k: v for k, v in EXP_CFG.items() if k != "implementation_version"}
    return VERITASV3(pretrained=CONFIG["use_pretrained"], **args).to(DEVICE)
'''


def source_hash():
    files = [Path(__file__), ROOT / "ow_v3_runtime.py", ROOT / "ow_v3_training.py",
             ROOT / "veritas-app/backend/inference/architecture_v3.py",
             ROOT / "veritas-app/backend/inference/architecture_v2.py",
             ROOT / "make_ow_v2_notebooks.py", ROOT / "make_ow_notebooks.py"]
    return hashlib.sha256("\n".join(p.read_text(encoding="utf-8") for p in files).encode()).hexdigest()


def template(variant):
    cfg = variant["exp_cfg"]
    v = dict(legacy.VARIANTS[0], file=variant["file"], exp=variant["exp"], title=variant["exp"],
             multi_stream=cfg["multi_stream"], segmentation=cfg["segmentation"], legacy_baseline=False)
    return v2.build_training_nb(v, "early" if cfg["multi_stream"] else "rgb")


PROTOCOL_SOURCE = '''
# Hash source data/targets and common training settings; architecture is separate.
assert len(MASKS) == 190335, "Attach the complete OW00 v7 masks"
quality_path = CONFIG.get("seg_quality_csv")
quality = None
if quality_path:
    quality = pd.read_csv(quality_path)
    if not {"rel_key", "seg_valid"} <= set(quality) or quality.rel_key.duplicated().any():
        raise ValueError("Quality CSV requires unique rel_key and binary seg_valid")
    if not quality.seg_valid.isin([0, 1]).all():
        raise ValueError("seg_valid must be binary, based on an external quality assessment")
    if quality.rel_key.str.startswith("test/").any():
        raise ValueError("Quality decisions must use Train/Validation, not Test")
for split, frame in INDEX.items():
    frame["seg_valid"] = 1.0
    if quality is not None and split != "test":
        flags = quality.set_index("rel_key").seg_valid
        frame["seg_valid"] = frame.rel_key.map(flags).fillna(1).astype(float)
mask_cols = ["rel_key", "label", "mask_file", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "positive_frac", "teacher"]
mask_hash = hashlib.sha256(MASKS[mask_cols].sort_values("rel_key").to_csv(index=False).encode()).hexdigest()
index_hash = hashlib.sha256(pd.concat([INDEX[s][["rel_key", "label", "seg_valid"]].sort_values("rel_key").assign(split=s)
    for s in ("train", "val", "test")]).to_csv(index=False).encode()).hexdigest()
keys = ["run_mode", "seed", "epochs", "batch_size", "image_size", "learning_rate", "weight_decay", "use_pretrained",
        "classification_threshold", "mask_threshold", "lambda_cls", "lambda_seg", "dice_weight", "use_amp",
        "gradient_accumulation", "backbone_lr_multiplier", "warmup_fraction", "freeze_backbone_bn", "source_code_hash"]
protocol = {k: CONFIG[k] for k in keys}
protocol.update(mask_source_hash=mask_hash, data_index_hash=index_hash, selection_metric="val_accuracy",
                supervision_policy="real_zero_fake_nonempty_v3", augmentation="aligned_epoch_flip_v3")
PROTOCOL_HASH = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
(WORKING / "logs/protocol.json").write_text(json.dumps(dict(hash=PROTOCOL_HASH, protocol=protocol,
    experiment=CONFIG["experiment"], exp_cfg=EXP_CFG), indent=2), encoding="utf-8")
print("PROTOCOL_HASH", PROTOCOL_HASH)
'''


def dataset_source(nb):
    src = v2.cell_starting(nb, "IMAGENET_MEAN_NP =").source
    marker = "class GeneratedMaskDataset(Dataset):"
    helper = '''def letterbox_valid_mask(crop, size):
    h, w = crop.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = max(round(w * scale), 1), max(round(h * scale), 1)
    left, top = (size - nw) // 2, (size - nh) // 2
    result = np.zeros((size, size), dtype=np.float32)
    result[top:top+nh, left:left+nw] = 1
    return result


'''
    src = v2.replace_once(src, marker, helper + marker)
    src = v2.replace_once(src, '        canvas = letterbox_rgb(crop, self.size)',
        '        canvas = letterbox_rgb(crop, self.size)\n        valid_canvas = letterbox_valid_mask(crop, self.size)')
    src = v2.replace_once(src, '            mask = np.ascontiguousarray(mask[:, ::-1])',
        '            mask = np.ascontiguousarray(mask[:, ::-1])\n            valid_canvas = np.ascontiguousarray(valid_canvas[:, ::-1])')
    src = v2.replace_once(src, '            "mask": torch.from_numpy(target[None].copy()),',
        '''            "mask": torch.from_numpy(target[None].copy()),
            "valid_mask": torch.from_numpy(valid_canvas[None].copy()),
            "seg_valid": torch.tensor(float(row.seg_valid)),
            "rel_key": row.rel_key,''')
    return src


def build_training_nb(variant):
    nb = template(variant)
    cfg = variant["exp_cfg"]
    nb.cells[0].source = f'''# OW v3 {variant["exp"]}

Fresh ImageNet B7 experiment on Wild with completed OW00 v7 pseudo masks.
Architecture: `{json.dumps(cfg, sort_keys=True)}`.

Attach `ahmedgamal/openforensics-wild-dataset` and the completed OW00 version7 output.
For continuation, attach only this experiment's latest full output. Smoke/full and
source/protocol mismatches are rejected. Do not load legacy teacher or v2 weights.
Default smoke trains one epoch and checks Validation only. Full runs train three
epochs, one per saved run, select by Validation accuracy, and evaluate Test once
training is complete. Binary masks are teacher targets, not independent GT.
See the GitHub notebooks-ow-v3 README for changes, ablations and limitations.
'''
    config = dict(run_mode="smoke", seed=42, implementation_version="ow_v3", use_pretrained=True,
                  image_size=600, batch_size=4, num_workers=2, epochs=3, learning_rate=1e-4,
                  weight_decay=1e-4, classification_threshold=0.5, mask_threshold=0.5,
                  lambda_cls=1.0, lambda_seg=0.25, dice_weight=1.0, use_amp=True,
                  gradient_accumulation=1, skip_training=False, resume=True, resume_from_input=True,
                  time_budget_hours=11.0, epochs_per_run=1, mid_epoch_checkpoint_minutes=30,
                  experiment=variant["exp"], max_per_split=None, backbone_lr_multiplier=0.1,
                  warmup_fraction=0.05, freeze_backbone_bn=True, seg_quality_csv=None,
                  benchmark_inference=True, source_code_hash=source_hash())
    source = "CONFIG = " + repr(config) + "\nEXP_CFG = " + repr(cfg) + '''
if os.name == "nt":
    CONFIG["num_workers"] = 0
if CONFIG["run_mode"] not in ("smoke", "full"):
    raise ValueError("run_mode must be smoke or full")
if CONFIG["run_mode"] == "smoke":
    CONFIG.update(max_per_split=64, epochs=1)
if CONFIG["run_mode"] == "full" and not torch.cuda.is_available():
    raise RuntimeError("Full training requires a Kaggle GPU")
if int(CONFIG["gradient_accumulation"]) < 1:
    raise ValueError("gradient_accumulation must be positive")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("EXPERIMENT", CONFIG["experiment"], EXP_CFG, CONFIG["run_mode"])
'''
    v2.cell_starting(nb, "CONFIG = {").source = source
    masks = v2.cell_starting(nb, "EXTS = {")
    masks.source = masks.source[:masks.source.index("# Hash stable mask metadata")] + PROTOCOL_SOURCE
    data = v2.cell_starting(nb, "IMAGENET_MEAN_NP =")
    data.source = dataset_source(nb)
    model = next(c for c in nb.cells if c.cell_type == "code" and "class VERITASV2(" in c.source)
    model.source = model_source() + f'\nMODEL_SOURCE_HASH = "{hashlib.sha256(model_source().encode()).hexdigest()}"\n'
    seed = v2.cell_starting(nb, "def set_seed(")
    seed.source = seed.source[:seed.source.index('print("legacy_baseline"')]
    loss = next(c for c in nb.cells if c.cell_type == "code" and "class VERITASLossV2(" in c.source)
    loss.source = (ROOT / "ow_v3_training.py").read_text(encoding="utf-8")
    checkpoint = v2.cell_starting(nb, 'EXP = CONFIG["experiment"]')
    checkpoint.source = "# Checkpoint and training functions are defined together below.\n"
    v2.cell_starting(nb, "def mask_metrics_from_tensors").source = (ROOT / "ow_v3_runtime.py").read_text(encoding="utf-8") + "\nRUN_RESULT = run_notebook()\n"
    nb.cells[-1].source = '''## Research interpretation

Accuracy/F1 use Wild folder labels. Fake Dice/IoU measure teacher agreement on
available targets; Real positive-mask area is reported separately. All variants
share the training protocol; architecture flags distinguish checkpoints. Keep
Test out of tuning. A higher score must be demonstrated, not assumed.
'''
    return nb


AUDIT_SOURCE = '''
import matplotlib.pyplot as plt

summary, selected = [], []
for split in ("train", "val"):
    frame = list_images(split).merge(MASKS[["rel_key", "mask_path", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "positive_frac"]], on="rel_key", how="left")
    if frame.mask_path.isna().any():
        raise RuntimeError(f"Missing masks in {split}")
    frame["positive_frac"] = pd.to_numeric(frame.positive_frac, errors="raise")
    for label, group in frame.groupby("label"):
        summary.append(dict(split=split, label=int(label), n=len(group),
            teacher_positive_mean=float(group.positive_frac.mean()),
            teacher_empty_rate=float((group.positive_frac == 0).mean()),
            teacher_positive_std=float(group.positive_frac.std())))
        selected.append(group.sample(n=min(8, len(group)), random_state=42))
audit = pd.DataFrame(summary)
audit.to_csv(WORKING / "mask_quality_train_val.csv", index=False)
print(audit.to_string(index=False))
print("Real training targets become ZERO; nonempty fake masks remain uncertain teacher targets.")
print("Area is not confidence. No independent localization accuracy is established.")
samples = pd.concat(selected, ignore_index=True)
fig, axes = plt.subplots(8, 4, figsize=(16, 28))
for ax, row in zip(axes.flat, samples.itertuples()):
    bgr = cv2.imread(row.path)
    mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None or mask.shape != (600, 600):
        raise RuntimeError("Unreadable image or invalid saved mask")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    crop = rgb[int(float(row.crop_y0)):int(float(row.crop_y1)), int(float(row.crop_x0)):int(float(row.crop_x1))]
    canvas = letterbox_rgb(crop, 600)
    ax.imshow(canvas)
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.35, vmin=0, vmax=255)
    ax.set_title(f"{row.split} {'Real' if row.label == 0 else 'Fake'} | teacher {row.positive_frac:.3f}")
    ax.axis("off")
for ax in axes.flat[len(samples):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(WORKING / "mask_audit_train_val.png", dpi=100)
plt.show()
print("MASK_AUDIT_COMPLETED_TRAIN_VAL_ONLY", len(MASKS))
'''


def build_audit_nb():
    nb = template(variants()[0])
    masks = v2.cell_starting(nb, "EXTS = {").source
    masks = masks[:masks.index("MASKS = load_mask_table()")] + "MASKS = load_mask_table()\nassert len(MASKS) == 190335\n"
    data = v2.cell_starting(nb, "IMAGENET_MEAN_NP =").source
    function = next(n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name == "letterbox_rgb")
    code = [c for c in nb.cells if c.cell_type == "code"]
    return legacy._nb([legacy.md("# OW v3 mask quality audit\nTrain/Validation metadata and overlays only. No training or Test quality analysis. Attach Wild and OW00 v7. CPU is sufficient."),
                       code[0], code[1], code[2], legacy.cd(masks),
                       legacy.cd(ast.get_source_segment(data, function)), legacy.cd(AUDIT_SOURCE)])


def build_compare_nb():
    src = v2.COMPARE_SOURCE.replace('EXPECTED = '+repr([v[0]["exp"] for v in v2.variants()]), 'EXPECTED = '+repr([v["exp"] for v in variants()[:4]]))
    # The v2 literal uses double quotes; replace its complete EXPECTED line.
    lines = src.splitlines()
    lines = ['EXPECTED = '+repr([v["exp"] for v in variants()[:4]]) if line.startswith("EXPECTED =") else line for line in lines]
    src = "\n".join(lines).replace('"ow_v2"', '"ow_v3"').replace("v2", "v3").replace("All five", "All four").replace("all five", "all four")
    src = src.replace('Dice measures teacher-mask agreement, not independently annotated localization.',
                      'Fake Dice measures teacher agreement; Real positive-mask area is separate.')
    return legacy._nb([legacy.md("# OW v3 matched final comparison\nAttach the four completed core experiment outputs. Ablations use the same protocol but are compared separately on Validation during development."), legacy.cd(src)])


def write_nb(nb, name):
    for i, cell in enumerate(nb.cells):
        cell.id = hashlib.sha1(f"{name}:{i}".encode()).hexdigest()[:8]
        if cell.cell_type == "code":
            compile(cell.source, f"{name}:cell{i}", "exec")
    nbf.validate(nb)
    nbf.write(nb, DEST / name)


def main():
    DEST.mkdir(exist_ok=True)
    write_nb(build_audit_nb(), "OWV3_00_mask_audit.ipynb")
    for variant in variants():
        write_nb(build_training_nb(variant), variant["file"])
    write_nb(build_compare_nb(), "OWV3_11_compare.ipynb")
    print("Wrote", DEST, "source", source_hash())


if __name__ == "__main__":
    main()
