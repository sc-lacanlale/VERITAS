"""Generated Kaggle notebooks must execute without repository-relative imports."""
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import make_ow_v3_notebooks as generator
import make_ow_v2_notebooks as v2
from ow_v3_training import VERITASLossV3

torch.set_num_threads(2)


def test_all_twelve_generated_notebooks_compile_and_validate():
    notebooks = [("audit", generator.build_audit_nb()), ("compare", generator.build_compare_nb())]
    notebooks.extend((variant["file"], generator.build_training_nb(variant)) for variant in generator.variants())
    assert len(notebooks) == 12
    assert len({variant["exp"] for variant in generator.variants()}) == 10
    assert len(list((ROOT / "notebooks-ow-v3").glob("*.ipynb"))) == 12
    for filename, notebook in notebooks:
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{filename}:cell{index}", "exec")
    for path in (ROOT / "notebooks-ow-v3").glob("*.ipynb"):
        nbformat.validate(nbformat.read(path, as_version=4))


def test_embedded_model_loss_and_runtime_execute_without_local_imports():
    env = dict(np=np, torch=torch, nn=torch.nn, F=torch.nn.functional,
               IMAGENET_MEAN=torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
               IMAGENET_STD=torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
               CONFIG={"use_pretrained": False}, DEVICE=torch.device("cpu"))
    # Start from just normal notebook imports/constants, not imported repo classes.
    exec(generator.model_source(), env)
    nb = generator.build_training_nb(generator.variants()[0])
    helper = next(cell.source for cell in nb.cells if cell.cell_type == "code" and "class VERITASLossV3(" in cell.source)
    exec(helper, env)
    runtime = next(cell.source for cell in nb.cells if cell.cell_type == "code" and "def run_notebook():" in cell.source)
    exec(runtime[:runtime.rindex("RUN_RESULT = run_notebook()")], env)
    assert callable(env["VERITASLossV3"]) and callable(env["run_notebook"])
    for index in (0, 3):
        env["EXP_CFG"] = generator.variants()[index]["exp_cfg"]
        model = env["build_model"]().eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 64), torch.ones(1, 1, 64, 64))
        assert outputs["classification_logits"].shape == (1, 1)
        assert all(torch.isfinite(value).all() for value in outputs.values())
        if index == 3:
            assert outputs["segmentation_logits"].shape == (1, 1, 64, 64)
        else:
            assert "segmentation_logits" not in outputs
        del model


def test_ablation_chain_changes_one_architecture_field_at_each_step():
    variants = generator.variants()
    transitions = [(5, 6, "fusion"), (6, 7, "pooling"), (7, 8, "color_srm"),
                   (8, 9, "frequency"), (9, 3, "decoder")]
    for before, after, field in transitions:
        old, new = variants[before]["exp_cfg"], variants[after]["exp_cfg"]
        assert {key for key in old if old[key] != new[key]} == {field}
    assert variants[5]["exp_cfg"]["fusion"] == "late_additive"
    assert variants[3]["exp_cfg"]["fusion"] == "late_nonlinear"


def test_dataset_flips_valid_pixels_rgb_and_teacher_mask_together(tmp_path):
    nb = generator.build_training_nb(generator.variants()[3])
    src = v2.cell_starting(nb, "IMAGENET_MEAN_NP =").source
    src = src[:src.index('if int(CONFIG["image_size"])')]
    env = dict(np=np, torch=torch, cv2=cv2, pd=pd, random=random,
               Dataset=torch.utils.data.Dataset, DataLoader=torch.utils.data.DataLoader,
               CONFIG=dict(seed=42, image_size=600, batch_size=4, num_workers=0))
    exec(src, env)
    rgb = np.zeros((5, 4, 3), dtype=np.uint8)
    rgb[:, 0] = 255
    image, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    cv2.imwrite(str(image), rgb)
    canvas = env["letterbox_rgb"](rgb, 7)
    teacher = canvas[:, :, 0].copy()
    cv2.imwrite(str(mask_path), teacher)
    frame = pd.DataFrame([dict(path=str(image), mask_path=str(mask_path), crop_x0=0, crop_y0=0,
                               crop_x1=4, crop_y1=5, label=0, file_name=image.name,
                               seg_valid=0.0, rel_key="train/real/image.png")])
    dataset = env["GeneratedMaskDataset"](frame, 7, augment=True)
    with patch.object(random, "Random") as rng:
        rng.return_value.random.side_effect = [1.0, 0.0]  # Keep brightness; flip geometry.
        sample = dataset[0]
    recovered = sample["image"][0] * 0.229 + 0.485
    torch.testing.assert_close((recovered > 0.5).float(), sample["mask"][0])
    expected_valid = np.ascontiguousarray(env["letterbox_valid_mask"](rgb, 7)[:, ::-1])
    torch.testing.assert_close(sample["valid_mask"][0], torch.from_numpy(expected_valid))
    assert sample["rel_key"] == "train/real/image.png" and sample["seg_valid"] == 0
    # The saved teacher target remains unchanged; the loss corrects Real targets in memory.
    assert sample["mask"].sum() > 0
    np.testing.assert_array_equal(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), teacher)
    logits = torch.zeros(1, 1, 7, 7, requires_grad=True)
    outputs = dict(classification_logits=torch.zeros(1, 1, requires_grad=True), segmentation_logits=logits)
    losses = VERITASLossV3(segmentation=True)(outputs, sample["label"][None], sample["mask"][None],
                                             sample["valid_mask"][None], sample["seg_valid"][None])
    losses["total"].backward()
    assert (logits.grad[sample["valid_mask"][None].bool()] > 0).all()
    assert logits.grad[~sample["valid_mask"][None].bool()].abs().sum() == 0
    assert losses["segmentation_positive_samples"] == 0


@pytest.mark.parametrize("invalid", [None, "smoke", "protocol", "missing", "duplicate", "size", "version", "nan"])
def test_compare_accepts_only_four_matched_final_core_results(tmp_path, monkeypatch, invalid):
    monkeypatch.chdir(tmp_path)
    rows = []
    for index, variant in enumerate(generator.variants()[:4]):
        rows.append(dict(experiment=variant["exp"], protocol_hash="shared", split="test", run_mode="full",
                         implementation_version="ow_v3", epochs_completed=3, n=10905,
                         accuracy=0.8 + index * 0.01, f1=0.8, precision=0.8, recall=0.8))
    if invalid == "smoke":
        rows[-1].update(run_mode="smoke", split="val", epochs_completed=1, n=128)
    elif invalid == "protocol":
        rows[-1]["protocol_hash"] = "different"
    elif invalid == "missing":
        rows.pop()
    elif invalid == "duplicate":
        rows.append(dict(rows[-1]))
    elif invalid == "size":
        rows[-1]["n"] = 128
    elif invalid == "version":
        rows[-1]["implementation_version"] = "ow_v2"
    elif invalid == "nan":
        rows[-1]["accuracy"] = float("nan")
    pd.DataFrame(rows).to_csv(tmp_path / "experiment_results.csv", index=False)
    source = generator.build_compare_nb().cells[-1].source
    if invalid:
        with pytest.raises(ValueError):
            exec(source, {})
    else:
        exec(source, {})
        result = pd.read_csv(tmp_path / "kaggle_working/ow_v3_compare.csv")
        assert len(result) == 4
        assert result.accuracy_delta_pp_vs_b7.iloc[0] == 0
        assert result.accuracy_delta_pp_vs_b7.iloc[-1] == pytest.approx(3)


def test_mask_audit_executes_only_train_and_validation(tmp_path, monkeypatch):
    image, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    cv2.imwrite(str(image), np.full((3, 3, 3), 128, dtype=np.uint8))
    teacher = np.zeros((600, 600), dtype=np.uint8)
    teacher[:300] = 255
    cv2.imwrite(str(mask_path), teacher)
    mask_rows = [dict(rel_key=f"{split}/{label}", mask_path=str(mask_path), crop_x0=0, crop_y0=0,
                      crop_x1=3, crop_y1=3, positive_frac=0.5)
                 for split in ("train", "val") for label in (0, 1)]
    observed = []

    def list_images(split):
        observed.append(split)
        assert split in ("train", "val"), "Quality decisions must not inspect Test"
        return pd.DataFrame([dict(rel_key=f"{split}/{label}", path=str(image), split=split, label=label)
                             for label in (0, 1)])

    axes = [SimpleNamespace(imshow=lambda *args, **kwargs: None, set_title=lambda *args: None,
                            axis=lambda *args: None) for _ in range(32)]
    figure = SimpleNamespace(tight_layout=lambda: None, savefig=lambda *args, **kwargs: None)
    fake_plot = SimpleNamespace(subplots=lambda *args, **kwargs: (figure, np.array(axes).reshape(8, 4)), show=lambda: None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plot)
    import matplotlib
    monkeypatch.setattr(matplotlib, "pyplot", fake_plot, raising=False)
    audit = generator.build_audit_nb()
    helper = next(cell.source for cell in audit.cells if cell.cell_type == "code" and cell.source.startswith("def letterbox_rgb"))
    env = dict(pd=pd, np=np, cv2=cv2, WORKING=tmp_path, list_images=list_images, MASKS=pd.DataFrame(mask_rows))
    exec(helper, env)
    exec(generator.AUDIT_SOURCE, env)
    assert observed == ["train", "val"]
    summary = pd.read_csv(tmp_path / "mask_quality_train_val.csv")
    assert set(summary.split) == {"train", "val"} and len(summary) == 4
    # The audit notebook omits training indexing/loaders, rather than filtering Test later.
    code = "\n".join(cell.source for cell in audit.cells if cell.cell_type == "code")
    assert "INDEX = {}" not in code and "TRAIN_LOADER =" not in code and "run_notebook()" not in code
