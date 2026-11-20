"""CPU integration checks for checkpointing, seeded continuation, and Test isolation."""
import importlib
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ow_v3_training import (VERITASLossV3, build_optimizer, build_scheduler,
                            freeze_backbone_bn, validate_resume_checkpoint)

torch.set_num_threads(2)


class EpochSeededSampler(torch.utils.data.Sampler):
    def __init__(self, n, seed):
        self.n, self.seed, self.epoch, self.start_index = n, seed, 0, 0

    def set_epoch(self, epoch, start_index=0):
        self.epoch, self.start_index = epoch, start_index

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return iter(torch.randperm(self.n, generator=generator).tolist()[self.start_index:])

    def __len__(self):
        return max(self.n - self.start_index, 0)


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, size=8, split="train", seed=42):
        self.size, self.split, self.seed, self.epoch = size, split, seed, 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rng = random.Random(self.seed + self.epoch * self.size + index)
        value = (index + 1) / self.size
        if self.split == "train":
            value *= rng.uniform(0.8, 1.2)
        label = index % 2
        target = torch.zeros(1, 2, 2)
        target[:, 0, 0] = 1  # Teacher can even mark Real pixels; the v3 loss corrects them.
        valid = torch.ones_like(target)
        if label:
            valid[:, :, 1] = 0
        return dict(image=torch.full((1, 2, 2), value), label=torch.tensor([float(label)]),
                    mask=target, valid_mask=valid, seg_valid=torch.tensor(float(index != 3)),
                    rel_key=f"{self.split}/{index}", file_name=str(index))


class TinyModel(nn.Module):
    segmentation = True

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 4), nn.Tanh())
        self.dropout = nn.Dropout(0.35)
        self.head = nn.Linear(4, 1)
        self.seg_head = nn.Linear(4, 1)

    def forward(self, x, valid_mask=None):
        assert valid_mask is not None and valid_mask.shape == (len(x), 1, 2, 2)
        hidden = self.encoder(x.mean((-2, -1)))
        logits = self.head(self.dropout(hidden))
        seg = self.seg_head(hidden).view(-1, 1, 1, 1).expand(-1, 1, 2, 2)
        return dict(classification_logits=logits, fake_probability=logits.sigmoid(),
                    segmentation_logits=seg, segmentation_prob=seg.sigmoid())


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    module = importlib.import_module("ow_v3_runtime")
    config = dict(experiment="toy_ow_v3", seed=42, epochs=1, epochs_per_run=1, batch_size=2, image_size=2,
                  learning_rate=0.02, weight_decay=0.001, backbone_lr_multiplier=0.1,
                  warmup_fraction=0.5, lambda_cls=1.0, lambda_seg=0.25, dice_weight=1.0,
                  gradient_accumulation=3, use_amp=False, freeze_backbone_bn=True,
                  resume=True, resume_from_input=False, skip_training=False,
                  time_budget_hours=1, mid_epoch_checkpoint_minutes=30,
                  classification_threshold=0.5, mask_threshold=0.5, run_mode="full")
    values = dict(CONFIG=config, EXP_CFG={"segmentation": True, "implementation_version": "ow_v3"},
                  DEVICE=torch.device("cpu"), PROTOCOL_HASH="matched-protocol", MODEL_SOURCE_HASH="matched-source",
                  VERITASLossV3=VERITASLossV3, build_optimizer=build_optimizer, build_scheduler=build_scheduler,
                  freeze_backbone_bn=freeze_backbone_bn, validate_resume_checkpoint=validate_resume_checkpoint,
                  set_seed=set_seed, build_model=TinyModel, MASKS=range(190335),
                  tqdm=lambda iterable, **kwargs: iterable)
    for name, value in values.items():
        monkeypatch.setattr(module, name, value, raising=False)

    def setup(folder, train_size=8, val_size=8, test_size=8):
        working = tmp_path / folder
        for subfolder in ("checkpoints", "logs", "predictions"):
            (working / subfolder).mkdir(parents=True, exist_ok=True)
        train = TinyDataset(train_size)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=config["batch_size"], sampler=EpochSeededSampler(len(train), config["seed"]),
            generator=torch.Generator().manual_seed(config["seed"]), drop_last=True)
        val_loader = torch.utils.data.DataLoader(TinyDataset(val_size, "val"), batch_size=config["batch_size"])
        test_loader = torch.utils.data.DataLoader(TinyDataset(test_size, "test"), batch_size=config["batch_size"])
        for name, value in dict(WORKING=working, TRAIN_LOADER=train_loader, VAL_LOADER=val_loader,
                                TEST_LOADER=test_loader, NOTEBOOK_START_TIME=time.time()).items():
            monkeypatch.setattr(module, name, value, raising=False)
        return working

    return module, setup


def assert_model_matches(actual, expected):
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, atol=0, rtol=0)


def test_mid_epoch_resume_matches_uninterrupted_weights_scheduler_and_loss(runtime):
    module, setup = runtime
    whole_path = setup("whole")
    expected, done, _, expected_history = module.train_one_experiment()
    assert done == 1
    whole_checkpoint = module.torch_load(whole_path / "checkpoints/toy_ow_v3_last.pt")

    resumed_path = setup("resumed")
    module.CONFIG["time_budget_hours"] = -1  # Stop at the first completed accumulation group.
    _, done, _, _ = module.train_one_experiment()
    assert done == 0
    partial = module.torch_load(resumed_path / "checkpoints/toy_ow_v3_last.pt")
    assert partial["partial"]["step"] == 3 and partial["partial"]["seen"] == 6
    assert partial["scheduler_state_dict"]["last_epoch"] == 1

    setup("resumed")  # Fresh loaders, as in a resumed Kaggle notebook session.
    module.CONFIG["time_budget_hours"] = 1
    actual, done, _, actual_history = module.train_one_experiment()
    assert done == 1
    assert_model_matches(actual, expected)
    actual_checkpoint = module.torch_load(resumed_path / "checkpoints/toy_ow_v3_last.pt")
    assert actual_checkpoint["scheduler_state_dict"] == whole_checkpoint["scheduler_state_dict"]
    assert actual_checkpoint["scheduler_state_dict"]["last_epoch"] == 2  # Group of 3 plus tail of 1.
    for name in ("train_total", "train_classification", "train_segmentation_bce", "train_segmentation_dice"):
        assert actual_history[-1][name] == pytest.approx(expected_history[-1][name], abs=1e-12)
    assert actual_history[-1]["segmentation_valid_samples"] == 7
    assert actual_history[-1]["segmentation_ignored_fake_samples"] == 1


def test_final_test_loads_validation_best_when_resume_is_disabled(runtime, monkeypatch):
    module, setup = runtime
    work = setup("no-resume")
    module.CONFIG.update(resume=False, epochs=2, epochs_per_run=2)
    real_evaluate = module.evaluate_loader
    val_calls, selected_state = [], {}

    def scripted_evaluation(model, loader, split):
        metrics, records = real_evaluate(model, loader, split)
        if split == "val":
            val_calls.append(1)
            metrics["accuracy"] = 1.0 if len(val_calls) == 1 else 0.0
        else:
            selected_state.update({name: value.clone() for name, value in model.state_dict().items()})
        return metrics, records

    monkeypatch.setattr(module, "evaluate_loader", scripted_evaluation)
    result = module.run_notebook()
    best = module.torch_load(work / "checkpoints/toy_ow_v3_best.pt")
    last = module.torch_load(work / "checkpoints/toy_ow_v3_last.pt")
    assert result["best_epoch"] == best["epoch"] == 1
    assert last["epoch"] == 2
    assert any(not torch.equal(best["model_state_dict"][name], tensor)
               for name, tensor in last["model_state_dict"].items())
    for name, tensor in best["model_state_dict"].items():
        torch.testing.assert_close(selected_state[name], tensor, atol=0, rtol=0)
    assert result["split"] == "test" and (work / "experiment_results.csv").exists()


def test_inference_benchmark_preserves_numeric_classification_precision(runtime):
    module, setup = runtime
    module.CONFIG.update(benchmark_inference=True, image_size=2)
    work = setup("benchmark")
    result = module.run_notebook()
    assert isinstance(result["precision"], float) and 0 <= result["precision"] <= 1
    assert result["inference_precision"] == "float32"
    saved = module.pd.read_csv(work / "experiment_results.csv")
    assert module.pd.api.types.is_numeric_dtype(saved["precision"])
    assert saved.precision.iloc[0] == pytest.approx(result["precision"])
    assert saved.inference_precision.iloc[0] == "float32"
    assert (work / "logs/inference_benchmark.json").is_file()


def test_smoke_uses_validation_and_never_writes_final_test_results(runtime, monkeypatch):
    module, setup = runtime
    module.CONFIG.update(run_mode="smoke", batch_size=8)
    work = setup("smoke", train_size=128, val_size=128)
    real_evaluate, observed = module.evaluate_loader, []

    def track_split(model, loader, split):
        observed.append(split)
        assert split == "val", "Smoke must never evaluate Test"
        return real_evaluate(model, loader, split)

    class ForbiddenTestLoader:
        def __iter__(self):
            raise AssertionError("Smoke attempted to read Test")

    monkeypatch.setattr(module, "evaluate_loader", track_split)
    monkeypatch.setattr(module, "TEST_LOADER", ForbiddenTestLoader())
    result = module.run_notebook()
    assert observed == ["val", "val"]
    assert result["status"] == "SMOKE_PASSED" and result["split"] == "val"
    assert result["metrics"]["n"] == 128
    assert (work / "smoke_results.json").is_file()
    assert not (work / "experiment_results.csv").exists()
    assert not list((work / "predictions").iterdir())


@pytest.mark.parametrize("change", ["experiment", "architecture", "protocol"])
def test_checkpoint_validation_precedes_parameter_mutation(runtime, tmp_path, change):
    module, _ = runtime
    model = TinyModel()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint = dict(experiment_name=module.CONFIG["experiment"], exp_cfg=dict(module.EXP_CFG),
                      protocol_hash=module.PROTOCOL_HASH, model_state_dict={name: torch.full_like(value, 99)
                                                                         for name, value in original.items()})
    if change == "experiment":
        checkpoint["experiment_name"] = "other"
    elif change == "architecture":
        checkpoint["exp_cfg"] = {"implementation_version": "ow_v2"}
    else:
        checkpoint["protocol_hash"] = "smoke-protocol"
    path = tmp_path / "invalid.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        module.load_checkpoint(path, model)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)


def test_stages_highest_progress_from_wrapped_and_unpacked_outputs(runtime, tmp_path, monkeypatch):
    module, setup = runtime
    work = setup("staging")
    module.CONFIG["resume_from_input"] = True
    inputs = tmp_path / "inputs"
    (inputs / "older").mkdir(parents=True)
    (inputs / "newer").mkdir()
    name = "toy_ow_v3_last.pt"

    def payload(epoch, step=0):
        return dict(epoch=epoch, partial={"step": step} if step else None,
                    experiment_name=module.CONFIG["experiment"], exp_cfg=module.EXP_CFG,
                    protocol_hash=module.PROTOCOL_HASH)

    torch.save(payload(0, 3), inputs / "older" / name)
    packed = tmp_path / "packed.pt"
    torch.save(payload(1), packed)
    with zipfile.ZipFile(inputs / "newer/_output_.zip", "w") as archive:
        archive.write(packed, f"output/checkpoints/{name}")
    real_path = Path
    monkeypatch.setattr(module, "Path", lambda value: inputs if str(value) == "/kaggle/input" else real_path(value))
    module.stage_checkpoints()
    assert module.torch_load(work / "checkpoints" / name)["epoch"] == 1


def test_stage_rejects_attached_checkpoint_with_wrong_protocol(runtime, tmp_path, monkeypatch):
    module, setup = runtime
    setup("invalid-stage")
    module.CONFIG["resume_from_input"] = True
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    torch.save(dict(epoch=3, experiment_name=module.CONFIG["experiment"], exp_cfg=module.EXP_CFG,
                    protocol_hash="different"), inputs / "toy_ow_v3_last.pt")
    real_path = Path
    monkeypatch.setattr(module, "Path", lambda value: inputs if str(value) == "/kaggle/input" else real_path(value))
    with pytest.raises(ValueError, match="protocol"):
        module.stage_checkpoints()


def test_wrapped_outputs_with_identical_parent_names_do_not_collide(runtime, tmp_path, monkeypatch):
    module, setup = runtime
    work = setup("collision")
    module.CONFIG["resume_from_input"] = True
    inputs = tmp_path / "inputs"
    for folder, epoch in (("a/output", 2), ("b/output", 1)):
        directory = inputs / folder
        directory.mkdir(parents=True)
        checkpoint = tmp_path / f"epoch-{epoch}.pt"
        torch.save(dict(epoch=epoch, experiment_name=module.CONFIG["experiment"],
                        exp_cfg=module.EXP_CFG, protocol_hash=module.PROTOCOL_HASH), checkpoint)
        with zipfile.ZipFile(directory / "_output_.zip", "w") as archive:
            archive.write(checkpoint, "output/checkpoints/toy_ow_v3_last.pt")
    real_path = Path
    monkeypatch.setattr(module, "Path", lambda value: inputs if str(value) == "/kaggle/input" else real_path(value))
    module.stage_checkpoints()
    assert module.torch_load(work / "checkpoints/toy_ow_v3_last.pt")["epoch"] == 2
    assert not list((work / "checkpoints").glob("_stage_*"))
