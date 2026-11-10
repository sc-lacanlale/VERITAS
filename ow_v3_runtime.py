"""Training/evaluation runtime embedded verbatim in the self-contained OW v3 notebooks.

Notebook setup supplies CONFIG, EXP_CFG, INDEX, WORKING, loaders, DEVICE,
PROTOCOL_HASH, MODEL_SOURCE_HASH, EpochSeededSampler, set_seed and build_model.
Importing this module does not start training.
"""
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm.auto import tqdm


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def stage_checkpoints():
    if not CONFIG.get("resume", True) or not CONFIG.get("resume_from_input", True):
        return
    root = Path("/kaggle/input")
    if not root.exists():
        return
    for suffix in ("last", "best"):
        name = f'{CONFIG["experiment"]}_{suffix}.pt'
        dst = WORKING / "checkpoints" / name
        candidates = list(root.rglob(name)) + ([dst] if dst.exists() else [])
        extracted = []
        # Kaggle can wrap outputs in a ZIP; extract only this experiment's files.
        for archive in root.rglob("_output_.zip"):
            with zipfile.ZipFile(archive) as z:
                for member in z.namelist():
                    if Path(member.replace("\\", "/")).name == name:
                        identity = hashlib.sha256((str(archive.resolve()) + ":" + member).encode()).hexdigest()[:20]
                        staged = WORKING / "checkpoints" / ("_stage_" + identity + "_" + name)
                        with z.open(member) as source, staged.open("wb") as target:
                            shutil.copyfileobj(source, target)
                        candidates.append(staged)
                        extracted.append(staged)
        selected, progress = None, (-1, -1)
        for path in candidates:
            ck = torch_load(path)
            validate_resume_checkpoint(ck, CONFIG["experiment"], EXP_CFG, PROTOCOL_HASH)
            score = (int(ck["epoch"]), int((ck.get("partial") or {}).get("step", 0)))
            if score > progress:
                selected, progress = path, score
            del ck
        if selected is not None and selected != dst:
            shutil.copy2(selected, dst)
            print("STAGED", name, "progress", progress)
        for path in extracted:
            path.unlink()


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None):
    ck = torch_load(path)
    validate_resume_checkpoint(ck, CONFIG["experiment"], EXP_CFG, PROTOCOL_HASH)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer_state_dict"])
    if scaler is not None and ck.get("scaler_state_dict"):
        scaler.load_state_dict(ck["scaler_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ck["scheduler_state_dict"])
    return ck


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, best_metric, history, partial=None):
    payload = dict(epoch=epoch, partial=partial, best_metric=best_metric, history=history,
                   model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                   scaler_state_dict=scaler.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                   torch_rng_state=torch.get_rng_state(),
                   cuda_rng_state=torch.cuda.get_rng_state() if DEVICE.type == "cuda" else None,
                   experiment_name=CONFIG["experiment"], config=CONFIG, exp_cfg=EXP_CFG,
                   protocol_hash=PROTOCOL_HASH, model_source_hash=MODEL_SOURCE_HASH,
                   selection_metric="val_accuracy", supervision_policy="real_zero_fake_nonempty_v3")
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate_loader(model, loader, split):
    model.eval()
    true, predicted, records, dice, iou, real_area = [], [], [], [], [], []
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        valid = batch["valid_mask"].to(DEVICE, non_blocking=True)
        outputs = model(images, valid_mask=valid)
        probabilities = outputs["fake_probability"].float().cpu().numpy().reshape(-1)
        labels = batch["label"].numpy().reshape(-1).astype(int)
        guesses = (probabilities >= CONFIG["classification_threshold"]).astype(int)
        true.extend(labels.tolist())
        predicted.extend(guesses.tolist())
        seg = outputs.get("segmentation_prob")
        for j, label in enumerate(labels):
            row = dict(rel_key=batch["rel_key"][j], file_name=batch["file_name"][j], split=split,
                       ground_truth_label=int(label), predicted_label=int(guesses[j]),
                       fake_probability=float(probabilities[j]))
            if seg is not None:
                content = valid[j, 0].bool()
                p = seg[j, 0][content] >= CONFIG["mask_threshold"]
                target = batch["mask"][j, 0].to(DEVICE)[content] > 0.5
                if label == 0:
                    area = float(p.float().mean())
                    real_area.append(area)
                    row["real_positive_fraction"] = area
                elif bool(batch["seg_valid"][j]) and bool(target.any()):
                    inter = float((p & target).sum())
                    denom, union = float(p.sum() + target.sum()), float((p | target).sum())
                    di, io = 2 * inter / max(denom, 1), inter / max(union, 1)
                    dice.append(di)
                    iou.append(io)
                    row.update(dice_teacher_fake=di, iou_teacher_fake=io)
            records.append(row)
    if not true:
        raise RuntimeError(f"No samples available for {split} evaluation")
    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
    metrics = dict(accuracy=float(accuracy_score(true, predicted)),
                   f1=float(f1_score(true, predicted, zero_division=0)),
                   precision=float(precision_score(true, predicted, zero_division=0)),
                   recall=float(recall_score(true, predicted, zero_division=0)), n=len(true),
                   tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))
    if dice:
        metrics.update(dice_teacher_fake_only=float(np.mean(dice)),
                       miou_teacher_fake_only=float(np.mean(iou)), n_seg_teacher_fake=len(dice))
    if real_area:
        metrics.update(real_mean_positive_fraction=float(np.mean(real_area)),
                       real_mask_nonempty_rate=float(np.mean(np.asarray(real_area) > 0)))
    return metrics, records


def train_one_experiment():
    set_seed(CONFIG["seed"])
    model = build_model()
    criterion = VERITASLossV3(CONFIG["lambda_cls"], CONFIG["lambda_seg"],
                             CONFIG["dice_weight"], EXP_CFG["segmentation"])
    optimizer = build_optimizer(model, CONFIG["learning_rate"], CONFIG["weight_decay"],
                                CONFIG["backbone_lr_multiplier"])
    accum = int(CONFIG["gradient_accumulation"])
    full_batches = len(TRAIN_LOADER)
    scheduler = build_scheduler(optimizer, math.ceil(full_batches / accum) * CONFIG["epochs"],
                                CONFIG["warmup_fraction"])
    amp = bool(CONFIG["use_amp"] and DEVICE.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_path = WORKING / "checkpoints" / f'{CONFIG["experiment"]}_best.pt'
    last_path = WORKING / "checkpoints" / f'{CONFIG["experiment"]}_last.pt'
    stage_checkpoints()
    resumed = {}
    if CONFIG.get("resume", True):
        resume_path = last_path if last_path.exists() else best_path
        if resume_path.exists():
            resumed = load_checkpoint(resume_path, model, optimizer, scaler, scheduler)
    elif last_path.exists() or best_path.exists():
        raise RuntimeError("Fresh training requires a new working session; existing checkpoints are present")
    completed = int(resumed.get("epoch", 0))
    best = float(resumed.get("best_metric", -1))
    history = list(resumed.get("history", []))
    partial = resumed.get("partial")
    print("START OF RUN", CONFIG["experiment"], completed, "/", CONFIG["epochs"])
    if CONFIG.get("skip_training", False):
        if completed != CONFIG["epochs"] or partial:
            raise RuntimeError("Evaluation-only mode requires a completed full checkpoint")
        return model, completed, best, history
    stop = min(CONFIG["epochs"], completed + int(CONFIG["epochs_per_run"]))
    for epoch in range(completed, stop):
        offset = int(partial["step"]) if partial and int(partial["epoch"]) == epoch else 0
        TRAIN_LOADER.sampler.set_epoch(epoch, offset * CONFIG["batch_size"])
        TRAIN_LOADER.dataset.epoch = epoch
        TRAIN_LOADER.generator.manual_seed(CONFIG["seed"] + epoch)
        set_seed(CONFIG["seed"] + epoch)
        if offset:
            torch.set_rng_state(resumed["torch_rng_state"])
            if DEVICE.type == "cuda" and resumed.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(resumed["cuda_rng_state"], DEVICE)
        sums = dict(partial.get("sums", {})) if offset else {}
        seen = int(partial.get("seen", 0)) if offset else 0
        prior_seconds = float(partial.get("seconds", 0)) if offset else 0
        partial = None
        started = last_save = time.time()
        model.train()
        if CONFIG["freeze_backbone_bn"]:
            freeze_backbone_bn(model)
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(tqdm(TRAIN_LOADER, desc=f'epoch {epoch+1}', leave=True)):
            step = offset + i
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)
            valid = batch["valid_mask"].to(DEVICE, non_blocking=True)
            trusted = batch["seg_valid"].to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=amp):
                outputs = model(images, valid_mask=valid)
                losses = criterion(outputs, labels, masks, valid_mask=valid, seg_valid=trusted)
                group_size = min(accum, full_batches - (step // accum) * accum)
                loss = losses["total"] / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}, batch {step+1}")
            scaler.scale(loss).backward()
            boundary = (step + 1) % accum == 0 or step + 1 == full_batches
            if boundary:
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            size = images.shape[0]
            seen += size
            for key in ("total", "classification", "segmentation_bce", "segmentation_dice"):
                sums[key] = sums.get(key, 0.0) + float(losses[key].detach()) * size
            for key in ("segmentation_valid_samples", "segmentation_positive_samples", "segmentation_ignored_fake_samples"):
                sums[key] = sums.get(key, 0.0) + float(losses[key])
            elapsed = time.time() - NOTEBOOK_START_TIME
            if i + 1 == min(10, max(full_batches - offset, 1)):
                rate = (i + 1) * CONFIG["batch_size"] / max(time.time() - started, 1e-6)
                throughput = dict(epoch=epoch+1, next_step=step+1, images_per_second=rate,
                                  estimated_training_only_epoch_hours=len(TRAIN_LOADER.dataset)/rate/3600,
                                  excludes_validation_and_setup=True)
                (WORKING / "logs/throughput.json").write_text(json.dumps(throughput, indent=2), encoding="utf-8")
                print("TRAIN_THROUGHPUT", json.dumps(throughput))
            out_of_time = elapsed >= CONFIG["time_budget_hours"] * 3600
            if boundary and (out_of_time or time.time() - last_save >= CONFIG["mid_epoch_checkpoint_minutes"] * 60):
                state = dict(epoch=epoch, step=step+1, sums=sums, seen=seen,
                             seconds=prior_seconds+time.time()-started)
                save_checkpoint(last_path, model, optimizer, scaler, scheduler, epoch, best, history, state)
                last_save = time.time()
                if out_of_time:
                    print("PARTIAL_EPOCH_SAVED", json.dumps(state))
                    return model, epoch, best, history
            del images, labels, masks, valid, trusted, outputs, losses, loss
        validation, _ = evaluate_loader(model, VAL_LOADER, "val")
        row = dict(epoch=epoch+1, seconds=prior_seconds+time.time()-started,
                   **{"train_"+k: v/max(seen, 1) for k, v in sums.items()
                      if not k.startswith("segmentation_") or k.endswith(("bce", "dice"))},
                   **{"val_"+k: v for k, v in validation.items()},
                   segmentation_valid_samples=sums.get("segmentation_valid_samples", 0),
                   segmentation_ignored_fake_samples=sums.get("segmentation_ignored_fake_samples", 0),
                   backbone_lr=optimizer.param_groups[0]["lr"],
                   heads_lr=optimizer.param_groups[-1]["lr"])
        history.append(row)
        score = float(validation["accuracy"])
        if score >= best:
            best = score
            save_checkpoint(best_path, model, optimizer, scaler, scheduler, epoch+1, best, history)
        save_checkpoint(last_path, model, optimizer, scaler, scheduler, epoch+1, best, history)
        pd.DataFrame(history).to_csv(WORKING / "logs/validation_history.csv", index=False)
        print("EPOCH_COMPLETE", json.dumps(row))
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return model, stop, best, history


@torch.no_grad()
def benchmark_inference(model):
    batch = next(iter(VAL_LOADER))
    images, valid = batch["image"].to(DEVICE), batch["valid_mask"].to(DEVICE)
    model.eval()
    model(images, valid_mask=valid)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(3):
        model(images, valid_mask=valid)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    milliseconds = (time.perf_counter()-started) * 1000 / 3
    result = dict(batch_size=len(images), image_size=CONFIG["image_size"],
                  batch_latency_ms=milliseconds, per_image_ms_at_batch_size=milliseconds/len(images),
                  peak_allocated_vram_bytes=torch.cuda.max_memory_allocated() if DEVICE.type == "cuda" else None,
                  device=torch.cuda.get_device_name() if DEVICE.type == "cuda" else "cpu",
                  torch_version=torch.__version__, inference_precision="float32", includes_preprocessing=False,
                  repeats=3, warmup_batches=1)
    (WORKING / "logs/inference_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("INFERENCE_BENCHMARK", json.dumps(result))
    return result


def run_notebook():
    model, done, best, history = train_one_experiment()
    state = dict(experiment=CONFIG["experiment"], epochs_completed=done, epochs_total=CONFIG["epochs"],
                 protocol_hash=PROTOCOL_HASH, model_source_hash=MODEL_SOURCE_HASH, exp_cfg=EXP_CFG)
    (WORKING / "logs/run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if done < CONFIG["epochs"]:
        print("RUN_SAVED_FOR_CONTINUATION", json.dumps(state))
        return state
    best_path = WORKING / "checkpoints" / f'{CONFIG["experiment"]}_best.pt'
    selected = load_checkpoint(best_path, model)
    state["best_epoch"] = int(selected["epoch"])
    del selected
    if CONFIG["run_mode"] == "smoke":
        metrics, _ = evaluate_loader(model, VAL_LOADER, "val")
        assert metrics["n"] == 128 and len(MASKS) == 190335
        assert all(math.isfinite(metrics[k]) for k in ("accuracy", "f1", "precision", "recall"))
        assert done == 1
        state.update(split="val", metrics=metrics, status="SMOKE_PASSED")
        (WORKING / "smoke_results.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        if CONFIG.get("benchmark_inference", False):
            benchmark_inference(model)
        print("TRAIN_SMOKE_VALIDATION_PASSED", CONFIG["experiment"])
        return state
    if time.time() - NOTEBOOK_START_TIME >= CONFIG["time_budget_hours"] * 3600:
        print("TRAINING_COMPLETE_TEST_PENDING_TIME_BUDGET", json.dumps(state))
        return state
    metrics, records = evaluate_loader(model, TEST_LOADER, "test")
    if CONFIG.get("benchmark_inference", False):
        metrics.update(benchmark_inference(model))
    pd.DataFrame(records).to_csv(WORKING / "predictions" / f'{CONFIG["experiment"]}_predictions.csv', index=False)
    row = dict(**state, **metrics, split="test", run_mode="full", implementation_version="ow_v3", status="ok")
    row["exp_cfg"] = json.dumps(EXP_CFG, sort_keys=True)
    pd.DataFrame([row]).to_csv(WORKING / "experiment_results.csv", index=False)
    print("FINAL_TEST", json.dumps(row))
    return row
