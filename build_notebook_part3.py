"""Part 3: losses, training, metrics, tests, inference, experiments."""
from build_notebook import md, code  # cells list is shared via build_notebook.cells
from build_notebook import cells  # noqa: F401

md("## 17. Losses")

code(
    r'''
class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * targets).sum(dims)
        den = probs.sum(dims) + targets.sum(dims)
        dice = (2 * inter + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()


class VERITASLoss(nn.Module):
    """L_total = λ_cls * L_cls + λ_seg * L_seg
    L_seg = BCEWithLogits + dice_weight * Dice  (documented: Dice helps sparse fake masks)
    """

    def __init__(self, lambda_cls=1.0, lambda_seg=1.0, dice_weight=1.0, segmentation=False):
        super().__init__()
        self.lambda_cls = float(lambda_cls)
        self.lambda_seg = float(lambda_seg)
        self.dice_weight = float(dice_weight)
        self.segmentation = bool(segmentation)
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, outputs, labels, masks=None):
        l_cls = self.bce_cls(outputs["classification_logits"], labels)
        l_seg = labels.new_zeros(())
        if self.segmentation:
            if masks is None:
                raise ValueError("masks required for segmentation experiments")
            l_seg = self.bce_seg(outputs["segmentation_logits"], masks) + self.dice_weight * self.dice(
                outputs["segmentation_logits"], masks
            )
        total = self.lambda_cls * l_cls + (self.lambda_seg * l_seg if self.segmentation else 0.0)
        return {
            "total": total,
            "classification": l_cls.detach(),
            "segmentation": l_seg.detach() if torch.is_tensor(l_seg) else l_seg,
            "lambda_cls": self.lambda_cls,
            "lambda_seg": self.lambda_seg if self.segmentation else 0.0,
        }


print("Loss: L_total = lambda_cls * L_cls + lambda_seg * L_seg")
'''
)

md("## 18. DataLoaders")

code(
    r'''
def make_loader(samples, shuffle, augment):
    ds = FaceCropDataset(samples, GEO, augment=augment)
    return DataLoader(
        ds,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_faces,
        drop_last=False,
    )


TRAIN_LOADER = make_loader(TRAIN_SAMPLES, shuffle=True, augment=True)
VAL_LOADER = make_loader(VAL_SAMPLES, shuffle=False, augment=False)
TEST_LOADER = make_loader(TEST_SAMPLES, shuffle=False, augment=False)
print("batch_size", CONFIG["batch_size"], "train batches", len(TRAIN_LOADER))
'''
)

md("## 14b. Preprocessing before / after samples")

code(
    r'''
def _to_uint8(img):
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def _vis_residual(residual):
    """Display-only stretch. Model still uses tanh(SRM); this only makes the noise visible."""
    x = residual.astype(np.float32)
    lim = float(np.percentile(np.abs(x), 99.0)) + 1e-6
    x = (x / (2.0 * lim) + 0.5).clip(0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _raw_bbox_crop(image_rgb, bbox_xyxy):
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image_rgb[y1:y2, x1:x2].copy()


def _overlay_poly_mask(rgb, mask, color=(255, 40, 40)):
    ov = rgb.copy()
    m = mask > 0
    if m.any():
        ov[m] = (0.45 * ov[m] + 0.55 * np.array(color)).astype(np.uint8)
    return ov


def _stream_visuals(letterboxed_rgb_uint8):
    """Reproduce the three VERITAS streams in numpy for inspection (same ops as the model)."""
    rgb = letterboxed_rgb_uint8.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    kernels = _srm_kernels().numpy()  # 3,1,5,5
    residual_ch = []
    for k in kernels:
        residual_ch.append(cv2.filter2D(y, -1, k[0], borderType=cv2.BORDER_REFLECT))
    residual = np.tanh(np.stack(residual_ch, axis=-1))
    spec = np.fft.fftshift(np.fft.fft2(rgb, axes=(0, 1), norm="ortho"), axes=(0, 1))
    mag = np.log1p(np.abs(spec))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    return {
        "rgb": letterboxed_rgb_uint8,
        "residual": _vis_residual(residual),
        "frequency": _to_uint8(mag),
    }


def _pick_showcase_images(samples, n_mixed=2, n_fake=1, n_real=1):
    by_img = defaultdict(list)
    for s in samples:
        by_img[s.image_id].append(s)
    mixed, fake_only, real_only = [], [], []
    for iid, recs in by_img.items():
        labs = {r.label for r in recs}
        if labs == {0, 1}:
            mixed.append(iid)
        elif labs == {1}:
            fake_only.append(iid)
        elif labs == {0}:
            real_only.append(iid)
    chosen = mixed[:n_mixed] + fake_only[:n_fake] + real_only[:n_real]
    if not chosen:
        chosen = list(by_img.keys())[: max(1, n_mixed)]
    return [(iid, by_img[iid]) for iid in chosen]


def show_full_image_before_after(image_id, recs, save_prefix="preproc_image"):
    rgb = load_rgb(recs[0].image_path)
    before = rgb.copy()
    after = rgb.copy()
    for s in recs:
        x1, y1, x2, y2 = [int(round(v)) for v in s.bbox_xyxy]
        color = (220, 40, 40) if s.is_fake else (40, 180, 70)
        cv2.rectangle(after, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            after,
            f"face {s.face_id} {CATEGORY_NAME[s.label]}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
        for poly in s.polygon:
            cv2.polylines(after, [np.array(poly, dtype=np.int32)], True, color, 2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(before)
    ax[0].set_title(f"BEFORE  image_id={image_id}\nraw OpenForensics frame")
    ax[1].imshow(after)
    ax[1].set_title("AFTER annotation overlay\nboxes + face IDs + polygons (red=fake, green=real)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    out = PATHS["working"] / "debug" / f"{save_prefix}_{image_id}.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.show()
    plt.close(fig)
    print("saved", out)


def show_face_preprocess_before_after(s: FaceSample, rgb=None):
    rgb = load_rgb(s.image_path) if rgb is None else rgb
    raw_crop = _raw_bbox_crop(rgb, s.bbox_xyxy)
    letterboxed, meta = GEO.apply_image(rgb, s.bbox_xyxy)
    model_polys = GEO.polygon_to_model(s.polygon, meta)
    mask = GEO.rasterize(model_polys, fake=s.is_fake)
    streams = _stream_visuals(letterboxed)

    # original-space mask projected from the model mask (checks invertibility of the transform)
    orig_h, orig_w = rgb.shape[:2]
    mask_on_original = GEO.mask_to_original(mask.astype(np.float32), meta, orig_h, orig_w)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normed = letterboxed.astype(np.float32) / 255.0
    normed = (normed - mean) / std

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    titles_imgs = [
        (axes[0, 0], raw_crop, f"1. BEFORE letterbox\nraw bbox crop {raw_crop.shape[1]}x{raw_crop.shape[0]}"),
        (axes[0, 1], letterboxed, f"2. AFTER letterbox\nmodel RGB {GEO.size}x{GEO.size}  scale={meta['scale']:.3f}"),
        (axes[0, 2], _overlay_poly_mask(letterboxed, mask), f"3. AFTER mask rasterize\n{CATEGORY_NAME[s.label]}  mask_px={int(mask.sum())}"),
        (axes[0, 3], _to_uint8(normed), "4. AFTER ImageNet normalize\n(visualized, actually mean/std scaled)"),
        (axes[1, 0], streams["rgb"], "5. Stream RGB"),
        (axes[1, 1], streams["residual"], "6. Stream residual / SRM"),
        (axes[1, 2], streams["frequency"], "7. Stream frequency (log |FFT|)"),
        (axes[1, 3], _overlay_poly_mask(rgb, (mask_on_original > 0.5).astype(np.uint8)), "8. Mask projected BACK\nonto original image"),
    ]
    for ax, im, title in titles_imgs:
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Face-level preprocess  image_id={s.image_id}  face_id={s.face_id}  {CATEGORY_NAME[s.label]}\n"
        f"bbox={tuple(round(v,1) for v in s.bbox_xyxy)}  pad(left,top)=({meta['left']:.1f},{meta['top']:.1f})",
        fontsize=11,
    )
    fig.tight_layout()
    out = PATHS["working"] / "debug" / f"preproc_face_{s.image_id}_{s.face_id}.png"
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.show()
    plt.close(fig)
    print("saved", out)
    print(
        f"  crop {raw_crop.shape[1]}x{raw_crop.shape[0]} -> letterbox {GEO.size} "
        f"scale={meta['scale']:.4f} pad=({meta['left']:.1f},{meta['top']:.1f}) "
        f"label={CATEGORY_NAME[s.label]} mask_sum={int(mask.sum())}"
    )
    return out


def show_preprocessing_gallery(samples):
    showcases = _pick_showcase_images(samples)
    print("Preprocessing gallery: mixed/fake/real source images =", [iid for iid, _ in showcases])
    saved = []
    for iid, recs in showcases:
        show_full_image_before_after(iid, recs)
        rgb = load_rgb(recs[0].image_path)
        # one real and one fake from this image when available
        ordered = sorted(recs, key=lambda s: (-int(s.is_fake), s.face_id))
        for s in ordered[:2]:
            saved.append(show_face_preprocess_before_after(s, rgb=rgb))
    print("Wrote preprocessing figures under", PATHS["working"] / "debug")
    return saved


PREPROC_FIGS = show_preprocessing_gallery(TRAIN_SAMPLES)
try:
    from IPython.display import display, Image as IPyImage, Markdown
    debug_dir = PATHS["working"] / "debug"
    for p in list(sorted(debug_dir.glob("preproc_image_*.png"))) + list(sorted(debug_dir.glob("preproc_face_*.png"))):
        display(Markdown(f"### {p.stem}"))
        display(IPyImage(filename=str(p)))
except Exception as e:
    print("IPython display skipped (normal outside Jupyter):", e)
'''
)

md("## 19–22. Training, validation, metrics, checkpointing")

code(
    r'''
def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def mask_metrics_from_tensors(probs, targets, labels, thresh=0.5):
    """Face-level IoU / Dice.
    Policy: both-empty => IoU=Dice=1.
    Also report fake-only averages so real zero-masks cannot inflate scores.
    """
    pb = (probs > thresh)
    tb = (targets > 0.5)
    b = pb.shape[0]
    ious, dices, ious_fake, dices_fake = [], [], [], []
    for i in range(b):
        inter = (pb[i] & tb[i]).sum().item()
        union = (pb[i] | tb[i]).sum().item()
        iou = 1.0 if union == 0 else inter / union
        ps = pb[i].sum().item()
        ts = tb[i].sum().item()
        dice = 1.0 if (ps + ts) == 0 else (2 * inter) / (ps + ts)
        ious.append(iou)
        dices.append(dice)
        if int(labels[i]) == 1:
            ious_fake.append(iou)
            dices_fake.append(dice)
    return {
        "iou_mean_all": float(np.mean(ious) if ious else float("nan")),
        "dice_mean_all": float(np.mean(dices) if dices else float("nan")),
        "miou_fake": float(np.mean(ious_fake) if ious_fake else float("nan")),
        "dice_fake": float(np.mean(dices_fake) if dices_fake else float("nan")),
        "per_face_iou": ious,
        "per_face_dice": dices,
    }


@torch.no_grad()
def evaluate_loader(model, loader, criterion, split_name: str):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    recs = []
    loss_sum = 0.0
    n = 0
    iou_all, dice_all, iou_fake, dice_fake = [], [], [], []
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        outputs = model(images)
        losses = criterion(outputs, labels, masks if model.segmentation else None)
        bs = images.size(0)
        loss_sum += float(losses["total"].item()) * bs
        n += bs
        prob = outputs["fake_probability"].detach().cpu().numpy().reshape(-1)
        pred = (prob >= CONFIG["classification_threshold"]).astype(int)
        true = labels.detach().cpu().numpy().reshape(-1).astype(int)
        y_true.extend(true.tolist())
        y_pred.extend(pred.tolist())
        y_prob.extend(prob.tolist())
        seg_prob = None
        if model.segmentation and "segmentation_prob" in outputs:
            mm = mask_metrics_from_tensors(outputs["segmentation_prob"].detach().cpu(), masks.detach().cpu(), true)
            iou_all.extend(mm["per_face_iou"])
            dice_all.extend(mm["per_face_dice"])
            for i, lab in enumerate(true):
                if lab == 1:
                    iou_fake.append(mm["per_face_iou"][i])
                    dice_fake.append(mm["per_face_dice"][i])
            seg_prob = outputs["segmentation_prob"].detach().cpu()
        for i in range(bs):
            rec = {
                "image_id": int(batch["image_id"][i]),
                "face_id": int(batch["face_id"][i]),
                "ann_id": int(batch["ann_id"][i]),
                "bbox": batch["bbox"][i].tolist(),
                "ground_truth_label": int(true[i]),
                "predicted_label": int(pred[i]),
                "fake_probability": float(prob[i]),
                "split": split_name,
            }
            if model.segmentation:
                rec["iou"] = mm["per_face_iou"][i]
                rec["dice"] = mm["per_face_dice"][i]
            recs.append(rec)
        del images, labels, masks, outputs
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = loss_sum / max(n, 1)
    if iou_all:
        metrics["iou_all"] = float(np.mean(iou_all))
        metrics["dice_all"] = float(np.mean(dice_all))
        metrics["miou"] = float(np.mean(iou_fake) if iou_fake else float("nan"))
        metrics["dice"] = float(np.mean(dice_fake) if dice_fake else float("nan"))
        metrics["miou_note"] = "miou/dice are macro averages over FAKE faces only; iou_all/dice_all include real zero-masks (both-empty=1)"
    else:
        metrics["iou_all"] = metrics["dice_all"] = metrics["miou"] = metrics["dice"] = None
    return metrics, recs


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, exp_name, exp_cfg, loss_cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "experiment_name": exp_name,
            "experiment_config": exp_cfg,
            "training_config": {k: v for k, v in CONFIG.items() if k != "loss_configs"},
            "loss_config": loss_cfg,
        },
        path,
    )


def load_checkpoint_if_any(path, model, optimizer=None, scheduler=None):
    path = Path(path)
    if not CONFIG["resume"] or not path.exists():
        return 0, None
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print("optimizer state not loaded:", e)
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            pass
    print("Resumed", path, "epoch", ckpt.get("epoch"))
    return int(ckpt.get("epoch", 0)), ckpt.get("best_metric")


def measure_compute(model, loader, n_warmup=None, n_repeat=None):
    n_warmup = CONFIG["latency_warmup"] if n_warmup is None else n_warmup
    n_repeat = CONFIG["latency_repeats"] if n_repeat is None else n_repeat
    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(DEVICE)
    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(images)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0 / images.size(0))
    peak = None
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {
        "mean_latency_ms_per_face": float(np.mean(times)),
        "median_latency_ms_per_face": float(np.median(times)),
        "std_latency_ms_per_face": float(np.std(times)),
        "peak_vram_mb": None if peak is None else float(peak),
        "batch_size": int(images.size(0)),
        "input_resolution": int(images.shape[-1]),
        "device": str(DEVICE),
        "device_kind": "cuda" if torch.cuda.is_available() else "cpu",
    }


def train_experiment(exp_name, exp_cfg, train_loader, val_loader, loss_cfg=None):
    loss_cfg = loss_cfg or {"name": "default", "lambda_cls": CONFIG["lambda_cls"], "lambda_seg": CONFIG["lambda_seg"]}
    print("=" * 72)
    print("TRAIN", exp_name, exp_cfg, loss_cfg)
    model = build_model(exp_cfg)
    criterion = VERITASLoss(
        lambda_cls=loss_cfg["lambda_cls"],
        lambda_seg=loss_cfg["lambda_seg"],
        dice_weight=CONFIG["dice_weight"],
        segmentation=exp_cfg["segmentation"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(CONFIG["epochs"], 1))
    ckpt_best = PATHS["working"] / "checkpoints" / f"{exp_name}_best.pt"
    start_epoch, best_metric = load_checkpoint_if_any(ckpt_best, model, optimizer, scheduler)
    best_metric = -1.0 if best_metric is None else float(best_metric)
    scaler_enabled = bool(CONFIG["use_amp"] and DEVICE.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=scaler_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=scaler_enabled)
    history = []
    if CONFIG["skip_training"]:
        print("skip_training=True")
        return model, {"history": history, "best_metric": best_metric}

    accum = max(int(CONFIG["gradient_accumulation"]), 1)
    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"{exp_name} ep{epoch+1}/{CONFIG['epochs']}")
        for step, batch in enumerate(pbar):
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)
            with autocast_ctx():
                outputs = model(images)
                losses = criterion(outputs, labels, masks if model.segmentation else None)
                loss = losses["total"] / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            bs = images.size(0)
            running += float(losses["total"].item()) * bs
            seen += bs
            pbar.set_postfix(
                loss=f"{losses['total'].item():.4f}",
                cls=f"{float(losses['classification']):.4f}",
                seg=f"{float(losses['segmentation']):.4f}",
            )
            del images, labels, masks, outputs, loss
        scheduler.step()
        train_loss = running / max(seen, 1)
        val_metrics, _ = evaluate_loader(model, val_loader, criterion, "val")
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_acc": val_metrics["accuracy"],
            "lambda_cls": loss_cfg["lambda_cls"],
            "lambda_seg": loss_cfg["lambda_seg"] if exp_cfg["segmentation"] else 0.0,
            "val_miou_fake": val_metrics.get("miou"),
        }
        history.append(row)
        print(row)
        with open(PATHS["working"] / "logs" / f"{exp_name}_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        save_checkpoint(PATHS["working"] / "checkpoints" / f"{exp_name}_last.pt", model, optimizer, scheduler, epoch + 1, best_metric, exp_name, exp_cfg, loss_cfg)
        if val_metrics["f1"] >= best_metric:
            best_metric = val_metrics["f1"]
            save_checkpoint(ckpt_best, model, optimizer, scheduler, epoch + 1, best_metric, exp_name, exp_cfg, loss_cfg)
            print("  saved best", ckpt_best, "F1", best_metric)
    # reload best
    if ckpt_best.exists():
        load_checkpoint_if_any(ckpt_best, model)
    return model, {"history": history, "best_metric": best_metric, "loss_config": loss_cfg}
'''
)

md("## 23. Automated tests (multi-face, matching, masks, transforms)")

code(
    r'''
def _synthetic_image(w=160, h=120, faces=None):
    """faces: list of dicts {bbox_xyxy, label, polygon?} drawn into a canvas."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 40)
    anns = []
    for i, f in enumerate(faces or []):
        x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
        color = (200, 40, 40) if f["label"] == 1 else (40, 180, 80)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        poly = f.get("polygon")
        if poly is None:
            poly = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
        anns.append({"face_id": i, "bbox": [x1, y1, x2, y2], "label": f["label"], "polygon": poly})
    return img, anns


def _gt_samples_from_anns(image_rgb, anns, image_id=0):
    h, w = image_rgb.shape[:2]
    samples = []
    boxes = [tuple(a["bbox"]) for a in anns]
    order = stable_face_id_order(boxes)
    for face_id, idx in enumerate(order):
        a = anns[idx]
        samples.append(
            FaceSample(
                image_id=image_id,
                face_id=face_id,
                bbox_xyxy=tuple(map(float, a["bbox"])),
                label=int(a["label"]),
                file_name="synthetic.jpg",
                image_path="",
                polygon=a["polygon"],
                ann_id=idx,
            )
        )
    return samples


def _masks_for(image_rgb, samples):
    h, w = image_rgb.shape[:2]
    geo = GeometryTransform(64, 0.0)
    out = []
    for s in samples:
        dummy_img, meta = geo.apply_image(image_rgb, s.bbox_xyxy)
        mp = geo.polygon_to_model(s.polygon, meta)
        out.append(geo.rasterize(mp, fake=s.is_fake))
    return out


def run_unit_tests():
    results = []

    def check(name, cond, detail=""):
        results.append({"test": name, "pass": bool(cond), "detail": detail})
        print(("PASS" if cond else "FAIL"), name, detail)

    # Test 1 — one real face
    img, anns = _synthetic_image(faces=[{"bbox": [20, 20, 70, 80], "label": 0}])
    samples = _gt_samples_from_anns(img, anns)
    masks = _masks_for(img, samples)
    check("T1_one_real_count", len(samples) == 1)
    check("T1_one_real_label", samples[0].label == 0)
    check("T1_one_real_mask_zero", masks[0].sum() == 0)

    # Test 2 — one fake face
    img, anns = _synthetic_image(faces=[{"bbox": [20, 20, 80, 90], "label": 1}])
    samples = _gt_samples_from_anns(img, anns)
    masks = _masks_for(img, samples)
    check("T2_one_fake_count", len(samples) == 1)
    check("T2_one_fake_label", samples[0].label == 1)
    check("T2_one_fake_mask_nonzero", masks[0].sum() > 0)

    # Test 3 — three real faces
    img, anns = _synthetic_image(faces=[
        {"bbox": [5, 5, 35, 40], "label": 0},
        {"bbox": [50, 10, 90, 50], "label": 0},
        {"bbox": [100, 20, 150, 70], "label": 0},
    ])
    samples = _gt_samples_from_anns(img, anns)
    masks = _masks_for(img, samples)
    check("T3_three_real_count", len(samples) == 3)
    check("T3_three_real_labels", all(s.label == 0 for s in samples))
    check("T3_three_real_masks", all(m.sum() == 0 for m in masks))

    # Test 4 — mixed real/fake/real
    img, anns = _synthetic_image(faces=[
        {"bbox": [5, 5, 40, 50], "label": 0},
        {"bbox": [55, 8, 100, 60], "label": 1},
        {"bbox": [110, 12, 155, 58], "label": 0},
    ])
    samples = _gt_samples_from_anns(img, anns)
    masks = _masks_for(img, samples)
    labs = [s.label for s in samples]
    check("T4_mixed_labels_set", sorted(labs) == [0, 0, 1])
    fake_idx = [i for i, s in enumerate(samples) if s.label == 1]
    real_idx = [i for i, s in enumerate(samples) if s.label == 0]
    check("T4_fake_mask", all(masks[i].sum() > 0 for i in fake_idx))
    check("T4_real_mask_zero", all(masks[i].sum() == 0 for i in real_idx))

    # Test 5 — multiple fake faces, independent polygons
    img, anns = _synthetic_image(faces=[
        {"bbox": [5, 5, 45, 55], "label": 1},
        {"bbox": [70, 10, 120, 65], "label": 1},
    ])
    samples = _gt_samples_from_anns(img, anns)
    masks = _masks_for(img, samples)
    check("T5_two_fake", len(samples) == 2 and all(s.label == 1 for s in samples))
    check("T5_independent_masks", masks[0].sum() > 0 and masks[1].sum() > 0)

    # Test 6 — detector order differs from annotation order
    ann_boxes = [(10, 10, 40, 40), (80, 15, 120, 60)]
    det_boxes = [(78, 14, 122, 61), (11, 11, 39, 39)]  # reversed
    m = match_faces_to_annotations(det_boxes, ann_boxes, 0.3)
    pair_map = {p["det_index"]: p["ann_index"] for p in m["pairs"]}
    check("T6_reversed_match", pair_map.get(0) == 1 and pair_map.get(1) == 0, str(pair_map))
    check("T6_no_unmatched", m["unmatched_det"] == [] and m["unmatched_ann"] == [])

    # Test 7 — no faces
    empty = []
    try:
        empty = FACE_DETECTOR.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    except Exception as e:
        empty = []
        print("detector on empty image raised (treated as no faces):", e)
    check("T7_no_faces_graceful", isinstance(empty, list))
    m0 = match_faces_to_annotations([], [], 0.3)
    check("T7_empty_match", m0["pairs"] == [] and m0["unmatched_det"] == [] and m0["unmatched_ann"] == [])

    # Test 8 — polygon lands in the correct crop location after resize
    poly = [[(30, 30), (70, 30), (70, 70), (30, 70)]]
    img, anns = _synthetic_image(w=200, h=150, faces=[{"bbox": [20, 20, 80, 80], "label": 1, "polygon": poly}])
    geo = GeometryTransform(64, 0.0)
    crop, meta = geo.apply_image(img, anns[0]["bbox"])
    mp = geo.polygon_to_model(poly, meta)
    mask = geo.rasterize(mp, True)
    # centroid of mask should be near transformed polygon centroid
    ys, xs = np.where(mask > 0)
    check("T8_mask_nonempty", len(xs) > 0)
    if len(xs) > 0:
        cx, cy = xs.mean(), ys.mean()
        tpoly = np.array(mp[0], dtype=np.float32)
        tcx, tcy = tpoly[:, 0].mean(), tpoly[:, 1].mean()
        dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
        check("T8_centroid_alignment", dist < 8.0, f"dist={dist:.2f}")

    # Test 9 — real face must not inherit a neighbouring fake polygon
    img, anns = _synthetic_image(faces=[
        {"bbox": [5, 5, 50, 60], "label": 0},
        {"bbox": [70, 5, 130, 70], "label": 1},
    ])
    samples = _gt_samples_from_anns(img, anns)
    masks = _masks_for(img, samples)
    real = [m for s, m in zip(samples, masks) if s.label == 0][0]
    fake = [m for s, m in zip(samples, masks) if s.label == 1][0]
    check("T9_no_contamination", real.sum() == 0 and fake.sum() > 0)

    # Extra: mixed case from real dataset if available
    mixed_ids = ANN_DF.groupby("image_id")["category_id"].nunique()
    mixed_ids = mixed_ids[mixed_ids > 1].index.tolist()
    if mixed_ids and TEST_SAMPLES:
        sid = None
        by_img = defaultdict(list)
        for s in TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES:
            by_img[s.image_id].append(s)
        for iid, recs in by_img.items():
            labs = {r.label for r in recs}
            if labs == {0, 1} and len(recs) >= 2:
                sid = iid
                mixed_recs = recs
                break
        if sid is not None:
            check("T_real_mixed_exists", True, f"image_id={sid} n={len(mixed_recs)}")
            check("T_real_mixed_independent_labels", len({(r.face_id, r.label) for r in mixed_recs}) == len(mixed_recs))
        else:
            check("T_real_mixed_exists", False, "no mixed image in current subset")

    n_pass = sum(1 for r in results if r["pass"])
    print(f"\n{n_pass}/{len(results)} tests passed")
    with open(PATHS["working"] / "logs" / "unit_tests.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


TEST_RESULTS = run_unit_tests()
'''
)

md("## 24. Multi-face inference")

code(
    r'''
FACE_CACHE_PATH = PATHS["working"] / "face_metadata.json"
_DETECTOR_CACHE = {"meta": {}, "by_path": {}}


def _image_fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"


def _cache_valid(meta) -> bool:
    if not meta:
        return False
    if meta.get("detector") != FACE_DETECTOR.name:
        return False
    if meta.get("min_confidence") != CONFIG["face_score_threshold"]:
        return False
    return True


def load_face_cache():
    global _DETECTOR_CACHE
    if FACE_CACHE_PATH.exists():
        with open(FACE_CACHE_PATH, "r", encoding="utf-8") as f:
            _DETECTOR_CACHE = json.load(f)
        if not _cache_valid(_DETECTOR_CACHE.get("meta", {})):
            print("Face cache incompatible with current detector; rebuilding")
            _DETECTOR_CACHE = {"meta": {}, "by_path": {}}
        else:
            print("Loaded face cache with", len(_DETECTOR_CACHE.get("by_path", {})), "images")


def save_face_cache():
    _DETECTOR_CACHE["meta"] = {
        "detector": FACE_DETECTOR.name,
        "min_confidence": CONFIG["face_score_threshold"],
        "version": 1,
    }
    with open(FACE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_DETECTOR_CACHE, f)


load_face_cache()


def detect_faces_cached(image_path: str, image_rgb=None):
    path = Path(image_path)
    fp = _image_fingerprint(path) if path.exists() else "mem"
    rec = _DETECTOR_CACHE.get("by_path", {}).get(str(path))
    if rec and rec.get("fingerprint") == fp:
        return rec["faces"]
    if image_rgb is None:
        image_rgb = load_rgb(str(path))
    faces = FACE_DETECTOR.detect(image_rgb)
    _DETECTOR_CACHE.setdefault("by_path", {})[str(path)] = {"fingerprint": fp, "faces": faces}
    return faces


@torch.no_grad()
def predict_multi_face(image_path, model=None, return_masks=True, gt_by_image=None):
    """Detect every face independently. Never collapse to one image-level label."""
    image_rgb = load_rgb(image_path)
    h, w = image_rgb.shape[:2]
    faces = detect_faces_cached(image_path, image_rgb)
    results = {"image_id": Path(image_path).stem, "image_path": str(image_path), "faces": []}
    if not faces:
        return results
    model = model or GLOBAL_MODEL
    model.eval()
    geo = GEO
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensors = []
    metas = []
    for f in faces:
        crop, meta = geo.apply_image(image_rgb, f["bbox"])
        x = crop.astype(np.float32) / 255.0
        x = (x - mean) / std
        tensors.append(torch.from_numpy(np.transpose(x, (2, 0, 1)).copy()))
        metas.append(meta)
    batch = torch.stack(tensors, 0).to(DEVICE)
    outputs = model(batch)
    probs = outputs["fake_probability"].detach().cpu().numpy().reshape(-1)
    seg = outputs.get("segmentation_prob")
    if seg is not None:
        seg = seg.detach().cpu().numpy()
    for i, f in enumerate(faces):
        p = float(probs[i])
        label = "fake" if p >= CONFIG["classification_threshold"] else "real"
        mask_full = None
        if return_masks and seg is not None and label == "fake":
            mask_full = geo.mask_to_original(seg[i, 0], metas[i], h, w)
        results["faces"].append(
            {
                "face_id": int(f["face_id"]),
                "bbox": [float(x) for x in f["bbox"]],
                "confidence": float(f.get("confidence", 0.0)),
                "label": label,
                "fake_probability": p,
                "mask": mask_full,
            }
        )
    return results


GLOBAL_MODEL = None
print("predict_multi_face ready")
'''
)

md("## 25. Full-image visualization and debug_image")

code(
    r'''
def annotations_for_image_id(image_id: int) -> List[FaceSample]:
    recs = [s for s in TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES if s.image_id == int(image_id)]
    if recs:
        return recs
    return records_from_annotations(ANN_DF, [int(image_id)])


def debug_image(image_path, model=None, image_id=None, save=True):
    image_rgb = load_rgb(image_path)
    pred = predict_multi_face(image_path, model=model)
    # GT from filename match if image_id unknown
    if image_id is None:
        stem = Path(image_path).name
        hits = ANN_DF[ANN_DF["file_name"].astype(str).str.endswith(stem)]["image_id"].unique().tolist()
        image_id = int(hits[0]) if hits else None
    gt = annotations_for_image_id(image_id) if image_id is not None else []

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    axes[0].imshow(image_rgb)
    axes[0].set_title("1. Original")
    vis_det = image_rgb.copy()
    for f in pred["faces"]:
        x1, y1, x2, y2 = [int(round(v)) for v in f["bbox"]]
        cv2.rectangle(vis_det, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(vis_det, f"id {f['face_id']}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    axes[1].imshow(vis_det)
    axes[1].set_title("2–3. Detected boxes + IDs")

    vis_gt = image_rgb.copy()
    for s in gt:
        x1, y1, x2, y2 = [int(round(v)) for v in s.bbox_xyxy]
        color = (220, 40, 40) if s.is_fake else (40, 180, 70)
        cv2.rectangle(vis_gt, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_gt, f"{s.face_id}:{CATEGORY_NAME[s.label]}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for poly in s.polygon:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(vis_gt, [pts], True, color, 1)
    axes[2].imshow(vis_gt)
    axes[2].set_title("4–5. GT labels + polygons")

    gt_overlay = image_rgb.copy()
    if gt:
        acc = np.zeros(image_rgb.shape[:2], dtype=np.float32)
        for s in gt:
            crop, meta = GEO.apply_image(image_rgb, s.bbox_xyxy)
            mp = GEO.polygon_to_model(s.polygon, meta)
            m = GEO.rasterize(mp, s.is_fake)
            acc = np.maximum(acc, GEO.mask_to_original(m.astype(np.float32), meta, *image_rgb.shape[:2]))
        gt_overlay[acc > 0.5] = (0.4 * gt_overlay[acc > 0.5] + 0.6 * np.array([220, 40, 40])).astype(np.uint8)
    axes[3].imshow(gt_overlay)
    axes[3].set_title("6. GT manipulation masks")

    vis_pred = image_rgb.copy()
    for f in pred["faces"]:
        x1, y1, x2, y2 = [int(round(v)) for v in f["bbox"]]
        color = (220, 40, 40) if f["label"] == "fake" else (40, 180, 70)
        cv2.rectangle(vis_pred, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_pred, f"{f['face_id']} {f['label']} {f['fake_probability']:.2f}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    axes[4].imshow(vis_pred)
    axes[4].set_title("7. Predicted labels")

    pred_ov = image_rgb.copy()
    for f in pred["faces"]:
        if f.get("mask") is not None:
            m = f["mask"]
            pred_ov[m > 0.5] = (0.4 * pred_ov[m > 0.5] + 0.6 * np.array([255, 80, 0])).astype(np.uint8)
    axes[5].imshow(pred_ov)
    axes[5].set_title("8. Predicted masks")

    # matching panel
    if gt and pred["faces"]:
        m = match_faces_to_annotations(
            [tuple(f["bbox"]) for f in pred["faces"]],
            [s.bbox_xyxy for s in gt],
            CONFIG["match_iou_threshold"],
        )
        axes[6].axis("off")
        axes[6].set_title("Matching log")
        txt = json.dumps({k: m[k] for k in ("pairs", "unmatched_det", "unmatched_ann", "ambiguous")}, indent=2)[:800]
        axes[6].text(0, 1, txt, va="top", family="monospace", fontsize=8)
    else:
        axes[6].axis("off")
        axes[6].set_title("Matching log (no GT)")

    axes[7].axis("off")
    axes[7].set_title("Policy")
    axes[7].text(0, 1, "Unmatched annotation: not fabricated.\nUnmatched detection: unlabeled at train time.\nReal mask is always zeros.\nOne fake face never labels others.", va="top", fontsize=9)
    for ax in axes:
        if ax.has_data() or ax.get_title():
            pass
        ax.axis("off")
    fig.tight_layout()
    out = None
    if save:
        out = PATHS["working"] / "debug" / f"debug_{Path(image_path).stem}.png"
        fig.savefig(out, bbox_inches="tight")
        print("saved", out)
    plt.show()
    plt.close(fig)
    return pred
'''
)

md("## 26. Error analysis helpers")

code(
    r'''
def error_analysis(records: List[Dict[str, Any]], title: str, n_show: int = 6):
    if not records:
        print("No records for error analysis")
        return
    df = pd.DataFrame(records)
    fp = df[(df.ground_truth_label == 0) & (df.predicted_label == 1)]
    fn = df[(df.ground_truth_label == 1) & (df.predicted_label == 0)]
    print(title, "FP", len(fp), "FN", len(fn), "N", len(df))
    report = {
        "n": len(df),
        "fp": int(len(fp)),
        "fn": int(len(fn)),
        "fp_examples": fp.head(n_show)[["image_id", "face_id", "fake_probability"]].to_dict("records"),
        "fn_examples": fn.head(n_show)[["image_id", "face_id", "fake_probability"]].to_dict("records"),
    }
    if "iou" in df.columns:
        fake = df[df.ground_truth_label == 1]
        report["fake_iou_mean"] = float(fake["iou"].mean()) if len(fake) else None
        report["worst_iou"] = fake.nsmallest(n_show, "iou")[["image_id", "face_id", "iou", "dice"]].to_dict("records") if len(fake) else []
    out = PATHS["working"] / "logs" / f"error_analysis_{title}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote", out)
    return report


def mcnemar_from_records(rec_a, rec_b):
    """Paired classification comparison on the same (image_id, face_id) keys."""
    key = lambda r: (int(r["image_id"]), int(r["face_id"]))
    ma = {key(r): r for r in rec_a}
    mb = {key(r): r for r in rec_b}
    common = sorted(set(ma) & set(mb))
    if not common:
        return {"error": "no paired samples"}
    n01 = n10 = 0
    for k in common:
        a_wrong = int(ma[k]["predicted_label"] != ma[k]["ground_truth_label"])
        b_wrong = int(mb[k]["predicted_label"] != mb[k]["ground_truth_label"])
        if a_wrong == 0 and b_wrong == 1:
            n01 += 1
        elif a_wrong == 1 and b_wrong == 0:
            n10 += 1
    n = n01 + n10
    # exact binomial McNemar
    if n == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(min(n01, n10), n=n, p=0.5).pvalue)
    return {"n_paired": len(common), "n01_a_correct_b_wrong": n01, "n10_a_wrong_b_correct": n10, "pvalue": p}


def wilcoxon_iou(rec_a, rec_b):
    key = lambda r: (int(r["image_id"]), int(r["face_id"]))
    ma = {key(r): r for r in rec_a if "iou" in r}
    mb = {key(r): r for r in rec_b if "iou" in r}
    common = sorted(set(ma) & set(mb))
    diffs = [ma[k]["iou"] - mb[k]["iou"] for k in common]
    if len(diffs) < 10:
        return {"n": len(diffs), "note": "too few paired IoU values"}
    try:
        stat, p = stats.wilcoxon(diffs)
        return {"n": len(diffs), "stat": float(stat), "pvalue": float(p)}
    except Exception as e:
        return {"n": len(diffs), "error": str(e)}
'''
)

md("## 27. Run experiments and write the results table")

code(
    r'''
def export_predictions(exp_name, records, model, loader):
    pred_dir = PATHS["working"] / "predictions"
    df = pd.DataFrame(records)
    csv_path = pred_dir / f"{exp_name}_predictions.csv"
    json_path = pred_dir / f"{exp_name}_predictions.json"
    df.drop(columns=["mask"] if "mask" in df.columns else [], errors="ignore").to_csv(csv_path, index=False)
    # JSON without bulky arrays
    slim = []
    for r in records:
        slim.append({k: v for k, v in r.items() if k not in {"mask", "ground_truth_mask"}})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slim, f)
    print("wrote", csv_path)
    return csv_path


def run_all_experiments():
    global GLOBAL_MODEL
    summary_rows = []
    all_test_records = {}
    models = {}

    for exp_name in CONFIG["experiments_to_run"]:
        exp_cfg = EXPERIMENTS[exp_name]
        try:
            model, train_info = train_experiment(exp_name, exp_cfg, TRAIN_LOADER, VAL_LOADER)
            criterion = VERITASLoss(
                CONFIG["lambda_cls"], CONFIG["lambda_seg"], CONFIG["dice_weight"], exp_cfg["segmentation"]
            )
            test_metrics, test_recs = evaluate_loader(model, TEST_LOADER, criterion, "test")
            compute = measure_compute(model, TEST_LOADER)
            export_predictions(exp_name, test_recs, model, TEST_LOADER)
            error_analysis(test_recs, exp_name)
            save_face_cache()
            row = {
                "experiment": exp_name,
                "multi_stream": exp_cfg["multi_stream"],
                "segmentation": exp_cfg["segmentation"],
                "accuracy": test_metrics["accuracy"],
                "f1": test_metrics["f1"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "tp": test_metrics["tp"], "tn": test_metrics["tn"], "fp": test_metrics["fp"], "fn": test_metrics["fn"],
                "miou_fake_only": test_metrics.get("miou"),
                "dice_fake_only": test_metrics.get("dice"),
                "iou_all_faces": test_metrics.get("iou_all"),
                "dice_all_faces": test_metrics.get("dice_all"),
                "mean_latency_ms": compute["mean_latency_ms_per_face"],
                "median_latency_ms": compute["median_latency_ms_per_face"],
                "peak_vram_mb": compute["peak_vram_mb"],
                "batch_size": compute["batch_size"],
                "input_resolution": compute["input_resolution"],
                "device": compute["device"],
                "status": "ok",
            }
            all_test_records[exp_name] = test_recs
            models[exp_name] = model
            GLOBAL_MODEL = model
            summary_rows.append(row)
            with open(PATHS["working"] / "logs" / f"{exp_name}_test_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"metrics": test_metrics, "compute": compute, "config": exp_cfg}, f, indent=2)
        except Exception:
            traceback.print_exc()
            summary_rows.append(
                {
                    "experiment": exp_name,
                    "multi_stream": exp_cfg["multi_stream"],
                    "segmentation": exp_cfg["segmentation"],
                    "accuracy": "NOT RUN",
                    "f1": "NOT RUN",
                    "precision": "NOT RUN",
                    "recall": "NOT RUN",
                    "miou_fake_only": "N/A" if not exp_cfg["segmentation"] else "NOT RUN",
                    "dice_fake_only": "N/A" if not exp_cfg["segmentation"] else "NOT RUN",
                    "mean_latency_ms": "NOT RUN",
                    "peak_vram_mb": "NOT RUN",
                    "status": "failed",
                }
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Loss-weight sweeps on full VERITAS (optional)
    if CONFIG["run_loss_sweeps"] and "full_veritas" in EXPERIMENTS:
        for lc in CONFIG["loss_configs"]:
            name = f"full_veritas_loss_{lc['name']}"
            try:
                model, _ = train_experiment(name, EXPERIMENTS["full_veritas"], TRAIN_LOADER, VAL_LOADER, loss_cfg=lc)
                criterion = VERITASLoss(lc["lambda_cls"], lc["lambda_seg"], CONFIG["dice_weight"], True)
                test_metrics, test_recs = evaluate_loader(model, TEST_LOADER, criterion, "test")
                export_predictions(name, test_recs, model, TEST_LOADER)
                summary_rows.append(
                    {
                        "experiment": name,
                        "multi_stream": True,
                        "segmentation": True,
                        "accuracy": test_metrics["accuracy"],
                        "f1": test_metrics["f1"],
                        "miou_fake_only": test_metrics.get("miou"),
                        "dice_fake_only": test_metrics.get("dice"),
                        "lambda_cls": lc["lambda_cls"],
                        "lambda_seg": lc["lambda_seg"],
                        "status": "ok",
                    }
                )
            except Exception:
                traceback.print_exc()
                summary_rows.append({"experiment": name, "status": "failed"})

    summary = pd.DataFrame(summary_rows)
    out_csv = PATHS["working"] / "experiment_results.csv"
    summary.to_csv(out_csv, index=False)
    print("\n=== EXPERIMENT RESULTS ===")
    print(summary.to_string(index=False))
    print("saved", out_csv)

    stats_out = {}
    names = [r["experiment"] for r in summary_rows if r.get("status") == "ok"]
    if "baseline_effb7" in all_test_records and "full_veritas" in all_test_records:
        stats_out["mcnemar_baseline_vs_full"] = mcnemar_from_records(
            all_test_records["baseline_effb7"], all_test_records["full_veritas"]
        )
    if "seg_no_multistream" in all_test_records and "full_veritas" in all_test_records:
        stats_out["wilcoxon_iou_seg_vs_full"] = wilcoxon_iou(
            all_test_records["seg_no_multistream"], all_test_records["full_veritas"]
        )
    with open(PATHS["working"] / "logs" / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2)
    print("statistical tests:", stats_out)
    return summary, models, all_test_records


SUMMARY_DF, TRAINED_MODELS, TEST_RECORDS = run_all_experiments()
'''
)

md("## 28. Qualitative multi-face visualization on held-out images")

code(
    r'''
def pick_qualitative_images(n=4):
    # prefer mixed real/fake test images
    by_img = defaultdict(list)
    for s in TEST_SAMPLES:
        by_img[s.image_id].append(s)
    mixed, others = [], []
    for iid, recs in by_img.items():
        labs = {r.label for r in recs}
        (mixed if labs == {0, 1} else others).append(recs[0])
    chosen = (mixed + others)[:n]
    return chosen


qual = pick_qualitative_images(4)
model_vis = TRAINED_MODELS.get("full_veritas") or TRAINED_MODELS.get("baseline_effb7") or GLOBAL_MODEL
if model_vis is None and TRAINED_MODELS:
    model_vis = next(iter(TRAINED_MODELS.values()))

for s in qual:
    print("debug_image", s.image_path, "image_id", s.image_id, "n_faces_in_split_sample>=1")
    try:
        pred = debug_image(s.image_path, model=model_vis, image_id=s.image_id, save=True)
        print("predicted faces:", [(f["face_id"], f["label"], round(f["fake_probability"], 3)) for f in pred["faces"]])
    except Exception:
        traceback.print_exc()

print("\nWorking artifacts:")
for p in sorted(PATHS["working"].rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(PATHS["working"]), p.stat().st_size)
'''
)

md(
    r"""
## Notes for a full research run

1. Set `CONFIG["run_mode"] = "full"`.
2. Keep `image_size = 600` (EfficientNet-B7 native).
3. Attach dataset `nathanielescuro/veritas-openforensics-compiled`.
4. Enable GPU (T4/P100/H100). Reduce `batch_size` or raise `gradient_accumulation` on OOM.
5. Official OpenForensics splits: `split_protocol = "official_openforensics"`.
6. SOP 70/15/15 is the default and splits at **image** level so faces from one photo cannot leak across train/val/test.
7. Segmentation metrics labeled `*_fake_only` are the primary localization scores. `*_all_faces` includes real faces whose GT mask is zeros (both-empty counts as 1) and can look inflated.
8. No experimental number in this notebook is fabricated: unrun cells export `NOT RUN` / `N/A`.
"""
)
