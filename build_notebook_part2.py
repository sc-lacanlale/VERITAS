"""Part 2 of VERITAS notebook cell definitions. Imported by assemble_notebook.py."""
from build_notebook import cells, md, code

md("## 8. Face-level data model, geometry, and coordinate transforms")

code(
    r'''
@dataclass
class FaceSample:
    image_id: int
    face_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    label: int  # 0 real, 1 fake
    file_name: str
    image_path: str
    polygon: List[List[Tuple[float, float]]]
    ann_id: Optional[int] = None
    split_hint: str = ""
    match_status: str = "ground_truth"
    detector_score: float = 1.0

    @property
    def is_fake(self) -> bool:
        return int(self.label) == 1


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return float(x), float(y), float(x + w), float(y + h)


def clip_bbox(x1, y1, x2, y2, width, height):
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, x1 + 1.0), width)
    y2 = min(max(y2, y1 + 1.0), height)
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, width, height, margin: float):
    bw, bh = x2 - x1, y2 - y1
    x1e = x1 - bw * margin
    y1e = y1 - bh * margin
    x2e = x2 + bw * margin
    y2e = y2 + bh * margin
    return clip_bbox(x1e, y1e, x2e, y2e, width, height)


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def stable_face_id_order(boxes: Sequence[Tuple[float, float, float, float]]) -> List[int]:
    """Spatial order: top-to-bottom, then left-to-right. Never detector list order."""
    keyed = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        keyed.append(((y1 + y2) * 0.5, (x1 + x2) * 0.5, i))
    keyed.sort()
    return [i for _, _, i in keyed]


class GeometryTransform:
    """One geometric mapping applied to image, bbox, polygon, and mask.

    Pipeline:
        original image pixels
            -> expand/clip face crop (offset x1,y1)
            -> letterbox resize into size x size (scale + pad)
            -> model tensor
    """

    def __init__(self, size: int, face_margin: float = 0.1):
        self.size = int(size)
        self.face_margin = float(face_margin)

    def crop_window(self, bbox_xyxy, image_w, image_h):
        x1, y1, x2, y2 = bbox_xyxy
        return expand_bbox(x1, y1, x2, y2, image_w, image_h, self.face_margin)

    def letterbox_params(self, crop_w: float, crop_h: float) -> Dict[str, float]:
        scale = min(self.size / max(crop_h, 1e-6), self.size / max(crop_w, 1e-6))
        nw = crop_w * scale
        nh = crop_h * scale
        left = (self.size - nw) / 2.0
        top = (self.size - nh) / 2.0
        return {"scale": scale, "left": left, "top": top, "nw": nw, "nh": nh, "crop_w": crop_w, "crop_h": crop_h}

    def apply_image(self, image_rgb: np.ndarray, bbox_xyxy) -> Tuple[np.ndarray, Dict[str, float]]:
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = self.crop_window(bbox_xyxy, w, h)
        x1i, y1i, x2i, y2i = map(int, [round(x1), round(y1), round(x2), round(y2)])
        x1i, y1i, x2i, y2i = clip_bbox(x1i, y1i, x2i, y2i, w, h)
        x1i, y1i, x2i, y2i = int(x1i), int(y1i), int(round(x2i)), int(round(y2i))
        crop = image_rgb[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            crop = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            meta = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **self.letterbox_params(1, 1)}
            return crop, meta
        ch, cw = crop.shape[:2]
        params = self.letterbox_params(cw, ch)
        nw, nh = int(round(params["nw"])), int(round(params["nh"]))
        nw, nh = max(nw, 1), max(nh, 1)
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        left = int(round(params["left"]))
        top = int(round(params["top"]))
        left = min(max(left, 0), self.size - nw)
        top = min(max(top, 0), self.size - nh)
        canvas[top:top + nh, left:left + nw] = resized
        meta = {
            "x1": float(x1i), "y1": float(y1i), "x2": float(x2i), "y2": float(y2i),
            "scale": params["scale"], "left": float(left), "top": float(top),
            "nw": float(nw), "nh": float(nh), "crop_w": float(cw), "crop_h": float(ch),
            "orig_w": float(w), "orig_h": float(h),
        }
        return canvas, meta

    def polygon_to_model(self, polygons, meta) -> List[List[Tuple[float, float]]]:
        out = []
        for poly in polygons:
            pts = []
            for x, y in poly:
                xc = x - meta["x1"]
                yc = y - meta["y1"]
                xm = xc * meta["scale"] + meta["left"]
                ym = yc * meta["scale"] + meta["top"]
                pts.append((xm, ym))
            out.append(pts)
        return out

    def rasterize(self, polygons_model, fake: bool) -> np.ndarray:
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if not fake:
            return mask
        for poly in polygons_model:
            if len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def mask_to_original(self, mask_hw: np.ndarray, meta, orig_h: int, orig_w: int) -> np.ndarray:
        """Project a model-space mask back onto the original image."""
        full = np.zeros((orig_h, orig_w), dtype=np.float32)
        left, top = int(round(meta["left"])), int(round(meta["top"]))
        nw, nh = int(round(meta["nw"])), int(round(meta["nh"]))
        x1, y1 = int(round(meta["x1"])), int(round(meta["y1"]))
        x2, y2 = int(round(meta["x2"])), int(round(meta["y2"]))
        crop_w, crop_h = max(x2 - x1, 1), max(y2 - y1, 1)
        region = mask_hw[top:top + nh, left:left + nw]
        if region.size == 0:
            return full
        restored = cv2.resize(region.astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        y2e = min(y1 + crop_h, orig_h)
        x2e = min(x1 + crop_w, orig_w)
        full[y1:y2e, x1:x2e] = restored[: y2e - y1, : x2e - x1]
        return full


GEO = GeometryTransform(CONFIG["image_size"], CONFIG["face_margin"])
print("GeometryTransform size", GEO.size, "margin", GEO.face_margin)
print("Coordinate convention: x=horizontal, y=vertical, mask shape=[H,W]")
'''
)

md("## 9. Face detector abstraction (MediaPipe Tasks, no deprecated solutions API)")

code(
    r'''
class FaceDetector:
    """Backend-agnostic detector. Output contract is a list of dicts."""

    name = "base"

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError


class MediaPipeTasksFaceDetector(FaceDetector):
    """Current MediaPipe Tasks Face Detector (not mediapipe.solutions)."""

    name = "mediapipe_tasks_blaze_face"

    MODEL_URLS = [
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ]

    def __init__(self, model_path: Path, min_confidence: float = 0.5):
        if not MP_TASKS_AVAILABLE:
            raise RuntimeError("MediaPipe Tasks API is not available")
        self.model_path = Path(model_path)
        self.min_confidence = min_confidence
        options = mp_vision.FaceDetectorOptions(
            base_options=MpBaseOptions(model_asset_path=str(self.model_path)),
            min_detection_confidence=float(min_confidence),
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        import mediapipe as mp_mod
        if image_rgb.ndim != 3:
            return []
        rgb = np.ascontiguousarray(image_rgb)
        mp_image = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        faces = []
        detections = result.detections or []
        for i, det in enumerate(detections):
            bb = det.bounding_box
            x1, y1 = float(bb.origin_x), float(bb.origin_y)
            x2, y2 = x1 + float(bb.width), y1 + float(bb.height)
            score = 0.0
            if det.categories:
                score = float(det.categories[0].score or 0.0)
            faces.append({"face_id": i, "bbox": [x1, y1, x2, y2], "confidence": score})
        order = stable_face_id_order([f["bbox"] for f in faces]) if faces else []
        restab = []
        for new_id, old in enumerate(order):
            item = faces[old]
            item["face_id"] = new_id
            restab.append(item)
        return restab


class OpenCVHaarFaceDetector(FaceDetector):
    """Last-resort fallback. Documented as weaker than MediaPipe Tasks."""

    name = "opencv_haar"

    def __init__(self, min_confidence: float = 0.5):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(xml)
        self.min_confidence = min_confidence

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        faces = []
        for i, (x, y, w, h) in enumerate(boxes):
            faces.append({"face_id": i, "bbox": [float(x), float(y), float(x + w), float(y + h)], "confidence": 1.0})
        order = stable_face_id_order([f["bbox"] for f in faces]) if faces else []
        return [{**faces[old], "face_id": i} for i, old in enumerate(order)]


class GroundTruthFaceDetector(FaceDetector):
    """Returns annotation boxes. Used when we evaluate classification independent of detection."""

    name = "ground_truth"

    def __init__(self, records_by_image: Dict[int, List[FaceSample]]):
        self.records_by_image = records_by_image

    def detect(self, image_rgb: np.ndarray, image_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if image_id is None:
            return []
        recs = self.records_by_image.get(int(image_id), [])
        return [
            {"face_id": r.face_id, "bbox": list(r.bbox_xyxy), "confidence": 1.0}
            for r in recs
        ]


def download_file(url: str, dest: Path) -> bool:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        print("Downloading", url)
        urllib.request.urlretrieve(url, dest)
        return dest.exists() and dest.stat().st_size > 1000
    except Exception as e:
        print("Download failed:", e)
        return False


def build_face_detector() -> FaceDetector:
    model_path = PATHS["working"] / "models" / "blaze_face_short_range.tflite"
    if MP_TASKS_AVAILABLE:
        ok = False
        for url in MediaPipeTasksFaceDetector.MODEL_URLS:
            if download_file(url, model_path):
                ok = True
                break
        if ok:
            try:
                det = MediaPipeTasksFaceDetector(model_path, CONFIG["face_score_threshold"])
                print("Using MediaPipe Tasks FaceDetector:", det.name)
                return det
            except Exception as e:
                print("MediaPipe Tasks init failed:", e)
    print("Falling back to OpenCV Haar cascade. Detection quality will be lower.")
    return OpenCVHaarFaceDetector(CONFIG["face_score_threshold"])


FACE_DETECTOR = build_face_detector()
print("Active detector:", FACE_DETECTOR.name)
'''
)

md("## 10. Face / annotation matching")

code(
    r'''
def match_faces_to_annotations(
    det_boxes: List[Tuple[float, float, float, float]],
    ann_boxes: List[Tuple[float, float, float, float]],
    iou_threshold: float,
) -> Dict[str, Any]:
    """Hungarian-style greedy IoU matching. Detector order is never trusted.

    Policy for unmatched annotation: log it and do not fabricate a detector face.
    Policy for unmatched detection: keep it unlabeled (label=None) for inference;
    drop it from supervised training construction.
    Ambiguous: two dets with IoU to same ann both above threshold after assignment
    of the winner — the loser is unmatched_detector.
    """
    n_d, n_a = len(det_boxes), len(ann_boxes)
    if n_d == 0 and n_a == 0:
        return {"pairs": [], "unmatched_det": [], "unmatched_ann": [], "ambiguous": []}
    iou_mat = np.zeros((n_d, n_a), dtype=np.float32)
    for i, db in enumerate(det_boxes):
        for j, ab in enumerate(ann_boxes):
            iou_mat[i, j] = bbox_iou(db, ab)

    pairs = []
    used_d, used_a = set(), set()
    flat = []
    for i in range(n_d):
        for j in range(n_a):
            flat.append((float(iou_mat[i, j]), i, j))
    flat.sort(reverse=True)
    ambiguous = []
    for iou, i, j in flat:
        if iou < iou_threshold:
            continue
        if i in used_d or j in used_a:
            if i not in used_d and j in used_a and iou >= iou_threshold:
                ambiguous.append({"det_index": i, "ann_index": j, "iou": iou, "reason": "ann_already_taken"})
            continue
        used_d.add(i)
        used_a.add(j)
        pairs.append({"det_index": i, "ann_index": j, "iou": iou})

    unmatched_det = [i for i in range(n_d) if i not in used_d]
    unmatched_ann = [j for j in range(n_a) if j not in used_a]
    return {
        "pairs": pairs,
        "unmatched_det": unmatched_det,
        "unmatched_ann": unmatched_ann,
        "ambiguous": ambiguous,
        "iou_matrix": iou_mat,
    }


# Quick self-check: swapped detector order must still match
_ann = [(10, 10, 50, 50), (200, 20, 260, 90)]
_det = [(198, 18, 262, 92), (12, 11, 49, 51)]  # reversed order
_m = match_faces_to_annotations(_det, _ann, 0.3)
assert _m["pairs"][0]["det_index"] != _m["pairs"][1]["det_index"]
print("Matching self-check:", _m["pairs"])
'''
)

md("## 11–12. Polygon → face-crop mask")

code(
    r'''
def polygon_to_mask(polygon, face_bbox, original_image_size, output_size, is_fake=True, face_margin=None) -> np.ndarray:
    """Rasterize a polygon defined in original image coordinates onto the model grid."""
    geo = GeometryTransform(output_size, CONFIG["face_margin"] if face_margin is None else face_margin)
    orig_h, orig_w = original_image_size
    dummy = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    _, meta = geo.apply_image(dummy, face_bbox)
    polys = polygon if polygon and isinstance(polygon[0], (list, tuple)) and polygon and isinstance(polygon[0][0], (list, tuple)) else [polygon] if polygon else []
    # normalize to List[List[Tuple]]
    norm = []
    for poly in (polygon or []):
        if not poly:
            continue
        if isinstance(poly[0], (list, tuple)):
            norm.append([(float(p[0]), float(p[1])) for p in poly])
        else:
            vals = list(poly)
            norm.append([(vals[k], vals[k + 1]) for k in range(0, len(vals) - 1, 2)])
    model_polys = geo.polygon_to_model(norm, meta)
    return geo.rasterize(model_polys, fake=bool(is_fake))


print("polygon_to_mask ready; real faces always receive a zero mask")
'''
)

md("## 13. Image path resolution and face-level dataset")

code(
    r'''
def resolve_image_path(file_name: str, image_dir: Path) -> Optional[Path]:
    raw = str(file_name).replace("\\", "/")
    candidates = [
        image_dir / Path(raw).name,
        image_dir.parent / raw,
        image_dir / raw,
        Path(raw),
    ]
    # extra_files.csv used split subfolders for a handful of extras; try those too
    for split in ("Train", "Val", "Test-Dev", "Test-Challenge"):
        candidates.append(image_dir / split / Path(raw).name)
    for c in candidates:
        if c.exists():
            return c
    return None


def build_missing_set(paths: Dict[str, Path]) -> set:
    s = set()
    if paths.get("missing_csv") and Path(paths["missing_csv"]).exists():
        m = pd.read_csv(paths["missing_csv"])
        if "file_name" in m.columns:
            s |= set(m["file_name"].map(lambda x: Path(str(x)).name))
        if "orig_image_id" in m.columns:
            s |= set(m["orig_image_id"].astype(str))
    return s


MISSING_NAMES = build_missing_set(PATHS)
print("missing_images.csv names:", len(MISSING_NAMES))


def records_from_annotations(df: pd.DataFrame, image_ids: Sequence[int]) -> List[FaceSample]:
    """Supervised samples come from ground-truth boxes, not from the detector.

    Detector order is irrelevant here. face_id is a stable spatial index per image.
    """
    sub = df[df["image_id"].isin(set(map(int, image_ids)))].copy()
    samples: List[FaceSample] = []
    skipped_missing = 0
    for image_id, g in sub.groupby("image_id"):
        rows = list(g.itertuples(index=False))
        boxes = [xywh_to_xyxy(r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h) for r in rows]
        order = stable_face_id_order(boxes)
        file_name = str(rows[0].file_name)
        path = resolve_image_path(file_name, PATHS["images"])
        if path is None or Path(path).name in MISSING_NAMES:
            skipped_missing += 1
            continue
        for face_id, idx in enumerate(order):
            r = rows[idx]
            samples.append(
                FaceSample(
                    image_id=int(image_id),
                    face_id=int(face_id),
                    bbox_xyxy=boxes[idx],
                    label=int(r.category_id),
                    file_name=file_name,
                    image_path=str(path),
                    polygon=parse_segmentation(r.segmentation),
                    ann_id=int(getattr(r, "ann_id", -1)),
                    split_hint=str(getattr(r, "split", "")),
                    match_status="ground_truth",
                )
            )
    print(f"Built {len(samples)} face samples from {len(set(s.image_id for s in samples))} images; skipped missing {skipped_missing}")
    return samples


class FaceCropDataset(Dataset):
    """Face-level dataset. One item = one face, never a whole-image label."""

    def __init__(self, samples: List[FaceSample], geo: GeometryTransform, augment: bool = False):
        self.samples = samples
        self.geo = geo
        self.augment = augment
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-0.15, 0.15)
            beta = random.uniform(-12, 12)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        bgr = cv2.imread(s.image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            img = np.zeros((self.geo.size, self.geo.size, 3), dtype=np.uint8)
            mask = np.zeros((self.geo.size, self.geo.size), dtype=np.uint8)
            meta = {}
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img, meta = self.geo.apply_image(rgb, s.bbox_xyxy)
            model_polys = self.geo.polygon_to_model(s.polygon, meta)
            mask = self.geo.rasterize(model_polys, fake=s.is_fake)
        if self.augment:
            img, mask = self._augment(img, mask)
        x = img.astype(np.float32) / 255.0
        x = (x - self.imagenet_mean) / self.imagenet_std
        x = np.transpose(x, (2, 0, 1))
        return {
            "image": torch.from_numpy(x.copy()),
            "mask": torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            "label": torch.tensor([float(s.label)], dtype=torch.float32),
            "image_id": s.image_id,
            "face_id": s.face_id,
            "ann_id": -1 if s.ann_id is None else int(s.ann_id),
            "bbox": torch.tensor(s.bbox_xyxy, dtype=torch.float32),
            "path": s.image_path,
        }


def collate_faces(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch], 0),
        "mask": torch.stack([b["mask"] for b in batch], 0),
        "label": torch.stack([b["label"] for b in batch], 0),
        "image_id": [b["image_id"] for b in batch],
        "face_id": [b["face_id"] for b in batch],
        "ann_id": [b["ann_id"] for b in batch],
        "bbox": torch.stack([b["bbox"] for b in batch], 0),
        "path": [b["path"] for b in batch],
    }
    return out
'''
)

md("## 14. Visualization helpers")

code(
    r'''
def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_full_image(image_rgb, faces: List[Dict[str, Any]], title: str = "", save_path: Optional[Path] = None):
    vis = image_rgb.copy()
    overlay = vis.copy()
    h, w = vis.shape[:2]
    for f in faces:
        x1, y1, x2, y2 = [int(round(v)) for v in f["bbox"]]
        label = str(f.get("label", "")).upper()
        prob = f.get("fake_probability", None)
        is_fake = str(label).lower() in {"fake", "1"} or (prob is not None and prob >= CONFIG["classification_threshold"])
        color = (220, 40, 40) if is_fake else (40, 180, 70)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"Face {f.get('face_id', '?')} — {label}"
        if prob is not None:
            text += f" — {float(prob):.2f}"
        cv2.putText(vis, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        mask = f.get("mask")
        if mask is not None and is_fake:
            if mask.shape[:2] != (h, w):
                continue
            binm = (mask > 0.5).astype(np.uint8)
            overlay[binm == 1] = (0.45 * overlay[binm == 1] + 0.55 * np.array(color)).astype(np.uint8)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(vis)
    ax.set_title(title)
    ax.axis("off")
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return vis


print("Visualization helpers ready")
'''
)

md("## 15. Dataset split (image-level, no face leakage)")

code(
    r'''
def existing_image_ids(df: pd.DataFrame) -> pd.DataFrame:
    # Fast path: treat missing_images.csv as the missing set. Resolve paths later per sample.
    uniq = df.drop_duplicates("image_id")[["image_id", "file_name", "split"]].copy()
    uniq["exists"] = ~uniq["file_name"].map(lambda x: Path(str(x)).name in MISSING_NAMES)
    print("images after missing-csv filter:", int(uniq["exists"].sum()), "/", len(uniq))
    return uniq


IMAGE_TABLE = existing_image_ids(ANN_DF)


def split_image_ids(image_table: pd.DataFrame) -> Dict[str, List[int]]:
    protocol = CONFIG["split_protocol"]
    pool = CONFIG["split_pool"]
    table = image_table[image_table["exists"]].copy()
    if pool == "train_val_testdev":
        table = table[table["split"].isin(["Train", "Val", "Test-Dev"])]
    elif pool == "train_only":
        table = table[table["split"] == "Train"]
    elif pool == "all":
        pass
    else:
        raise ValueError(pool)

    rng = np.random.RandomState(CONFIG["seed"])
    if protocol == "official_openforensics":
        splits = {
            "train": sorted(table[table["split"] == "Train"]["image_id"].tolist()),
            "val": sorted(table[table["split"] == "Val"]["image_id"].tolist()),
            "test": sorted(table[table["split"] == "Test-Dev"]["image_id"].tolist()),
        }
        print("Using official OpenForensics Train / Val / Test-Dev")
    elif protocol == "70_15_15":
        ids = table["image_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        splits = {
            "train": sorted(ids[:n_train].tolist()),
            "val": sorted(ids[n_train:n_train + n_val].tolist()),
            "test": sorted(ids[n_train + n_val:].tolist()),
        }
        print("Using SOP image-level 70/15/15 split on pool", pool, "n=", n)
    else:
        raise ValueError(protocol)

    def cap(key, max_n):
        if max_n is None:
            return splits[key]
        rng2 = np.random.RandomState(CONFIG["seed"] + {"train": 1, "val": 2, "test": 3}[key])
        arr = np.array(splits[key], copy=True)
        rng2.shuffle(arr)
        return sorted(arr[:max_n].tolist())

    splits = {
        "train": cap("train", CONFIG["max_train_images"]),
        "val": cap("val", CONFIG["max_val_images"]),
        "test": cap("test", CONFIG["max_test_images"]),
    }
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    for k, v in splits.items():
        print(f"  {k}: {len(v)} images")
    split_path = PATHS["working"] / "logs" / "image_splits.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({k: list(map(int, v)) for k, v in splits.items()}, f)
    print("Saved", split_path)
    return splits


SPLITS = split_image_ids(IMAGE_TABLE)
TRAIN_SAMPLES = records_from_annotations(ANN_DF, SPLITS["train"])
VAL_SAMPLES = records_from_annotations(ANN_DF, SPLITS["val"])
TEST_SAMPLES = records_from_annotations(ANN_DF, SPLITS["test"])
print("face counts train/val/test:", len(TRAIN_SAMPLES), len(VAL_SAMPLES), len(TEST_SAMPLES))
print("train label balance:", sum(s.label for s in TRAIN_SAMPLES), "fake /", len(TRAIN_SAMPLES))
'''
)

md("## 16. Model definitions — configurable VERITAS")

code(
    r'''
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _srm_kernels() -> torch.Tensor:
    """Three SRM high-pass kernels used for the noise/residual stream (Zhou et al. style)."""
    k1 = np.array(
        [[0, 0, 0, 0, 0],
         [0, -1, 2, -1, 0],
         [0, 2, -4, 2, 0],
         [0, -1, 2, -1, 0],
         [0, 0, 0, 0, 0]], dtype=np.float32
    )
    k2 = np.array(
        [[-1, 2, -2, 2, -1],
         [2, -6, 8, -6, 2],
         [-2, 8, -12, 8, -2],
         [2, -6, 8, -6, 2],
         [-1, 2, -2, 2, -1]], dtype=np.float32
    )
    k3 = np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, -2, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=np.float32
    )
    ks = []
    for k in (k1, k2, k3):
        s = np.abs(k).sum()
        ks.append(k / (s if s > 0 else 1.0))
    weight = np.stack(ks, 0)[:, None, :, :]  # 3,1,5,5
    return torch.from_numpy(weight)


class MultiStreamExtractor(nn.Module):
    """RGB + SRM residual + log-FFT frequency streams, fused by channel concat.

    Streams are computed from the RGB face crop so every experiment sees the
    same geometric crop; only the representation changes.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("srm_weight", _srm_kernels())

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(x.device, x.dtype)
        std = IMAGENET_STD.to(x.device, x.dtype)
        return (x * std + mean).clamp(0, 1)

    def forward(self, x_rgb_norm: torch.Tensor) -> torch.Tensor:
        rgb = self._denorm(x_rgb_norm)
        # Residual / SRM on luminance
        y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        residual = F.conv2d(y, self.srm_weight.to(dtype=rgb.dtype), padding=2)
        residual = residual.tanh()
        # Frequency: log magnitude spectrum per RGB channel, min-max per-sample
        spec = torch.fft.fftshift(torch.fft.fft2(rgb, norm="ortho"), dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        b, c, h, w = mag.shape
        mag_flat = mag.reshape(b, c, -1)
        mn = mag_flat.min(dim=-1, keepdim=True).values
        mx = mag_flat.max(dim=-1, keepdim=True).values
        mag = ((mag_flat - mn) / (mx - mn + 1e-6)).reshape(b, c, h, w)
        fused = torch.cat([rgb, residual, mag], dim=1)  # B,9,H,W
        return fused


class FeatureFusion(nn.Module):
    """Project 9-channel fused tensor with an adapted ImageNet stem conv."""

    def __init__(self, conv3: nn.Conv2d):
        super().__init__()
        conv9 = nn.Conv2d(
            9, conv3.out_channels, kernel_size=conv3.kernel_size,
            stride=conv3.stride, padding=conv3.padding, bias=False,
        )
        with torch.no_grad():
            w = conv3.weight  # out,3,k,k
            conv9.weight[:, 0:3].copy_(w)
            conv9.weight[:, 3:6].copy_(w)
            conv9.weight[:, 6:9].copy_(w)
            conv9.weight.mul_(1.0 / 3.0)
        self.conv = conv9

    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ))
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = [b(x) for b in self.branches]
        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[-2:], mode="bilinear", align_corners=False)
        res.append(gap)
        return self.project(torch.cat(res, dim=1))


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, high_ch: int, low_ch: int, num_classes: int = 1):
        super().__init__()
        self.aspp = ASPP(high_ch, 256)
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, high, low, out_size):
        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low)
        x = torch.cat([high, low], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, in_ch: int, dropout: float = 0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, 1),
        )

    def forward(self, feat):
        z = self.pool(feat).flatten(1)
        return self.fc(z)


class EfficientNetBackbone(nn.Module):
    """Wrap torchvision EfficientNet-B7 and expose stride-4 / stride-16 / final maps."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True, multi_stream: bool = False):
        super().__init__()
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b7(weights=weights)
        stem = net.features[0]  # Conv2dNormActivation
        self.stem_rest = nn.Sequential(*list(stem.children())[1:])  # BN + SiLU
        conv0 = stem[0]
        self.multi_stream = multi_stream
        self.stream = MultiStreamExtractor() if multi_stream else None
        if multi_stream:
            self.fusion = FeatureFusion(conv0)
            self.stem_conv = self.fusion.conv
        else:
            self.fusion = None
            self.stem_conv = conv0
        self.stages = nn.ModuleList(list(net.features.children())[1:])
        self.out_channels = 2560
        self.low_stage = 1   # features[2], typically stride 4
        self.high_stage = 4  # features[5], typically stride 16

    def forward(self, x):
        if self.multi_stream:
            x9 = self.stream(x)
            x = self.stem_conv(x9)
        else:
            x = self.stem_conv(x)
        x = self.stem_rest(x)
        low = None
        high = None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.low_stage:
                low = x
            if i == self.high_stage:
                high = x
        if high is None:
            high = x
        if low is None:
            low = x
        return {"final": x, "low": low, "high": high}


class VERITASModel(nn.Module):
    def __init__(self, backbone="efficientnet_b7", multi_stream=False, segmentation=False, pretrained=True):
        super().__init__()
        if backbone != "efficientnet_b7":
            raise ValueError("This study uses EfficientNet-B7 only")
        self.multi_stream = bool(multi_stream)
        self.segmentation = bool(segmentation)
        self.encoder = EfficientNetBackbone(
            in_channels=3, pretrained=pretrained, multi_stream=self.multi_stream
        )
        self.cls_head = ClassificationHead(self.encoder.out_channels)
        # torchvision EfficientNet-B7: features[2]=48ch stride4, features[5]=224ch stride16
        self.seg_head = DeepLabV3PlusHead(224, 48, 1) if self.segmentation else None

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.cls_head(feats["final"])
        out = {"classification_logits": logits, "fake_probability": torch.sigmoid(logits)}
        if self.segmentation:
            seg_logits = self.seg_head(feats["high"], feats["low"], out_size=x.shape[-2:])
            out["segmentation_logits"] = seg_logits
            out["segmentation_prob"] = torch.sigmoid(seg_logits)
        return out


def build_model(exp_cfg: Dict[str, Any]) -> VERITASModel:
    m = VERITASModel(
        backbone=exp_cfg.get("backbone", "efficientnet_b7"),
        multi_stream=exp_cfg["multi_stream"],
        segmentation=exp_cfg["segmentation"],
        pretrained=CONFIG["use_pretrained"],
    ).to(DEVICE)
    if m.segmentation:
        was_training = m.training
        m.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64, device=DEVICE)
            feats = m.encoder(dummy)
            low_ch, high_ch = feats["low"].shape[1], feats["high"].shape[1]
        if m.seg_head.low_proj[0].in_channels != low_ch or m.seg_head.aspp.branches[0][0].in_channels != high_ch:
            m.seg_head = DeepLabV3PlusHead(high_ch, low_ch, 1).to(DEVICE)
            print(f"Adapted DeepLabV3+ head to low_ch={low_ch} high_ch={high_ch}")
        if was_training:
            m.train()
    return m


print("Model factory ready. Variants:")
for _name, _cfg in EXPERIMENTS.items():
    print(f"  {_name}: multi_stream={_cfg['multi_stream']} segmentation={_cfg['segmentation']}")
'''
)
