# VERITAS — FINAL IMPLEMENTATION & EXPERIMENTATION PROMPT

You are an AI coding/research agent responsible for implementing and experimentally evaluating the **VERITAS (Vision-based EfficientNet-Reinforced Image Tampering Authentication and ASPP-Segmentation MTL-Framework)** system described in the provided thesis/SOP and the supplied OpenForensics-derived Kaggle dataset.

The implementation must be done **entirely inside a Kaggle Notebook**.

Your job is NOT simply to make one model run. You must build a modular implementation that allows the different VERITAS configurations required by the study/SOP to be trained, evaluated, compared, and reproduced.

The implementation must preserve the intended research architecture while fixing the existing single-face limitation and correctly supporting images containing multiple faces, including images where some faces are real and others are fake.

---

# 1. PRIMARY RESEARCH REQUIREMENT

The thesis/SOP evaluates the contribution of several architectural components.

Therefore, implement VERITAS as a **configurable experimental framework** with explicit model variants rather than one monolithic model.

The experiments must distinguish at minimum:

### Variant A — Standalone EfficientNet-B7 Baseline

Purpose:

Establish the classification-only baseline.

Architecture:

```text
Face Crop
   ↓
RGB
   ↓
EfficientNet-B7
   ↓
Global Average Pooling
   ↓
Binary Classification Head
   ↓
Real / Fake
```

Requirements:

* EfficientNet-B7 backbone.
* ImageNet pretrained weights where appropriate.
* Single RGB input stream.
* No multi-stream feature extraction.
* No segmentation head.
* Binary classification only.
* No segmentation loss.
* Report classification metrics.
* Measure inference latency and peak VRAM where applicable.

This is the baseline against which the VERITAS variants are evaluated.

Do NOT replace EfficientNet-B7 with B0/B3/etc. unless the existing implementation or SOP explicitly requires it.

---

# 2. VARIANT B — MULTI-STREAM CLASSIFICATION MODEL

Purpose:

Measure the effect of the multi-scale/multi-stream representation independently from segmentation.

Architecture concept:

```text
                    ┌─ RGB Stream ──────────┐
                    │                        │
Face Crop ──────────┼─ Noise/Residual ──────┼─ Feature Fusion
                    │                        │
                    └─ Frequency Stream ────┘
                                             ↓
                                      EfficientNet Encoder
                                             ↓
                                      Classification Head
                                             ↓
                                         Real/Fake
```

Requirements:

* Multi-stream/multi-scale input representation as specified by the existing VERITAS design.
* Include the original RGB representation.
* Include the required noise/residual representation.
* Include the required frequency-domain representation.
* Process/fuse these representations according to the architecture described in the thesis/codebase.
* Classification only.
* NO DeepLabV3+ segmentation head.
* NO segmentation loss.

This experiment exists specifically to determine the effect of multi-stream extraction independently of segmentation.

Do not accidentally compare:

```text
RGB + segmentation
```

when the intended comparison is:

```text
single-stream RGB
vs.
multi-stream representation
```

Keep the experimental variables controlled.

---

# 3. VARIANT C — SEGMENTATION / MTL WITHOUT MULTI-STREAM

Purpose:

Measure the contribution of segmentation and joint task learning independently from multi-stream extraction.

Architecture:

```text
RGB Face Crop
      ↓
EfficientNet-B7
      ↓
Shared Feature Representation
      ├──────────────→ Classification Head
      │
      └──────────────→ DeepLabV3+ / ASPP
                           ↓
                     Segmentation Mask
```

Requirements:

* Single RGB stream.
* EfficientNet-B7.
* Classification head.
* DeepLabV3+ segmentation head.
* ASPP module.
* Joint classification + segmentation training.
* No multi-stream input representation.
* Classification loss + segmentation loss.
* Weighted total loss.

This isolates the contribution of the segmentation task.

---

# 4. VARIANT D — FULL VERITAS

This is the proposed full architecture.

It must combine:

1. Multi-stream/multi-scale feature extraction.
2. EfficientNet-B7.
3. Feature fusion.
4. Classification head.
5. DeepLabV3+.
6. ASPP.
7. Joint classification + segmentation optimization.
8. Weighted multi-task loss.

Conceptually:

```text
                         ┌─ RGB ──────────────┐
                         │                     │
                         ├─ Noise/Residual ───┼─ Multi-Stream
                         │                     │    Fusion
                         └─ Frequency ────────┘
                                      ↓
                               EfficientNet-B7
                                      ↓
                              Shared Feature Map
                                ┌─────┴─────┐
                                ↓           ↓
                       Classification     ASPP
                           Head             ↓
                                ↓       DeepLabV3+
                                ↓           ↓
                           Real/Fake      Mask
```

The full model must retain the intended VERITAS architecture described in the thesis/SOP.

---

# 5. OPTIONAL / REQUIRED ABLATION CONFIGURATION: LOSS-WEIGHT EXPERIMENTS

The SOP explicitly identifies weighted-loss calibration as an experimental factor.

Implement the total loss as a configurable function:

```text
L_total = λ_cls * L_cls + λ_seg * L_seg
```

where:

* `L_cls` = classification loss
* `L_seg` = segmentation loss
* `λ_cls` = classification weight
* `λ_seg` = segmentation weight

Make these configurable from one central configuration object.

The notebook must support testing multiple loss-weight configurations without rewriting the model.

For example:

```python
LOSS_CONFIGS = [
    {"name": "1_1", "lambda_cls": 1.0, "lambda_seg": 1.0},
    {"name": "2_1", "lambda_cls": 2.0, "lambda_seg": 1.0},
    {"name": "1_2", "lambda_cls": 1.0, "lambda_seg": 2.0},
]
```

Do NOT assume these exact ratios are mandated unless the SOP/code explicitly specifies them.

Instead:

1. Inspect the existing project/SOP for intended ratios.
2. Implement the loss function generically.
3. Make the tested ratios configurable.
4. Record every experiment and its configuration.
5. Compare classification and segmentation metrics.

---

# 6. REQUIRED EXPERIMENT MATRIX

Create a central experiment configuration such as:

```python
EXPERIMENTS = {
    "baseline_effb7": {
        "multi_stream": False,
        "segmentation": False,
        "backbone": "efficientnet_b7",
    },

    "multistream_no_seg": {
        "multi_stream": True,
        "segmentation": False,
        "backbone": "efficientnet_b7",
    },

    "seg_no_multistream": {
        "multi_stream": False,
        "segmentation": True,
        "backbone": "efficientnet_b7",
    },

    "full_veritas": {
        "multi_stream": True,
        "segmentation": True,
        "backbone": "efficientnet_b7",
    },
}
```

Adapt this structure to the actual codebase.

The important requirement is that the experiments must cleanly isolate:

| Experiment   | EfficientNet-B7 | Multi-Stream | Segmentation | Classification |
| ------------ | --------------: | -----------: | -----------: | -------------: |
| Baseline     |               ✓ |            ✗ |            ✗ |              ✓ |
| Multi-stream |               ✓ |            ✓ |            ✗ |              ✓ |
| Segmentation |               ✓ |            ✗ |            ✓ |              ✓ |
| Full VERITAS |               ✓ |            ✓ |            ✓ |              ✓ |

If the SOP specifies additional configurations, implement them as well.

Do not silently omit an experiment because the full model already contains its components.

---

# 7. IMPORTANT: THIS IS A MULTI-FACE SYSTEM

The existing implementation must NOT assume that an image contains only one face.

The system must support:

```text
Image
 ├── Face 0
 ├── Face 1
 ├── Face 2
 └── Face 3
```

Each face must be processed independently.

For example, an image containing:

```text
Face A = REAL
Face B = FAKE
Face C = REAL
```

must produce:

```text
Face A → real
Face B → fake + manipulation mask
Face C → real
```

It must NOT produce:

```text
Entire image → fake
```

simply because one face is fake.

---

# 8. FACE-LEVEL DATA MODEL

Create a consistent representation such as:

```python
FaceSample(
    image_id=...,
    face_id=...,
    bbox=...,
    face_crop=...,
    label=...,
    polygon=...,
    mask=...
)
```

Every detected face must have a stable:

```text
image_id
face_id
bbox
label
```

If segmentation annotations exist:

```text
polygon
mask
```

must also be attached to the correct face.

One source image containing three faces should therefore generate three logical face samples:

```text
image_001 / face_000
image_001 / face_001
image_001 / face_002
```

Do not discard the original `image_id`.

---

# 9. DATASET / ANNOTATION INSPECTION

Before writing assumptions about the dataset:

**INSPECT THE ACTUAL KAGGLE DATASET.**

The dataset is:

https://www.kaggle.com/datasets/nathanielescuro/veritas-openforensics-compiled

Inside Kaggle, dataset files will normally be accessible through:

```text
/kaggle/input/
```

but DO NOT assume an exact directory name.

First inspect:

```python
import os

for root, dirs, files in os.walk("/kaggle/input"):
    ...
```

Determine:

* actual dataset root
* image directories
* annotation directories
* `poly.json`
* label files
* face annotation files
* metadata
* existing train/validation/test information
* image naming conventions
* annotation schema
* polygon representation
* face bounding-box representation
* real/fake label representation

Do not invent a schema.

If the actual annotation structure differs from assumptions in this prompt, adapt the implementation to the real structure while preserving the intended functionality.

---

# 10. POLY.JSON IS GROUND TRUTH FOR SEGMENTATION

Inspect the actual `poly.json` structure.

Determine exactly:

* how image IDs are represented
* how face IDs are represented, if available
* how authenticity labels are represented
* how polygons are represented
* whether coordinates are absolute or normalized
* whether multiple polygons can belong to one face
* whether multiple faces can exist in one image

Never assume detector ordering matches annotation ordering.

---

# 11. FACE ↔ ANNOTATION MATCHING

This is critical.

The face detector's output order MUST NOT be assumed to correspond to annotation order.

For example:

```text
Detector:
face 0 = left face
face 1 = right face

Annotation:
annotation 0 = right face
annotation 1 = left face
```

must still match correctly.

Implement a robust matching layer.

Possible matching signals include:

* annotation bbox vs detector bbox IoU
* center-point distance
* normalized coordinate distance
* face size
* containment
* available face IDs
* annotation metadata
* detector confidence

Use the strongest available identifiers first.

If no explicit ID exists, use geometric matching.

Make the matching threshold configurable.

Log:

```text
matched
unmatched detector face
unmatched annotation
ambiguous match
```

Do not silently assign an incorrect annotation.

---

# 12. MULTIPLE FACE LABELS

The implementation must support:

### Case 1 — One real face

```text
Face 0 → real
mask = all zeros
```

### Case 2 — One fake face

```text
Face 0 → fake
mask = polygon rasterization
```

### Case 3 — Multiple real faces

```text
Face 0 → real
Face 1 → real
Face 2 → real
```

all masks must be zero.

### Case 4 — Multiple fake faces

Each fake face receives its own mask.

### Case 5 — Mixed real/fake

Example:

```text
Face 0 → real → zero mask
Face 1 → fake → polygon mask
Face 2 → real → zero mask
Face 3 → fake → polygon mask
```

This case MUST be explicitly tested.

### Case 6 — No faces

Return an empty face result gracefully.

Do not crash.

### Case 7 — Unmatched annotation

Log the issue and handle it according to a clearly documented policy.

Do not fabricate a label.

---

# 13. POLYGON → FACE-CROP MASK

If a face is fake, the binary ground-truth mask must be created from its polygon annotation.

The transformation must be correct.

The original polygon is defined in:

```text
original image coordinates
```

but the model operates on:

```text
face crop coordinates
```

and then:

```text
resized/padded model coordinates
```

Therefore implement an explicit transformation pipeline:

```text
Original Image
      ↓
Original Polygon
      ↓
Face Crop Coordinate System
      ↓
Resize / Padding Transform
      ↓
Model Input Coordinate System
      ↓
Rasterized Binary Mask
```

Do NOT simply resize the polygon independently without accounting for the crop offset and padding.

---

# 14. MASK RULES

For a fake face:

```python
mask[y, x] = 1
```

inside the annotated polygon.

Outside:

```python
mask[y, x] = 0
```

For a real face:

```python
mask[:] = 0
```

The real-face mask must NOT contain the manipulated region of another face.

Never construct a whole-image fake mask simply because another face is fake.

---

# 15. MASK RASTERIZATION

Create a dedicated function such as:

```python
polygon_to_mask(
    polygon,
    face_bbox,
    original_image_size,
    output_size
)
```

It should:

1. convert polygon coordinates to the face crop coordinate system
2. apply the exact image resize transform
3. account for padding/letterboxing if used
4. rasterize the polygon
5. return a binary mask

Use an appropriate rasterization method such as OpenCV `fillPoly`, PIL polygon drawing, or another deterministic method.

Document coordinate conventions clearly:

```text
x = horizontal
y = vertical
```

and:

```text
mask shape = [H, W]
```

---

# 16. IMAGE TRANSFORM CONSISTENCY

The exact same geometric transformation applied to the face image must be applied to the polygon.

Do not have separate undocumented resize logic.

Create a reusable transform object/function that can transform:

```text
image
bbox
polygon
mask
```

consistently.

This is critical for valid segmentation training.

---

# 17. FACE DETECTOR ARCHITECTURE

Create an abstraction:

```python
class FaceDetector:
    def detect(self, image):
        ...
```

Do not hard-code the rest of VERITAS to a specific detector.

The detector must return something like:

```python
[
    {
        "face_id": 0,
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.97,
    },
    ...
]
```

---

# 18. IMPORTANT MEDIAPIPE REQUIREMENT

The thesis may contain older MediaPipe Face Mesh implementation references.

However:

**DO NOT introduce new code using deprecated `mediapipe.solutions.*` APIs.**

Do NOT write new code such as:

```python
mp.solutions.face_mesh
```

or other deprecated `mediapipe.solutions` implementations.

First inspect the installed MediaPipe version:

```python
import mediapipe as mp
print(mp.__version__)
```

Then determine which supported API is available.

Prefer the current MediaPipe Tasks API or another supported face detector.

If an existing old implementation uses:

```text
mediapipe.solutions
```

migrate it where practical.

Do not blindly downgrade MediaPipe just to make old code work.

Do not blindly reinstall major Kaggle packages.

Keep detector implementation behind the `FaceDetector` abstraction so the backend can be replaced later.

---

# 19. FACE DETECTION CACHING

Face detection is expensive.

Cache detection results.

For example:

```text
/kaggle/working/face_metadata.json
```

Cache:

* image ID
* image path
* detector version/config
* image hash or modification information
* detected face IDs
* bboxes
* confidence scores

Validate cache freshness before using it.

If detector configuration changes, invalidate/rebuild the cache.

---

# 20. KAGGLE NOTEBOOK REQUIREMENTS

The entire project must run in a Kaggle Notebook.

Do NOT require:

* local repository setup
* manually cloning a private local project
* local filesystem paths
* manual desktop execution
* hidden scripts that only exist outside the notebook

Use:

```text
/kaggle/input/
```

for read-only datasets.

Use:

```text
/kaggle/working/
```

for generated artifacts.

Recommended directories:

```text
/kaggle/working/
├── checkpoints/
├── predictions/
├── masks/
├── debug/
└── logs/
```

Create them automatically.

---

# 21. NOTEBOOK CELL ORGANIZATION

Organize the notebook into clear sequential sections/cells.

Recommended structure:

### Cell 1 — Environment

* Python version
* PyTorch version
* torchvision version
* OpenCV version
* MediaPipe version
* CUDA availability
* GPU name
* VRAM

### Cell 2 — Imports

All required imports.

### Cell 3 — Dataset Discovery

Inspect `/kaggle/input`.

### Cell 4 — Dataset Structure

Identify images and annotations.

### Cell 5 — Central Configuration

Create a single configuration dictionary.

### Cell 6 — Reproducibility

Set:

* Python seed
* NumPy seed
* PyTorch seed
* CUDA seed
* deterministic settings where practical

### Cell 7 — Annotation Inspection

Load representative annotations and display schema.

### Cell 8 — Face Detector Initialization

Initialize the selected supported detector.

### Cell 9 — Multi-Face Detection

Detect all faces.

### Cell 10 — Face/Annotation Matching

Implement matching.

### Cell 11 — Polygon → Mask

Implement mask generation.

### Cell 12 — Coordinate Transformations

Implement crop/resize/padding transformations.

### Cell 13 — Dataset Class

Create face-level dataset.

### Cell 14 — Visualization

Visualize:

* original image
* face boxes
* face IDs
* labels
* polygons
* masks

### Cell 15 — Dataset Split

Implement the required:

```text
70% train
15% validation
15% test
```

protocol where applicable.

Avoid image/identity leakage.

If multiple face samples originate from one image, ensure they remain in the same split.

### Cell 16 — Model Definitions

Implement all model variants.

### Cell 17 — Losses

Classification and segmentation losses.

### Cell 18 — DataLoaders

Configure efficient loading.

### Cell 19 — Training

Generic training loop.

### Cell 20 — Validation

Validation loop.

### Cell 21 — Metrics

Calculate required metrics.

### Cell 22 — Checkpointing

Save checkpoints.

### Cell 23 — Automated Tests

Run unit/integration tests.

### Cell 24 — Multi-Face Inference

Implement `predict_multi_face`.

### Cell 25 — Full-Image Visualization

Project face-level predictions back onto the original image.

### Cell 26 — Error Analysis

Inspect false positives/false negatives and segmentation errors.

### Cell 27 — Final Experiment Summary

Generate tables and saved results.

---

# 22. DATA SPLITTING

The thesis/SOP specifies a:

```text
70% training
15% validation
15% testing
```

protocol.

Preserve this unless the actual SOP/project configuration explicitly says otherwise.

Most importantly:

**split at the IMAGE level, not the FACE level.**

If:

```text
image_001
 ├── face_0
 ├── face_1
 └── face_2
```

belongs to training, all three face samples must remain in training.

Never allow faces from the same source image to leak across train/validation/test.

If identity information is available, inspect whether identity-level separation is also required and document the chosen strategy.

---

# 23. DATASET SAMPLE FORMAT

Each sample should expose information such as:

```python
{
    "image_id": ...,
    "face_id": ...,
    "image_path": ...,
    "bbox": ...,
    "image": ...,
    "label": ...,
    "mask": ...,
    "polygon": ...,
}
```

For classification-only experiments, the mask can be omitted from model input.

For segmentation experiments, the mask must be available.

---

# 24. MODEL IMPLEMENTATION

Build reusable components.

For example:

```python
class EfficientNetBackbone:
    ...

class MultiStreamExtractor:
    ...

class FeatureFusion:
    ...

class ClassificationHead:
    ...

class ASPP:
    ...

class DeepLabV3PlusHead:
    ...

class VERITASModel:
    ...
```

The model should be configured rather than duplicated four times.

For example:

```python
VERITASModel(
    backbone="efficientnet_b7",
    multi_stream=True,
    segmentation=True,
)
```

---

# 25. MULTI-STREAM IMPLEMENTATION

Inspect the existing VERITAS implementation/SOP carefully before implementing the streams.

The thesis describes complementary representations including:

* RGB
* noise residuals
* frequency-domain artifacts

and feature fusion before the classification/segmentation heads.

Preserve this conceptual design.

Do not invent arbitrary modalities without documenting them.

If the existing source code already implements a particular multi-stream extraction mechanism, preserve it unless it is demonstrably incorrect.

---

# 26. CLASSIFICATION HEAD

The classification branch should follow the intended architecture:

```text
shared feature map
      ↓
Global Average Pooling
      ↓
Fully Connected Layer
      ↓
Sigmoid / binary probability
```

Return:

```python
fake_probability
```

and derive:

```python
label = fake_probability >= threshold
```

The default threshold may be `0.5` if consistent with the existing implementation/SOP, but make it configurable.

---

# 27. SEGMENTATION HEAD

The segmentation branch must use:

```text
ASPP
+
DeepLabV3+ style decoder
```

to produce a spatial manipulation probability map.

The architecture should output:

```python
segmentation_logits
```

and probabilities can be obtained with:

```python
torch.sigmoid(...)
```

Use an appropriate segmentation loss.

For example:

```text
BCEWithLogitsLoss
```

possibly combined with:

```text
Dice loss
```

if justified by the existing implementation.

Do not arbitrarily change the loss without recording the decision.

---

# 28. MULTI-TASK LOSS

For MTL:

```text
L_total = λ_cls L_cls + λ_seg L_seg
```

Implement this explicitly.

Log per epoch:

```text
classification loss
segmentation loss
total loss
λ_cls
λ_seg
```

This is required for the loss-ratio experiment.

---

# 29. TRAINING LOOP

Create one generic training loop capable of training all configurations.

Example:

```python
train_experiment(
    experiment_config,
    train_loader,
    val_loader,
    config
)
```

The loop must automatically know whether the model uses:

```text
classification only
classification + segmentation
single stream
multi stream
```

Do not duplicate four separate training implementations unnecessarily.

---

# 30. CHECKPOINTING

Save checkpoints to:

```text
/kaggle/working/checkpoints/
```

Each checkpoint should contain:

```python
{
    "epoch": ...,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "best_metric": ...,
    "experiment_config": ...,
    "training_config": ...,
    "loss_config": ...,
}
```

Use descriptive names such as:

```text
baseline_effb7_best.pt
multistream_no_seg_best.pt
seg_no_multistream_best.pt
full_veritas_best.pt
```

---

# 31. GPU / KAGGLE RESOURCE MANAGEMENT

Detect GPU automatically.

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Print:

```text
device
GPU
VRAM
```

EfficientNet-B7 + multi-stream + ASPP can consume substantial memory.

Therefore provide configurable:

```python
BATCH_SIZE
IMAGE_SIZE
NUM_WORKERS
USE_AMP
GRADIENT_ACCUMULATION
```

where appropriate.

Use mixed precision when compatible.

Use:

```python
torch.no_grad()
```

during validation/inference.

Avoid unnecessarily retaining tensors on GPU.

Do not store entire prediction datasets on the GPU.

If memory errors occur, gracefully reduce batch size or use accumulation.

---

# 32. REQUIRED CLASSIFICATION METRICS

At minimum calculate:

* Accuracy
* F1-score

Also report, where practical:

* Precision
* Recall
* Confusion matrix
* TP
* TN
* FP
* FN

For every experiment.

Do not compare metrics across experiments if they were calculated on different test populations.

All variants should use the same test split.

---

# 33. REQUIRED SEGMENTATION METRICS

For segmentation-enabled experiments calculate:

* IoU
* mIoU
* Dice coefficient

Report them clearly.

Be explicit about whether metrics are:

* per-face
* macro averaged
* micro averaged

Prefer a consistent face-level evaluation protocol.

Real faces with all-zero masks need a clearly documented policy for segmentation metrics.

Do not allow large numbers of zero-mask real faces to make segmentation appear artificially strong without explaining the metric calculation.

---

# 34. INFERENCE LATENCY AND VRAM

The SOP identifies inference latency and peak VRAM as computational measurements.

Measure these for at least:

```text
Baseline EfficientNet-B7
Full VERITAS
```

and preferably all experimental variants.

Use a consistent measurement protocol.

Report:

```text
mean latency
median latency
standard deviation if useful
peak VRAM
batch size
input resolution
device
```

For GPU timing, use appropriate CUDA synchronization.

Do not mix CPU and GPU measurements.

---

# 35. EXPERIMENT CONTROL

All experiments must use as much as possible:

* same train split
* same validation split
* same test split
* same random seed
* same evaluation protocol
* same face detection metadata
* same preprocessing
* same classification threshold

unless a particular experiment explicitly changes one of these.

This is necessary to make the ablation comparisons meaningful.

---

# 36. EXPERIMENT RESULTS TABLE

Automatically generate a table similar to:

| Experiment               | Multi-Stream | Segmentation | Accuracy |  F1 | mIoU | Dice | Latency | Peak VRAM |
| ------------------------ | -----------: | -----------: | -------: | --: | ---: | ---: | ------: | --------: |
| Baseline EfficientNet-B7 |           No |           No |      ... | ... |  N/A |  N/A |     ... |       ... |
| Multi-stream             |          Yes |           No |      ... | ... |  N/A |  N/A |     ... |       ... |
| Segmentation             |           No |          Yes |      ... | ... |  ... |  ... |     ... |       ... |
| Full VERITAS             |          Yes |          Yes |      ... | ... |  ... |  ... |     ... |       ... |

Do not fabricate values.

If an experiment has not been run, display:

```text
N/A
```

or:

```text
NOT RUN
```

---

# 37. STATISTICAL ANALYSIS SUPPORT

The SOP describes statistical comparisons including:

* McNemar's Test for classification comparisons
* Wilcoxon Signed-Rank Test for paired comparisons

The implementation should therefore save per-sample/per-face predictions in a form that makes these analyses possible.

For every test sample preserve:

```text
image_id
face_id
ground_truth
prediction
probability
```

For segmentation:

```text
image_id
face_id
ground_truth_mask
predicted_mask
IoU
Dice
```

Do not perform statistical testing using only aggregate accuracy values.

---

# 38. PREDICTION EXPORT

Save prediction records to:

```text
/kaggle/working/predictions/
```

Formats:

```text
CSV
JSON
```

Include at minimum:

```text
image_id
face_id
bbox
ground_truth_label
predicted_label
fake_probability
```

For segmentation-enabled experiments also include:

```text
mask_path
ground_truth_mask_path
```

where practical.

---

# 39. FULL-IMAGE MULTI-FACE INFERENCE

Create:

```python
predict_multi_face(image_path)
```

The function must:

1. Load the image.
2. Detect ALL faces.
3. Assign stable face IDs.
4. Crop each face.
5. Apply preprocessing.
6. Run the selected VERITAS model.
7. Produce an independent classification result for every face.
8. Produce segmentation only where supported.
9. Project each face's predicted mask back to original-image coordinates.
10. Return structured results.

Example:

```python
{
    "image_id": "...",
    "faces": [
        {
            "face_id": 0,
            "bbox": [...],
            "label": "real",
            "fake_probability": 0.12,
            "mask": None
        },
        {
            "face_id": 1,
            "bbox": [...],
            "label": "fake",
            "fake_probability": 0.94,
            "mask": ...
        }
    ]
}
```

---

# 40. FULL-IMAGE VISUALIZATION

Create an annotated visualization that shows:

* original image
* bounding boxes
* face IDs
* predicted real/fake labels
* fake probabilities
* predicted manipulation masks
* optionally ground-truth polygons/masks

Example:

```text
Face 0 — REAL — 0.08
Face 1 — FAKE — 0.94
Face 2 — REAL — 0.13
```

Only overlay a predicted segmentation mask for faces where the segmentation model produces a meaningful prediction.

---

# 41. DEBUG VISUALIZATION

Implement:

```python
debug_image(image_path)
```

It should display:

### Panel/outputs showing:

1. Original image
2. All detected face boxes
3. Face IDs
4. Ground-truth face labels
5. Ground-truth polygons
6. Generated ground-truth masks
7. Predicted labels
8. Predicted masks

This is especially important for validating coordinate transformations.

Save debug images to:

```text
/kaggle/working/debug/
```

---

# 42. UNIT TESTS

The notebook must contain executable tests.

At minimum test:

### Test 1

One real face.

Expected:

```text
1 face
label = real
mask = zeros
```

### Test 2

One fake face.

Expected:

```text
1 face
label = fake
mask contains polygon
```

### Test 3

Three real faces.

Expected:

```text
3 faces
all real
all masks zero
```

### Test 4

Mixed:

```text
real
fake
real
```

Expected:

```text
real → zero mask
fake → polygon mask
real → zero mask
```

### Test 5

Multiple fake faces.

Each face must receive its own polygon/mask.

### Test 6

Detector order differs from annotation order.

Verify correct geometric matching.

### Test 7

No faces.

Verify graceful handling.

### Test 8

Polygon transformation.

Verify that a known polygon lands in the correct face-crop location after resizing.

### Test 9

Real face containing another face's fake annotation.

Verify that annotations are not accidentally assigned to the wrong face.

---

# 43. EXISTING SINGLE-FACE FUNCTIONALITY

Do NOT break existing single-face functionality.

A one-face image should simply become:

```text
image_id
 └── face_id = 0
```

The old workflow should remain a valid special case of the new multi-face architecture.

Avoid maintaining two separate code paths unless absolutely necessary.

---

# 44. ARCHITECTURAL MODULARITY

Keep these components independent:

```text
Dataset discovery
      ↓
Annotation parser
      ↓
Face detector
      ↓
Face/annotation matcher
      ↓
Coordinate transformer
      ↓
Polygon rasterizer
      ↓
Face-level dataset
      ↓
Model
      ↓
Loss
      ↓
Training
      ↓
Evaluation
      ↓
Inference
      ↓
Visualization
```

This allows components to be tested independently.

---

# 45. DO NOT HIDE DATASET OR IMPLEMENTATION ASSUMPTIONS

Whenever the implementation encounters uncertainty, inspect the actual data.

Do NOT write:

```python
POLY_PATH = "/kaggle/input/something/poly.json"
```

until the actual path has been discovered.

Do NOT assume:

* annotation key names
* label encoding
* polygon format
* bbox format
* image dimensions
* face ordering
* number of faces
* train/test file structure

Inspect first.

---

# 46. PACKAGE MANAGEMENT

At the beginning of the notebook:

1. Print installed package versions.
2. Check whether required packages already exist.
3. Only install missing packages.
4. Avoid unnecessary upgrades.
5. Avoid downgrading Kaggle's preinstalled packages unless absolutely necessary.
6. If a dependency conflict occurs, diagnose it rather than repeatedly reinstalling packages.

Particular attention must be paid to:

```text
torch
torchvision
opencv
mediapipe
numpy
PIL
scikit-learn
```

---

# 47. MEDIA PIPE MIGRATION POLICY

If the original project uses:

```python
mediapipe.solutions.face_mesh
```

do not blindly preserve it.

Instead:

1. Inspect installed MediaPipe.
2. Determine supported APIs.
3. Prefer MediaPipe Tasks or another supported face detection implementation.
4. Wrap the detector behind `FaceDetector`.
5. Maintain the same bbox/output contract.
6. Document any detector change.
7. Do not make the entire model dependent on deprecated MediaPipe APIs.

---

# 48. REPRODUCIBILITY

Create one configuration object, for example:

```python
CONFIG = {
    "seed": 42,
    "image_size": 600,
    "batch_size": 4,
    "num_workers": 2,
    "learning_rate": ...,
    "epochs": ...,
    "classification_threshold": 0.5,
    "lambda_cls": ...,
    "lambda_seg": ...,
    "backbone": "efficientnet_b7",
    "use_pretrained": True,
}
```

Adapt values to the actual SOP/codebase.

Every experiment must save its configuration alongside its checkpoint and results.

---

# 49. RERUNNABILITY

The entire notebook must be rerunnable after a Kaggle runtime restart.

It must not depend on hidden Python state.

Cells should execute sequentially from top to bottom.

If cached metadata exists:

```text
/kaggle/working/face_metadata.json
```

reuse it only after validating compatibility.

If a checkpoint exists, provide an option to resume.

---

# 50. FINAL NOTEBOOK OUTPUTS

At the end, the notebook should produce:

```text
/kaggle/working/
├── checkpoints/
│   ├── baseline_effb7_best.pt
│   ├── multistream_no_seg_best.pt
│   ├── seg_no_multistream_best.pt
│   └── full_veritas_best.pt
│
├── predictions/
│   ├── baseline_predictions.csv
│   ├── multistream_predictions.csv
│   ├── segmentation_predictions.csv
│   └── full_veritas_predictions.csv
│
├── masks/
│   └── ...
│
├── debug/
│   └── ...
│
├── logs/
│   └── ...
│
└── experiment_results.csv
```

Adapt filenames as needed.

---

# 51. FINAL VALIDATION CHECKLIST

Before considering the implementation complete, verify every item:

## Dataset

* [ ] Actual Kaggle dataset structure inspected.
* [ ] `poly.json` inspected.
* [ ] Annotation schema verified.
* [ ] Image schema verified.
* [ ] Labels verified.
* [ ] Polygon coordinates verified.
* [ ] Multiple-face cases verified.

## Face Processing

* [ ] All faces detected.
* [ ] Stable face IDs generated.
* [ ] Face/annotation matching implemented.
* [ ] Detector ordering is NOT assumed to equal annotation ordering.
* [ ] Single-face case still works.
* [ ] Multi-face case works.
* [ ] Mixed real/fake case works.
* [ ] No-face case works.

## Masks

* [ ] Fake faces use polygon masks.
* [ ] Real faces use zero masks.
* [ ] Polygon coordinates transformed correctly.
* [ ] Crop offset accounted for.
* [ ] Resize accounted for.
* [ ] Padding accounted for.
* [ ] No image-level contamination between faces.

## Architecture

* [ ] EfficientNet-B7 baseline implemented.
* [ ] Multi-stream classification model implemented.
* [ ] Segmentation-only ablation implemented.
* [ ] Full VERITAS implemented.
* [ ] DeepLabV3+ implemented.
* [ ] ASPP implemented.
* [ ] MTL loss implemented.
* [ ] Loss ratios configurable.

## Experiments

* [ ] Same test split.
* [ ] Same evaluation protocol.
* [ ] Baseline evaluated.
* [ ] Multi-stream evaluated.
* [ ] Segmentation evaluated.
* [ ] Full VERITAS evaluated.
* [ ] Loss-ratio experiments supported.
* [ ] Results saved.

## Metrics

* [ ] Accuracy.
* [ ] F1-score.
* [ ] Precision.
* [ ] Recall.
* [ ] Confusion matrix.
* [ ] IoU.
* [ ] mIoU.
* [ ] Dice.
* [ ] Inference latency.
* [ ] Peak VRAM.

## Kaggle

* [ ] `/kaggle/input` treated as read-only.
* [ ] `/kaggle/working` used for generated files.
* [ ] GPU detected automatically.
* [ ] CPU fallback available.
* [ ] Memory usage controlled.
* [ ] Notebook reruns from top to bottom.
* [ ] Package versions documented.
* [ ] No unnecessary package downgrades.

## MediaPipe

* [ ] Installed version checked.
* [ ] No new deprecated `mediapipe.solutions.*` implementation.
* [ ] Current supported API preferred.
* [ ] Detector isolated behind abstraction.

---

# 52. MOST IMPORTANT IMPLEMENTATION PRINCIPLE

Do not optimize for merely getting a model to train.

The purpose of this implementation is to produce a **research-valid, reproducible VERITAS experimental framework**.

The final notebook must allow us to answer the research questions by independently measuring:

```text
EfficientNet-B7 baseline
        ↓
Effect of multi-stream extraction
        ↓
Effect of segmentation / MTL
        ↓
Full VERITAS
        ↓
Effect of loss-weight calibration
        ↓
Computational trade-offs
```

while simultaneously correcting the original system's single-face limitation.

The final system must therefore preserve the research architecture described in the SOP while extending it to correctly handle:

```text
ONE IMAGE
   ↓
MULTIPLE FACES
   ↓
INDEPENDENT FACE-LEVEL LABELS
   ↓
INDEPENDENT FACE-LEVEL MASKS
   ↓
INDEPENDENT FACE-LEVEL PREDICTIONS
   ↓
RECONSTRUCTED FULL-IMAGE FORENSIC VISUALIZATION
```

Do not collapse multiple faces into one image-level label.

Do not let one fake face make all faces fake.

Do not assign another face's polygon to the wrong face.

Do not assume detector ordering equals annotation ordering.

Do not use deprecated `mediapipe.solutions` APIs for new implementation.

Do not skip the baseline/ablation experiments.

Do not fabricate missing annotation information.

Do not fabricate experimental results.

Inspect the actual dataset and existing implementation first, then implement the complete system in the Kaggle Notebook.
