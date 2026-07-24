# VERITAS Implementation Agent Instructions

## 1. Project Context

This repository implements the VERITAS thesis project.

Before making any implementation decision, read:

```text
research-context.md
```

This file contains the complete thesis implementation context and is the primary source of truth for the research specification.

The implementation must follow the confirmed decisions in `research-context.md` unless a change is explicitly justified as a research-design modification.

Do not silently invent architectural, preprocessing, dataset, loss, evaluation, or experimental decisions that contradict the research context.

---

## 2. Core Objective

Implement VERITAS as a multi-task facial image-forensics system that performs:

1. Binary facial manipulation classification.
2. Pixel-level localization of manipulated regions.
3. Spatially interpretable forensic evidence through predicted manipulation masks.

The core pipeline is:

```text
Input Image
    ↓
Face Detection / Validation
    ↓
Face Crop
    ↓
600×600 Preprocessing
    ↓
RGB Representation
Noise Residual Representation
Frequency-Domain Representation
    ↓
Feature Extraction and Fusion
    ↓
EfficientNet-B7 Backbone
    ├── Classification Head
    └── DeepLabV3+ / ASPP Segmentation Head
```

The system is designed for facial manipulation detection and localization.

---

## 3. Confirmed Research Constraints

The following decisions are fixed unless the user explicitly authorizes a research-design change:

* Primary dataset: OpenForensics.
* Input resolution: 600×600.
* Primary backbone: EfficientNet-B7.
* Learning paradigm: multi-task learning.
* Tasks:

  * image-level binary manipulation classification;
  * pixel-level manipulation segmentation.
* Segmentation architecture: DeepLabV3+.
* Context module: ASPP.
* ASPP dilation rates: 1, 6, 12, 18.
* ASPP includes a global pooling branch.
* Classification head:

  * global average pooling;
  * fully connected layer;
  * sigmoid output.
* Classification threshold:

  * probability >= 0.5 → manipulated;
  * probability < 0.5 → authentic.
* Segmentation output:

  * pixel-level sigmoid probability map;
  * threshold 0.5 for binary manipulation mask.
* Segmentation loss:

```text
Lseg = 0.5 × BCE + 0.5 × DiceLoss
```

* Total multi-task loss:

```text
Ltotal = 0.4 × Lclassification + 0.6 × Lsegmentation
```

* Optimizer: AdamW.
* Learning rate: 1e-4.
* Dataset split:

  * 70% training;
  * 15% validation;
  * 15% testing.
* The test set must remain isolated from:

  * training;
  * weight updates;
  * hyperparameter tuning;
  * architecture selection.
* Intended manipulation categories:

  * authentic;
  * face swapping;
  * face reenactment;
  * generative inpainting.
* Primary environment: Google Colab.
* Intended inference batch size: 1.

---

## 4. Research Integrity Rules

Do not:

* use the test set during training or model selection;
* leak identities or duplicate images across dataset splits;
* resize segmentation masks with bilinear or bicubic interpolation;
* silently change the 600×600 input resolution;
* replace EfficientNet-B7 without explicit justification;
* remove either the classification or segmentation task;
* silently alter the specified loss weights;
* silently change the primary evaluation metrics;
* claim a fixed GPU model unless it was actually recorded;
* fabricate dataset statistics;
* fabricate experimental results;
* claim that a component was evaluated if it was not actually run.

All experimental claims must be backed by actual recorded results.

---

## 5. Before Implementing the Full Model

First inspect the repository and determine:

* current directory structure;
* existing source files;
* existing notebooks;
* existing dataset utilities;
* existing configuration files;
* existing tests;
* existing model implementations;
* available dependency configuration.

Do not overwrite existing working code without first understanding its purpose.

Create or maintain a clear modular structure.

Prefer separation of concerns such as:

```text
src/
├── data/
│   ├── dataset.py
│   ├── annotations.py
│   ├── masks.py
│   └── splits.py
├── preprocessing/
│   ├── face_detection.py
│   ├── cropping.py
│   ├── transforms.py
│   └── representations.py
├── models/
│   ├── efficientnet_backbone.py
│   ├── fusion.py
│   ├── classification_head.py
│   ├── aspp.py
│   ├── deeplab_decoder.py
│   └── veritas.py
├── losses/
│   └── multitask_loss.py
├── training/
│   ├── trainer.py
│   ├── validation.py
│   └── checkpoints.py
├── evaluation/
│   ├── classification.py
│   ├── segmentation.py
│   ├── computational.py
│   └── statistics.py
└── utils/
    ├── reproducibility.py
    ├── logging.py
    └── runtime.py
```

Adapt this structure to the existing repository instead of blindly imposing it.

---

## 6. Handle Unresolved Research Decisions Explicitly

The thesis context contains unresolved decisions.

Do not hide these decisions inside arbitrary implementation choices.

The following must be explicitly documented in configuration or implementation documentation:

### 6.1 Multi-Stream Architecture

The system must use:

* RGB;
* noise residual;
* frequency-domain representation.

Before implementing the final architecture, determine and document:

* whether each representation has a separate encoder;
* whether streams share early layers;
* the fusion point;
* the fusion operator;
* channel dimensions;
* whether fusion uses concatenation, addition, attention, or learned weighting.

If the repository does not already specify the final choice, select a technically defensible design and document the rationale.

Do not describe an implementation as "adaptive fusion" unless an actual adaptive mechanism is implemented.

---

### 6.2 Noise Residual Representation

The exact residual-generation method must be explicit and reproducible.

Document:

* the filter or transform;
* kernel size;
* output tensor shape;
* normalization;
* numerical range.

Do not use an unspecified or undocumented "noise residual" operation.

---

### 6.3 Frequency-Domain Representation

The exact transform must be explicit.

Examples include:

* FFT;
* DCT;
* wavelet transform.

The implementation must specify:

* transform type;
* output representation;
* channel construction;
* normalization;
* tensor shape.

---

### 6.4 Face Cropping

The final face-cropping method must be deterministic and documented.

The implementation must ensure that image and segmentation mask coordinates remain aligned.

If face detection fails, return:

```text
No Face Detected
```

and stop processing that sample for inference.

---

### 6.5 Augmentation

Augmentations must be applied consistently to:

* the image;
* the corresponding segmentation mask.

Geometric transformations must preserve image/mask alignment.

Mask interpolation must preserve discrete binary labels.

The augmentation configuration must be recorded.

---

## 7. Dataset Pipeline

Implement dataset validation before model training.

Each usable sample should contain:

```text
image.jpg
annotation.json
```

Validate:

* image exists;
* annotation exists;
* authenticity label exists;
* bounding-box information is valid where required;
* polygon annotation is valid where required;
* polygon coordinates are valid;
* image dimensions are valid;
* generated masks align with the image;
* supported manipulation categories are retained.

Convert polygon annotations into binary masks:

```text
0 = authentic/background
1 = manipulated region
```

Authentic images should have all-zero manipulation masks.

Perform leakage checks before training.

Where possible, check for:

* duplicate images;
* near-duplicate images;
* identity overlap;
* source overlap between splits.

Persist the final split membership so experiments can be reproduced.

---

## 8. Preprocessing Pipeline

The intended sequence is:

```text
Validate Image/Annotation Pair
    ↓
Detect Face
    ↓
Reject Invalid Face
    ↓
Determine Face Crop
    ↓
Crop Image and Mask Consistently
    ↓
Transform to 600×600
    ↓
Normalize Image
    ↓
Apply Training Augmentation
```

Use ImageNet normalization where appropriate for ImageNet-pretrained EfficientNet-B7.

Use nearest-neighbor interpolation for binary segmentation masks.

Ensure all transformations are deterministic and reproducible when a random seed is provided.

---

## 9. Model Architecture

Implement the final architecture as an explicit PyTorch module.

The model must expose:

```python
classification_probability
segmentation_probability_map
```

The forward pass should conceptually implement:

```text
Representations
    ↓
Feature Extraction
    ↓
Feature Fusion
    ↓
EfficientNet-B7 Feature Representation
    ├── Classification Head
    └── DeepLabV3+ / ASPP Segmentation Head
```

The classification head must output one manipulation probability.

The segmentation head must output a full-resolution or correctly upsampled 600×600 probability map.

The model must not apply irreversible thresholding during the forward pass.

Return probabilities/logits and apply thresholding during inference or evaluation.

---

## 10. ASPP / DeepLabV3+ Requirements

The segmentation pathway must include ASPP with:

```text
Dilation Rate 1
Dilation Rate 6
Dilation Rate 12
Dilation Rate 18
Global Pooling Branch
```

The segmentation pathway must include:

* multi-scale contextual feature extraction;
* decoder processing;
* feature refinement;
* appropriate upsampling;
* final 600×600 segmentation output.

Verify tensor dimensions at every stage.

Add shape tests for:

* backbone output;
* ASPP output;
* decoder output;
* final segmentation output.

---

## 11. Loss Implementation

Implement:

```text
Lseg = 0.5 × BCE + 0.5 × DiceLoss
```

and:

```text
Ltotal = 0.4 × Lclassification
       + 0.6 × Lsegmentation
```

Use numerically stable implementations.

Prefer logits internally and apply sigmoid only when probabilities are required.

Ensure the classification target shape and segmentation target shape are correct.

The loss implementation must be unit-tested independently.

---

## 12. Baseline

Implement a standalone EfficientNet-B7 classification baseline.

The baseline must use a preprocessing and evaluation protocol that is as comparable as reasonably possible to VERITAS.

The thesis references an 84.4% EfficientNet-B7 classification result.

Do not claim to reproduce 84.4% unless the exact experimental setup and result are actually reproduced.

Document:

* baseline architecture;
* preprocessing;
* training configuration;
* dataset split;
* random seed;
* evaluation results.

---

## 13. Training

Training must follow:

```text
Training Set
    ↓
Weight Updates
```

```text
Validation Set
    ↓
Monitoring
Hyperparameter Tuning
Model Selection
```

```text
Test Set
    ↓
Final Evaluation Only
```

Use:

```text
Optimizer: AdamW
Learning Rate: 1e-4
```

Record:

* random seeds;
* configuration;
* epoch count;
* batch size;
* learning-rate schedule, if any;
* checkpoint selection rule;
* runtime environment;
* GPU model;
* VRAM;
* Python version;
* PyTorch version;
* relevant library versions.

Do not assume that the Google Colab GPU is always a Tesla T4.

---

## 14. Evaluation

Report classification metrics:

* Accuracy;
* Precision;
* Recall;
* F1-score.

Report segmentation metrics:

* mIoU;
* Dice Coefficient.

Report computational metrics:

* inference latency;
* peak VRAM usage.

Latency measurement must define:

* warm-up runs;
* measured runs;
* whether preprocessing is included;
* whether post-processing is included.

Peak VRAM measurement must be performed using an explicit, reproducible protocol.

All comparisons must state the hardware and software environment.

---

## 15. Statistical Testing

The planned significance level is:

```text
α = 0.05
```

Potential methods include:

* McNemar's Test;
* Wilcoxon Signed-Rank Test;
* permutation testing.

Before implementing statistical tests, define the paired unit.

Do not apply McNemar's test directly to aggregate F1 scores.

McNemar's test requires paired binary correctness outcomes.

Statistical tests must match the actual experimental design and available repeated observations.

---

## 16. Ablation Studies

At minimum, support the following experiments:

### Ablation A

Classification-only EfficientNet-B7 versus joint classification + segmentation.

### Ablation B

RGB-only representation versus RGB + noise residual + frequency-domain representations.

### Ablation C

Alternative loss-weight configurations, if included in the research design.

Each ablation must change only the intended factor whenever possible.

Record all configuration differences.

---

## 17. Explainability

The system should provide:

* predicted manipulation masks;
* visual overlays;
* guid-Xmage pixel-contribution maps if guid-Xmage is retained.

Do not claim quantitative explainability performance unless a formal evaluation protocol has been implemented.

Segmentation masks may serve as spatial forensic evidence, but explainability claims must be clearly distinguished from formal attribution-faithfulness claims.

---

## 18. Reproducibility

Every experiment must record enough information to reproduce it.

At minimum:

```text
Dataset version
Dataset split
Preprocessing configuration
Augmentation configuration
Architecture configuration
Representation transforms
Fusion method
Loss weights
Optimizer
Learning rate
Random seeds
Epochs
Batch size
Runtime type
GPU model
VRAM
Python version
PyTorch version
Library versions
```

Use configuration files rather than scattering experimental constants throughout source code.

---

## 19. Engineering Standards

Write production-quality research code.

Requirements:

* clear type hints where practical;
* meaningful names;
* modular functions;
* docstrings for public components;
* explicit tensor shapes where useful;
* validation of configuration values;
* deterministic seed handling;
* informative error messages;
* no silent data corruption;
* no silent fallback to incompatible behavior.

Prefer fail-fast behavior for:

* malformed annotations;
* incompatible image/mask dimensions;
* missing required fields;
* invalid tensor shapes;
* unsupported categories;
* missing model weights.

---

## 20. Testing Requirements

Before declaring an implementation complete, test:

### Data

* annotation parsing;
* polygon-to-mask conversion;
* authentic all-zero masks;
* malformed annotation handling;
* image/mask alignment.

### Preprocessing

* face detection behavior;
* `"No Face Detected"` behavior;
* 600×600 output;
* image/mask geometric consistency;
* nearest-neighbor mask handling.

### Representations

* RGB tensor shape;
* noise residual tensor shape and range;
* frequency-domain tensor shape and range.

### Model

* forward-pass success;
* classification output shape;
* segmentation output shape;
* gradient flow;
* CPU compatibility where practical.

### Loss

* BCE component;
* Dice component;
* weighted segmentation loss;
* weighted total loss.

### Evaluation

* classification metrics;
* segmentation metrics;
* latency measurement;
* VRAM measurement.

---

## 21. Implementation Workflow

Follow this order unless the repository requires a justified variation:

1. Read `research-context.md`.
2. Inspect the repository.
3. Identify existing implementation and tests.
4. Freeze or document unresolved research decisions.
5. Implement dataset validation.
6. Implement polygon-to-mask conversion.
7. Implement split generation and leakage checks.
8. Implement face detection and cropping.
9. Implement 600×600 preprocessing.
10. Implement RGB, noise-residual, and frequency-domain representations.
11. Implement and test feature fusion.
12. Implement standalone EfficientNet-B7 baseline.
13. Implement the VERITAS multi-task architecture.
14. Implement ASPP and DeepLabV3+ decoding.
15. Implement the multi-task loss.
16. Implement training.
17. Implement evaluation.
18. Implement ablation support.
19. Implement reproducibility logging.
20. Run tests.
21. Run a small end-to-end smoke test.
22. Only then proceed to full training.

---

## 22. Decision-Making Rule

When the research context is ambiguous:

1. Identify the ambiguity explicitly.
2. Check the existing repository for an established decision.
3. Prefer the simplest technically defensible implementation consistent with the thesis.
4. Document the decision.
5. Do not silently change confirmed research decisions.
6. Do not claim that an unresolved research choice was part of the original thesis specification.

When a choice materially changes the research design, stop and ask the user before proceeding.

---

## 23. Completion Criteria

The implementation is not complete merely because the code runs.

A complete implementation must provide:

* validated dataset pipeline;
* reproducible train/validation/test split;
* leakage checks;
* correct image/mask alignment;
* 600×600 preprocessing;
* RGB representation;
* noise-residual representation;
* frequency-domain representation;
* documented feature fusion;
* EfficientNet-B7 backbone;
* classification head;
* DeepLabV3+/ASPP segmentation head;
* specified multi-task loss;
* AdamW optimization at 1e-4;
* baseline implementation;
* classification evaluation;
* segmentation evaluation;
* computational evaluation;
* ablation support;
* reproducibility metadata;
* automated tests;
* an end-to-end smoke test.

The implementation must be scientifically defensible, reproducible, and faithful to the research specification in `research-context.md`.
