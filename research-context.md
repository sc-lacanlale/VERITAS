# VERITAS THESIS — IMPLEMENTATION-READY CONTEXT

## 1. Thesis Overview

### Working Title

**VERITAS: A Vision-based EfficientNet-Reinforced Image Tampering Authentication and ASPP-Segmentation MTL-Framework**

The manuscript also uses the phrase **“Visual Evidence Reliability through Integrated Tampering Authentication and Segmentation”** as an alternate expansion of VERITAS. This naming inconsistency should be resolved before final thesis submission. Unless otherwise specified, use the formal working title above.

### Research Domain and Context

The thesis belongs to the fields of:

* Digital image forensics
* Deepfake and AI-generated facial manipulation detection
* Semantic segmentation
* Explainable artificial intelligence
* Multi-task learning
* Deep convolutional neural networks
* Efficient deep-learning architectures

The research addresses the increasing realism and accessibility of AI-generated facial manipulations, including face swapping, face reenactment, and generative inpainting.

### Core Problem

Many existing image-forensics systems provide only a global binary decision such as “real” or “fake.” They do not adequately identify the exact spatial regions responsible for the manipulation.

This creates several problems:

* A classification score alone provides limited forensic evidence.
* Black-box predictions are difficult to interpret.
* Users cannot easily verify which facial regions were manipulated.
* Classification and localization are often implemented as separate systems rather than jointly learned tasks.
* Sophisticated GAN- and diffusion-generated manipulations may evade conventional detection approaches.

### Central Research Objective

Develop and evaluate VERITAS as a multi-task deep-learning framework that jointly performs:

1. **Binary facial forgery authentication**, and
2. **Pixel-level localization of manipulated regions**.

The proposed system uses a shared EfficientNet-B7-based feature-extraction architecture and a DeepLabV3+/ASPP segmentation pathway.

---

# 2. Problem Statement and Rationale

## Specific Problem or Research Gap

The primary gap is the lack of a unified facial image-forensics framework that combines:

* high-capacity image-level manipulation classification;
* pixel-level localization of manipulated regions;
* spatially interpretable evidence;
* multi-representation forensic feature extraction; and
* evaluation of accuracy alongside computational cost.

The thesis identifies the 84.4% EfficientNet-B7 classification result as a relevant baseline and argues that classification alone does not provide sufficient spatial justification.

## Why the Problem Matters

The problem is relevant to:

* Digital forensics
* Journalism and newsroom media verification
* Cybersecurity
* Identity protection
* Legal and investigative contexts
* Explainable AI
* Digital literacy

A binary authenticity score may be insufficient when a user must determine which facial regions have been manipulated and whether the model's prediction is supported by visually meaningful evidence.

## Limitations of Existing Approaches

The manuscript identifies or implies the following limitations:

* Binary classification does not localize manipulation.
* Black-box predictions provide limited explanation.
* Single-domain feature extraction may miss artifacts present in other representations.
* Neural-rendered and diffusion-generated manipulations can be difficult to detect.
* High-capacity models can impose significant computational costs.
* Classification and segmentation are often treated as separate tasks.
* Classification-only systems do not directly provide pixel-level evidence.

## Motivation for the Proposed Solution

VERITAS is motivated by the hypothesis that jointly learning classification and segmentation can improve the quality of forensic representations.

The segmentation task is expected to provide spatial supervision that may act as a regularizer for the classification task. By learning both:

* whether an image is manipulated; and
* where the manipulation occurs,

the shared encoder may learn more spatially discriminative forensic features.

---

# 3. Research Objectives and Questions

## General Objective

Develop a multi-task learning framework for high-accuracy facial image authentication and pixel-level manipulation localization.

## Specific Objectives

### Objective 1 — Classification Performance

Evaluate classification performance using:

* Accuracy
* F1-score

Compare the proposed system with the 84.4% EfficientNet-B7 baseline.

Evaluate the contribution of:

* joint classification-segmentation learning;
* multi-scale or multi-representation stream extraction.

### Objective 2 — Localization Performance

Evaluate manipulation localization using:

* Mean Intersection over Union (mIoU)
* Dice Coefficient

### Objective 3 — Computational Cost

Compare VERITAS and standalone EfficientNet-B7 using:

* Inference latency
* Peak VRAM usage

### Objective 4 — Architectural and Training Optimization

Investigate:

* the effect of joint segmentation learning;
* the effect of multi-representation feature extraction;
* suitable hyperparameters;
* weighted loss ratios.

## Research Questions

1. Does joint classification-segmentation learning improve classification Accuracy and F1-score compared with classification-only learning?
2. What is the effect of multi-representation or multi-stream feature extraction compared with a single RGB stream?
3. How accurately can the segmentation pathway localize manipulated facial regions?
4. What loss weighting and hyperparameter configuration provide effective multi-task performance?
5. Does VERITAS exceed the 84.4% EfficientNet-B7 classification baseline?
6. What computational trade-offs in inference latency and VRAM usage result from the proposed multi-task architecture?

## Hypothesis Groups

The thesis includes hypothesis groups concerning:

1. VERITAS versus standalone EfficientNet-B7 classification performance.
2. Multi-representation feature extraction versus single-stream RGB.
3. DeepLabV3+/ASPP localization performance.
4. Weighted loss-ratio calibration.
5. VERITAS versus standalone EfficientNet-B7 computational performance.

Planned statistical methods include:

* McNemar's Test
* Wilcoxon Signed-Rank Test
* Permutation Testing

The significance level is:

**α = 0.05**

---

# 4. Proposed Solution / System

## System Overview

VERITAS is a multi-task deep-learning framework that performs both:

### Authentication

Determines whether a facial image is authentic or manipulated.

### Localization

Identifies the pixel-level regions believed to contain manipulation.

### Explainability

Provides spatial evidence through the predicted manipulation mask and the proposed guid-Xmage pixel-contribution mechanism.

## Intended Users and Stakeholders

The intended stakeholders include:

* Digital forensic investigators
* Journalists
* Newsroom verification teams
* Cybersecurity practitioners
* Identity-protection practitioners
* Researchers in digital forensics
* Researchers in explainable AI
* Digital-literacy initiatives

## Core System Workflow

```text
Input Image
    ↓
Face Detection
    ↓
Face Crop / Validation
    ↓
600×600 Preprocessing
    ↓
Multi-Representation Feature Extraction
    ├── RGB Representation
    ├── Noise Residual Representation
    └── Frequency-Domain Representation
    ↓
Feature Stream Processing and Fusion
    ↓
EfficientNet-B7 Shared Encoder
    ├── Classification Head
    │     ↓
    │   Manipulation Probability
    │
    └── DeepLabV3+ / ASPP Segmentation Head
          ↓
       Pixel-Level Manipulation Probability Map
          ↓
       Binary Manipulation Mask
```

## Authentication Output

The classification head produces a manipulation probability in the range `[0, 1]`.

Decision rule:

```text
score >= 0.5 → manipulated/fake
score < 0.5  → authentic/real
```

## Localization Output

The segmentation head produces:

1. A per-pixel probability map.
2. A binary manipulation mask.

The binary mask is produced using a threshold of `0.5`.

## Explainability Output

The proposed explainability approach combines:

* the DeepLabV3+ segmentation mask;
* spatial visualization of manipulated regions;
* guid-Xmage pixel contribution analysis.

The intended result is a spatially verifiable explanation of the classification decision.

## Scope

The system is limited to:

* Facial images
* Face swapping
* Face reenactment
* Generative inpainting
* OpenForensics data
* Single-image processing

The system does not target:

* General object or landscape splicing
* General-purpose image manipulation outside the defined facial-manipulation scope
* High-level adversarial defense
* Batch processing
* Real-time video processing on mobile hardware

Images with insufficiently visible facial structure may be rejected.

---

# 5. System Requirements

## Functional Requirements

### FR-1 — Image Input

The system must accept a facial image for analysis.

### FR-2 — Face Detection

The system must detect a valid face or facial topology before deep-learning inference.

The intended technology is MediaPipe Face Mesh or an equivalent face-mesh-based detection process.

### FR-3 — Invalid Input Handling

If no valid face is detected, processing must stop and return:

```text
No Face Detected
```

### FR-4 — Face Extraction

The system must extract or crop the relevant facial region.

### FR-5 — Annotation Processing

During training and evaluation, polygon annotations must be converted into binary masks.

### FR-6 — Image Standardization

Images must be transformed into the model's required 600×600 input format.

### FR-7 — Mask Standardization

Masks must be transformed consistently with their corresponding images.

Mask interpolation must preserve binary/discrete labels.

### FR-8 — Multi-Representation Feature Extraction

The system must use or evaluate:

* RGB information;
* noise residual information;
* frequency-domain information.

### FR-9 — Feature Fusion

The representations must be combined into a unified feature representation before or within the shared feature-extraction architecture, according to the final architecture specification.

### FR-10 — Classification

The system must produce a manipulation probability and binary authenticity decision.

### FR-11 — Segmentation

The system must produce a pixel-level manipulation probability map and binary manipulation mask.

### FR-12 — Joint Training

The classification and segmentation tasks must be jointly optimized under the multi-task learning objective.

### FR-13 — Explainability

The system should provide spatial evidence through segmentation and, if retained in the final implementation, guid-Xmage attribution.

### FR-14 — Evaluation

The system must support evaluation of:

* classification;
* segmentation;
* computational performance;
* statistical significance.

---

## Non-Functional Requirements

### Accuracy

The system must be evaluated against the 84.4% EfficientNet-B7 baseline.

### Explainability

The system must provide spatial evidence rather than only a binary prediction.

### Generalization

Performance must be evaluated on a held-out test set.

### Reproducibility

The data split, preprocessing, model configuration, training configuration, and evaluation protocol must be documented.

### Computational Feasibility

Inference is intended for batch size one.

The implementation must record the computational environment used for each experiment.

---

## Technical and Operational Constraints

* Input resolution: 600×600.
* Single-image processing.
* Batch size one for intended inference.
* Facial manipulation domain only.
* OpenForensics is the primary dataset.
* The test set must remain isolated from training and hyperparameter tuning.
* The implementation and experimentation environment is Google Colab.
* GPU type and available VRAM may vary depending on the assigned Google Colab runtime.
* The exact Colab runtime hardware should be recorded for each experiment or standardized where possible.

---

## Assumptions

The thesis assumes that:

* OpenForensics provides suitable data for the target problem.
* Ground-truth labels and polygon masks are sufficiently reliable.
* ImageNet-pretrained EfficientNet-B7 is a valid transfer-learning starting point.
* Joint classification and segmentation learning can provide useful task synergy.
* The train, validation, and test sets are sufficiently independent.
* Image-level or identity-level leakage between splits should be prevented.

---

# 6. System Architecture and Components

## 6.1 Data Ingestion

Each usable sample is expected to consist of:

```text
image.jpg
annotation.json
```

The annotation file contains, according to the manuscript:

* binary authenticity label;
* bounding-box coordinates;
* polygon-based segmentation annotations.

Incomplete image/annotation pairs should be excluded.

---

## 6.2 Face Detection and Cropping

The system detects facial structure before inference.

The facial structure may include:

* eyes;
* nose;
* mouth;
* jawline;
* other facial landmarks.

The system must determine whether a valid facial region exists.

If no valid face is found:

```text
No Face Detected
```

Processing stops.

The exact final face-cropping method must be standardized because the manuscript contains both landmark-based and bounding-box-based descriptions.

---

## 6.3 Preprocessing

The intended preprocessing sequence is:

```text
1. Validate image and annotation pair.
2. Detect face.
3. Determine facial crop.
4. Crop image and corresponding mask consistently.
5. Resize or pad to 600×600.
6. Preserve mask discreteness using nearest-neighbor interpolation.
7. Normalize image tensors using ImageNet normalization.
8. Apply training augmentation.
```

The manuscript discusses the following augmentation types:

* rotation: ±2°;
* width shift: ±10%;
* height shift: ±10%;
* shear: 10%;
* zoom: 5%;
* horizontal flip;
* occlusion augmentation is also mentioned and must be clarified as part of the final augmentation policy.

---

## 6.4 Multi-Representation Feature Extraction

The proposed feature representations are:

### RGB Stream

Captures conventional spatial and semantic visual information.

### Noise Residual Stream

Intended to capture low-level inconsistencies and forensic artifacts.

The exact residual-generation algorithm must be specified before implementation.

### Frequency-Domain Stream

Intended to capture frequency-domain artifacts associated with image generation or manipulation.

The exact transform and representation must be specified before implementation.

---

## 6.5 Feature Fusion

The thesis refers to an adaptive or multi-stream feature-fusion process.

The final implementation must specify:

* number of streams;
* exact input representation of each stream;
* whether each stream has its own encoder;
* whether the streams share early or late layers;
* exact fusion point;
* exact fusion operator;
* whether fusion is concatenation, addition, attention, learned weighting, or another mechanism.

This is currently one of the most important unresolved architectural decisions.

---

## 6.6 EfficientNet-B7 Shared Encoder

EfficientNet-B7 is the main backbone.

Relevant architectural components include:

* compound scaling;
* MBConv blocks;
* Squeeze-and-Excitation modules.

The backbone provides shared feature representations to both task heads.

The exact relationship between the multi-stream feature extractors and the EfficientNet-B7 backbone must be finalized.

Possible interpretations in the manuscript include:

```text
Representation Streams
        ↓
Feature Fusion
        ↓
EfficientNet-B7
        ↓
Classification + Segmentation
```

or:

```text
Representation Streams
        ↓
Separate Feature Extraction Paths
        ↓
Feature Fusion
        ↓
Shared or Partially Shared Features
        ↓
Classification + Segmentation
```

The final implementation must select and document one architecture.

---

## 6.7 Classification Head

The intended classification pipeline is:

```text
Shared Feature Map
        ↓
Global Average Pooling
        ↓
Fully Connected Layer
        ↓
Sigmoid
        ↓
Manipulation Probability
```

Decision rule:

```text
p >= 0.5 → manipulated
p < 0.5  → authentic
```

---

## 6.8 Segmentation Head

The segmentation pathway uses DeepLabV3+ with ASPP.

ASPP uses the following dilation rates:

```text
1
6
12
18
```

It also includes a global average pooling branch.

The general process is:

```text
Shared Feature Map
        ↓
ASPP
        ↓
Multi-Scale Context Features
        ↓
Decoder
        ↓
Feature Refinement / Skip Connections
        ↓
Upsampling
        ↓
600×600 Pixel-Level Probability Map
        ↓
0.5 Threshold
        ↓
Binary Manipulation Mask
```

---

# 7. Data and Information Requirements

## Dataset

The primary dataset is OpenForensics.

The manuscript describes approximately 225,000 annotated images and several contextual subsets, including:

* Face Annotations;
* Diversity of Faces;
* Pose Variations;
* Occlusions;
* Background Complexity.

The final dataset version and exact sample counts must be recorded for reproducibility.

---

## Required Data Structure

Each sample should contain:

```text
Image
 ├── Authenticity Label
 ├── Bounding Box
 ├── Polygon Annotation
 └── Derived Binary Mask
```

---

## Image-Level Labels

```text
0 = authentic
1 = manipulated
```

---

## Pixel-Level Labels

```text
0 = authentic/background
1 = manipulated region
```

Authentic images should use an all-zero manipulation mask.

---

## Supported Manipulation Categories

The implementation should retain:

* authentic images;
* face swapping;
* face reenactment;
* generative inpainting.

Other categories should be excluded unless the thesis scope is explicitly expanded.

---

## Dataset Split

```text
70% Training
15% Validation
15% Testing
```

The test set must not be used for:

* training;
* weight updates;
* hyperparameter tuning;
* architecture selection.

---

## Data Integrity Checks

The implementation should verify:

* image file exists;
* annotation file exists;
* authenticity label exists;
* required bounding box exists;
* polygon annotation exists where required;
* polygon coordinates are valid;
* image and mask dimensions align;
* no data leakage exists between splits.

Where possible, image-level and identity-level overlap between train, validation, and test sets should be checked.

---

# 8. Methodology and Implementation Plan

## Research Methodology

The thesis uses an experimental quantitative research design.

The main experimental variables include:

* classification-only versus multi-task learning;
* single RGB representation versus multi-representation streams;
* different loss-weight configurations;
* computational performance of the proposed architecture versus the baseline.

---

## Development Environment

The system will be implemented and evaluated primarily using:

* Google Colab;
* Python;
* PyTorch;
* OpenCV;
* NumPy;
* scikit-learn;
* Jupyter Notebook-compatible workflows;
* MediaPipe Face Mesh;
* EfficientNet-B7;
* DeepLabV3+;
* ASPP;
* AdamW;
* guid-Xmage, if retained in the final implementation.

### Google Colab Constraint

The GPU assigned by Google Colab may vary.

Therefore, every experiment should record:

* Colab runtime type;
* GPU model;
* available VRAM;
* CUDA version where relevant;
* PyTorch version;
* Python version;
* relevant library versions.

The thesis should not claim a fixed Tesla T4 or other GPU model unless that exact hardware was actually used and documented.

---

## Recommended Implementation Phases

### Phase 1 — Dataset Verification

Implement:

* dataset acquisition;
* image/JSON validation;
* manipulation-category filtering;
* polygon-to-mask conversion;
* dataset statistics;
* train/validation/test split;
* leakage checks.

---

### Phase 2 — Preprocessing Pipeline

Implement:

* face detection;
* invalid-face hard-stop behavior;
* facial crop;
* image/mask alignment;
* 600×600 resizing or padding;
* image normalization;
* augmentation.

---

### Phase 3 — Baseline Model

Implement standalone EfficientNet-B7 classification.

The baseline should establish a controlled comparison point.

The baseline evaluation should use a preprocessing and evaluation protocol as comparable as possible to the proposed model.

---

### Phase 4 — Representation Streams

Implement:

* RGB stream;
* noise residual stream;
* frequency-domain stream.

The exact algorithms and tensor shapes must be documented.

---

### Phase 5 — Feature Fusion

Implement the final fusion mechanism.

The fusion design must be explicitly documented and reproducible.

---

### Phase 6 — Shared EfficientNet-B7 Encoder

Integrate the fused representation with the EfficientNet-B7 backbone according to the final architecture decision.

---

### Phase 7 — Classification Head

Implement:

```text
Global Average Pooling
        ↓
Fully Connected Layer
        ↓
Sigmoid
```

---

### Phase 8 — DeepLabV3+/ASPP Segmentation Head

Implement:

* ASPP;
* dilation rates 1, 6, 12, and 18;
* global pooling branch;
* decoder;
* feature refinement;
* final 600×600 output;
* sigmoid probability map;
* binary mask thresholding at 0.5.

---

### Phase 9 — Multi-Task Loss

The specified loss structure is:

```text
Lseg = 0.5 × LBCE + 0.5 × LDice

Ltotal = 0.4 × Lclassification
       + 0.6 × Lsegmentation
```

The classification loss is based on binary classification loss.

The segmentation loss combines:

* Binary Cross-Entropy;
* Dice Loss.

---

### Phase 10 — Optimization

Specified optimizer:

```text
AdamW
```

Specified learning rate:

```text
1e-4
```

Other training hyperparameters must be documented when selected.

---

### Phase 11 — Training

Use:

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

---

### Phase 12 — Ablation Studies

At minimum, investigate:

#### Ablation A

Classification-only versus joint classification + segmentation.

#### Ablation B

Single RGB stream versus multi-representation streams.

#### Ablation C

Alternative loss-weight configurations, if experimentally evaluated.

#### Ablation D

Other architecture components only if explicitly justified and included in the research design.

---

### Phase 13 — Explainability

Generate:

* predicted manipulation masks;
* visual overlays;
* guid-Xmage pixel-contribution maps if retained.

The explainability evaluation protocol must be defined before claiming quantitative explainability performance.

---

# 9. Evaluation and Validation

## Classification Metrics

Primary metrics:

* Accuracy;
* F1-score.

Supporting metrics:

* Precision;
* Recall.

F1-score is particularly important because accuracy alone can be misleading in the presence of class imbalance.

---

## Segmentation Metrics

Primary metrics:

* Mean Intersection over Union (mIoU);
* Dice Coefficient.

These compare predicted manipulation masks with ground-truth masks.

---

## Computational Metrics

Compare VERITAS with standalone EfficientNet-B7 using:

### Inference Latency

Measure the time required to process one image.

The measurement protocol must define:

* warm-up runs;
* number of measured runs;
* whether preprocessing time is included;
* whether post-processing time is included.

### Peak VRAM

Measure maximum GPU memory usage during inference.

Because the system runs in Google Colab, the exact GPU model and available VRAM must be recorded for each experiment.

---

## Explainability Evaluation

The intended explainability outputs are:

* DeepLabV3+ segmentation masks;
* guid-Xmage pixel contribution matrices.

A formal quantitative faithfulness metric or human-evaluation protocol is not fully specified and must be defined before implementation if explainability is treated as a formal evaluation objective.

---

## Statistical Testing

The manuscript proposes:

### McNemar's Test

For paired binary classification outcomes between systems.

### Wilcoxon Signed-Rank Test

For paired nonparametric comparisons.

### Permutation Testing

For evaluating the robustness of differences such as F1-score differences.

The significance level is:

```text
α = 0.05
```

The exact unit of pairing must be defined before implementation.

For example, the statistical protocol must clarify whether paired observations are:

* individual images;
* folds;
* repeated experimental runs;
* evaluation batches.

McNemar's Test should be applied to paired binary correctness outcomes rather than directly treating a single aggregate F1-score as its input.

---

# 10. Important Decisions and Constraints

## Confirmed Architectural Decisions

The following are core thesis decisions:

* EfficientNet-B7 is the main backbone.
* The system performs multi-task learning.
* The tasks are classification and segmentation.
* Input resolution is 600×600.
* DeepLabV3+ is used for segmentation.
* ASPP is used for multi-scale contextual feature extraction.
* ASPP dilation rates are 1, 6, 12, and 18.
* A global pooling branch is included in ASPP.
* Classification uses global average pooling and a fully connected layer.
* Classification uses sigmoid output.
* Segmentation produces pixel-level probability outputs.
* The binary decision threshold is 0.5.
* Segmentation loss combines BCE and Dice Loss.
* The specified segmentation-loss weighting is 0.5 BCE / 0.5 Dice.
* The specified total loss weighting is 0.4 classification / 0.6 segmentation.
* AdamW is the specified optimizer.
* The specified learning rate is 1e-4.
* OpenForensics is the primary dataset.
* The dataset split is 70% / 15% / 15%.
* The test set must remain isolated until final evaluation.
* Google Colab is the primary development and experimentation environment.
* GPU type and VRAM must be recorded for reproducibility.

---

## Explicit Constraints

The system is constrained to:

* facial manipulation detection;
* single-image processing;
* batch size one for intended inference;
* OpenForensics-based experimentation;
* no explicit adversarial-defense subsystem;
* no mobile real-time video requirement;
* valid or sufficiently visible facial structure.

---

## Technology Choices

Primary technology stack:

* Google Colab;
* Python;
* PyTorch;
* OpenCV;
* NumPy;
* scikit-learn;
* MediaPipe Face Mesh;
* EfficientNet-B7;
* DeepLabV3+;
* ASPP;
* AdamW;
* guid-Xmage, if retained.

---

## Rules That Should Not Be Changed Without Justification

The following changes would constitute research-design changes:

* changing the dataset;
* changing the dataset split;
* changing the input resolution;
* replacing EfficientNet-B7;
* removing the segmentation task;
* removing the classification task;
* changing the loss formulation;
* changing the loss weights;
* changing the baseline;
* changing the primary evaluation metrics;
* using the test set during training or hyperparameter tuning.

---

# 11. Implementation-Ready Knowledge Base

## Key Terminology

### VERITAS

The proposed multi-task facial image-forensics framework.

### EfficientNet-B7

The primary high-capacity feature-extraction backbone.

### MBConv

A mobile inverted bottleneck convolutional block used in EfficientNet.

### Squeeze-and-Excitation

A channel-attention mechanism used within EfficientNet blocks.

### Multi-Task Learning

Joint optimization of multiple related tasks, here:

* image-level manipulation classification;
* pixel-level manipulation segmentation.

### ASPP

Atrous Spatial Pyramid Pooling.

A multi-branch contextual feature-extraction module using multiple dilation rates.

### DeepLabV3+

The segmentation architecture used for manipulation localization.

### Ground-Truth Mask

A binary mask identifying manipulated pixels.

### Manipulation Probability

The image-level sigmoid output indicating the estimated probability that an image is manipulated.

### Segmentation Probability Map

A per-pixel sigmoid output indicating the estimated probability that each pixel belongs to a manipulated region.

### Spatial Explainability

The ability to identify image regions associated with a model's decision.

### guid-Xmage

A proposed symbolic-execution-based pixel-contribution explanation mechanism.

### OpenForensics

The primary dataset used for training, validation, and testing.

---

## Important Entities and Relationships

```text
Image
 ├── Authenticity Label
 ├── Bounding Box
 ├── Polygon Annotation
 └── Derived Binary Mask
```

```text
Image
   ↓
Face Detector
   ↓
Face Crop
   ↓
Preprocessing
   ↓
Representation Generation
   ├── RGB
   ├── Noise Residual
   └── Frequency Domain
   ↓
Feature Extraction
   ↓
Feature Fusion
   ↓
EfficientNet-B7
   ├── Classification Head
   │     ↓
   │   Manipulation Probability
   │
   └── Segmentation Head
         ↓
      Manipulation Probability Map
         ↓
      Binary Manipulation Mask
```

---

## Critical Inference Workflow

```text
Input JPG
    ↓
Face Detection
    ├── Failure → "No Face Detected"
    │
    └── Success
          ↓
       Face Crop
          ↓
       600×600 Preprocessing
          ↓
       Representation Generation
          ↓
       Feature Extraction/Fusion
          ↓
       EfficientNet-B7
          ├── Classification Probability
          │       ↓
          │   Threshold 0.5
          │       ↓
          │   Authentic/Manipulated
          │
          └── Segmentation Probability Map
                  ↓
              Threshold 0.5
                  ↓
              Binary Manipulation Mask
```

---

## Critical Training Workflow

```text
OpenForensics
    ↓
Validate Image/JSON Pairs
    ↓
Filter Supported Categories
    ↓
Decode Polygon Masks
    ↓
Create 70/15/15 Split
    ↓
Preprocess and Augment
    ↓
Forward Pass
    ├── Classification Prediction
    └── Segmentation Prediction
    ↓
Compute Classification Loss
    ↓
Compute Segmentation Loss
    ↓
Combine Losses
    ↓
Backpropagation
    ↓
AdamW Update
```

---

## Unresolved Issues Requiring Clarification

### 1. VERITAS Acronym Expansion

The manuscript contains more than one expansion.

The formal working title should be selected and used consistently.

### 2. Exact Multi-Stream Architecture

The thesis does not fully specify whether the architecture uses:

* three independent full backbone streams;
* partially shared streams;
* separate early feature extractors;
* concatenated representations entering a single EfficientNet-B7;
* another arrangement.

This must be resolved.

### 3. Exact Noise Residual Algorithm

The thesis specifies noise residual features but not the exact implementation.

The final system must specify the filter, transform, or residual-generation method.

### 4. Exact Frequency-Domain Transform

The final system must specify whether it uses:

* FFT;
* DCT;
* wavelet representation;
* another frequency representation.

The thesis currently does not provide enough detail.

### 5. Exact Fusion Mechanism

The final implementation must specify:

* fusion point;
* fusion operator;
* channel dimensions;
* adaptive weighting mechanism, if any.

### 6. Face-Cropping Method

The manuscript contains both:

* landmark-based face-mesh processing;
* bounding-box-based face extraction.

The final implementation must choose and document the definitive process.

### 7. Augmentation Policy

The manuscript mentions geometric augmentation and also occlusion augmentation.

The final augmentation set must be explicitly defined.

### 8. Occlusion Handling

The dataset includes occlusion scenarios, but the system also requires sufficiently visible facial structure.

The treatment of partially occluded samples must be clarified.

### 9. Statistical Pairing

The exact paired unit for Wilcoxon and other comparisons must be defined.

### 10. Explainability Evaluation

The thesis proposes guid-Xmage but does not fully specify a quantitative evaluation protocol.

### 11. Baseline Reproduction

The exact implementation and preprocessing of the 84.4% EfficientNet-B7 baseline must be documented.

### 12. Google Colab Reproducibility

Because Colab hardware can vary, experiments must record:

* GPU model;
* available VRAM;
* runtime type;
* software versions;
* random seeds;
* training configuration.

---

## Potential Implementation Risks

### GPU Memory Usage

Multiple feature streams combined with EfficientNet-B7 and 600×600 inputs may create substantial GPU memory requirements.

### Training Cost

The proposed architecture may be significantly more expensive than a standalone classifier.

### Data Leakage

Identity-level or image-level overlap between splits could invalidate evaluation.

### Mask Misalignment

Incorrect crop coordinates, resizing, or interpolation can corrupt ground-truth segmentation masks.

### Pixel-Class Imbalance

Manipulated pixels may represent a small portion of the image.

### MTL Loss Imbalance

The classification and segmentation losses may have different numerical scales.

### Overfitting

The model may learn dataset-specific artifacts rather than general manipulation features.

### Representation Ambiguity

Unspecified noise and frequency transforms may make the architecture difficult to reproduce.

### Explainability Overhead

guid-Xmage may add computational complexity.

### Colab Runtime Variability

GPU availability, VRAM, session duration, and runtime configuration may vary.

---

# 12. Implementation Summary

The implementation should prioritize the following order:

## Step 1 — Freeze the Research Specification

Before coding, finalize:

* official VERITAS expansion;
* exact multi-stream architecture;
* exact noise residual transform;
* exact frequency-domain transform;
* exact feature-fusion method;
* exact face-cropping method;
* augmentation policy;
* occlusion handling;
* statistical comparison protocol;
* explainability evaluation protocol.

---

## Step 2 — Build the Dataset Pipeline

Implement:

* OpenForensics ingestion;
* image/JSON validation;
* category filtering;
* polygon-to-mask conversion;
* mask verification;
* train/validation/test splitting;
* leakage detection.

---

## Step 3 — Build the Preprocessing Pipeline

Implement:

* face detection;
* invalid-face rejection;
* facial crop;
* image/mask alignment;
* 600×600 transformation;
* normalization;
* augmentation.

---

## Step 4 — Implement the Baseline

Build standalone EfficientNet-B7 classification.

Use this to establish the baseline comparison.

---

## Step 5 — Implement the Feature Representations

Implement and test:

* RGB representation;
* noise residual representation;
* frequency-domain representation.

Verify tensor dimensions and numerical ranges.

---

## Step 6 — Implement Feature Fusion

Implement the final fusion mechanism.

Document it precisely.

---

## Step 7 — Implement the EfficientNet-B7 Backbone

Integrate the fused representation with the EfficientNet-B7 architecture.

---

## Step 8 — Implement the Classification Head

```text
Global Average Pooling
        ↓
Fully Connected Layer
        ↓
Sigmoid
        ↓
Manipulation Probability
```

---

## Step 9 — Implement the DeepLabV3+/ASPP Segmentation Head

Use:

```text
ASPP
  ├── Dilation Rate 1
  ├── Dilation Rate 6
  ├── Dilation Rate 12
  ├── Dilation Rate 18
  └── Global Pooling Branch
        ↓
     Decoder
        ↓
  600×600 Mask Output
```

---

## Step 10 — Implement the Multi-Task Loss

```text
Lseg = 0.5 BCE + 0.5 Dice

Ltotal = 0.4 Lclassification
       + 0.6 Lsegmentation
```

Use AdamW with a learning rate of `1e-4`.

---

## Step 11 — Train Under Strict Data Isolation

```text
Training Set
→ Weight Updates
```

```text
Validation Set
→ Monitoring and Hyperparameter Tuning
```

```text
Test Set
→ Final Evaluation Only
```

---

## Step 12 — Perform Ablation Studies

At minimum:

* EfficientNet-B7 classification-only versus VERITAS MTL;
* RGB-only versus multi-representation streams;
* alternative loss configurations if included in the research design.

---

## Step 13 — Evaluate the System

Report:

* Accuracy;
* Precision;
* Recall;
* F1-score;
* mIoU;
* Dice;
* inference latency;
* peak VRAM;
* statistical significance;
* qualitative localization results.

---

## Step 14 — Document Reproducibility

Record:

* dataset version;
* split membership;
* preprocessing configuration;
* augmentation configuration;
* architecture configuration;
* representation transforms;
* fusion method;
* loss weights;
* optimizer;
* learning rate;
* random seeds;
* Google Colab runtime type;
* GPU model;
* VRAM;
* Python version;
* PyTorch version;
* relevant library versions.

---

# CONTEXT FOR FUTURE PROMPTS

VERITAS is a thesis project for a multi-task facial image-forensics system. The system must classify an input facial image as authentic or manipulated and produce a pixel-level mask showing the suspected manipulated regions.

The core architecture is:

```text
Input Image
→ Face Detection/Crop
→ 600×600 Preprocessing
→ RGB + Noise Residual + Frequency-Domain Representations
→ Feature Extraction and Fusion
→ EfficientNet-B7
→ Classification Head + DeepLabV3+/ASPP Segmentation Head
```

The classification output is a sigmoid manipulation probability. A probability of `>=0.5` means manipulated; below `0.5` means authentic.

The segmentation output is a pixel-level sigmoid probability map. A threshold of `0.5` produces the binary manipulation mask.

The backbone is EfficientNet-B7. The segmentation branch uses DeepLabV3+ with ASPP and dilation rates of 1, 6, 12, and 18, plus a global pooling branch.

The dataset is OpenForensics. The intended manipulation categories are authentic images, face swapping, face reenactment, and generative inpainting.

Each training sample is expected to include an image and JSON annotation containing the authenticity label, bounding-box information, and polygon annotation used to derive the binary manipulation mask.

The dataset split is:

```text
70% Training
15% Validation
15% Testing
```

The test set must remain isolated until final evaluation.

The specified segmentation loss is:

```text
Lseg = 0.5 BCE + 0.5 Dice
```

The specified total loss is:

```text
Ltotal = 0.4 Lclassification + 0.6 Lsegmentation
```

The optimizer is AdamW with a learning rate of `1e-4`.

Primary evaluation metrics are:

* Accuracy;
* Precision;
* Recall;
* F1-score;
* mIoU;
* Dice;
* inference latency;
* peak VRAM.

Planned statistical methods include:

* McNemar's Test;
* Wilcoxon Signed-Rank Test;
* permutation testing.

The significance level is `α = 0.05`.

The baseline is standalone EfficientNet-B7, with the thesis referencing an 84.4% classification result.

The development and experimentation environment is **Google Colab**, not Google Cloud Vertex AI Studio. Because Colab hardware may vary, each experiment should record the assigned GPU model, available VRAM, runtime type, software versions, and relevant configuration.

The main unresolved technical decisions are:

1. Exact architecture of the RGB, noise-residual, and frequency-domain streams.
2. Exact noise-residual-generation method.
3. Exact frequency-domain transform.
4. Exact feature-fusion method.
5. Exact relationship between the streams and EfficientNet-B7.
6. Definitive face-cropping procedure.
7. Final augmentation policy.
8. Handling of partially occluded faces.
9. Exact statistical pairing protocol.
10. Quantitative explainability evaluation.
11. Reproducible implementation of the 84.4% baseline.

When implementing or extending the system, preserve the confirmed decisions unless a change is explicitly justified as a research-design modification. Prioritize data integrity, reproducibility, leakage prevention, correct image/mask alignment, architecture clarity, and controlled ablation studies.
