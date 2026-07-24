# VERITAS - Facial Image Forensics System

**VERITAS** (Vision-based EfficientNet-Reinforced Image Tampering Authentication and ASPP-Segmentation MTL-Framework) is a multi-task deep learning system for detecting and localizing facial manipulations in images.

## Overview

VERITAS jointly performs:
1. **Binary Classification**: Determines if a facial image is authentic or manipulated
2. **Pixel-Level Segmentation**: Localizes manipulated regions with binary masks

The system targets face swapping, face reenactment, and generative inpainting manipulations using a shared EfficientNet-B7 backbone with multi-representation feature extraction (RGB, noise residual, and frequency domain) and a DeepLabV3+/ASPP segmentation pathway.

## Project Structure

```
veritas-colab/
├── veritas_setup.ipynb          # Initial environment setup notebook
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   └── utils/
│       ├── __init__.py
│       ├── reproducibility.py   # Random seed utilities
│       └── environment.py       # Environment verification
├── dataset/                     # Dataset directory (on Google Drive)
│   ├── Train/
│   ├── Val/
│   └── Test-Dev/
└── README.md
```

## Google Drive Structure

When running on Google Colab, VERITAS creates the following structure on Google Drive:

```
/content/drive/MyDrive/VERITAS/
├── dataset/                     # Dataset storage
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs and metrics
├── results/                     # Evaluation results and visualizations
└── config.json                  # Configuration file
```

## Getting Started

### 1. Setup Environment

Open `veritas_setup.ipynb` in Google Colab and run all cells. This will:
- Mount Google Drive
- Create directory structure
- Install dependencies
- Verify CUDA/GPU availability
- Create configuration file
- Set up reproducibility

### 2. Dataset Preparation

Upload the OpenForensics dataset to Google Drive:
```
/content/drive/MyDrive/VERITAS/dataset/
├── Train/
│   ├── *.jpg
│   └── Train_poly.json
├── Val/
│   ├── *.jpg
│   └── Val_poly.json
└── Test-Dev/
    ├── *.jpg
    └── Test-Dev_poly.json
```

### 3. Configuration

The default configuration is created in `/content/drive/MyDrive/VERITAS/config.json`. You can load and modify it:

```python
from src.config import Config

# Load configuration
config = Config('/content/drive/MyDrive/VERITAS/config.json')

# Get values
batch_size = config.get('training.batch_size')
learning_rate = config.get('training.learning_rate')

# Update values
config.set('training.batch_size', 16)
config.update({'training': {'num_epochs': 100}})
```

### 4. Reproducibility

Set random seeds for reproducible experiments:

```python
from src.utils import set_random_seed

# Set seed (default: 42)
set_random_seed(42)
```

### 5. Environment Verification

Check your runtime environment:

```python
from src.utils.environment import verify_environment, print_environment_info

# Verify and print environment
env_info = verify_environment()
print_environment_info(env_info)
```

## Requirements

### Python Packages
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV (opencv-python-headless)
- NumPy
- scikit-learn
- scipy
- Pillow

### Hardware
- **Recommended**: Google Colab GPU runtime (T4, P100, or V100)
- **Minimum**: 12GB VRAM for batch size 8
- **Storage**: Google Drive with sufficient space for dataset and checkpoints

## Configuration

The configuration file (`config.json`) contains all settings:

### Model Configuration
- Input resolution: 600×600
- Backbone: EfficientNet-B7 (ImageNet pretrained)
- Multi-representation: RGB + Noise Residual + Frequency Domain

### Training Configuration
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: AdamW
- Epochs: 50
- Loss weights: Classification (0.4) + Segmentation (0.6)

### Data Configuration
- Train/Val/Test split: 70%/15%/15%
- Supported categories: authentic, face_swap, face_reenact, inpainting
- Random seed: 42

### Augmentation
- Rotation: ±2°
- Width/Height shift: ±10%
- Shear: up to 10%
- Zoom: up to 5%
- Horizontal flip: 50%

## Architecture

### Multi-Task Learning Pipeline
```
Input Image → Face Detection (YuNet) → Preprocessing (600×600)
    ↓
Multi-Representation (RGB + Noise + Frequency)
    ↓
Feature Fusion → EfficientNet-B7 Backbone
    ↓
    ├─→ Classification Head → Manipulation Probability
    └─→ Segmentation Head (DeepLabV3+/ASPP) → Binary Mask
```

### Key Components
- **Face Detection**: YuNet (OpenCV)
- **Noise Residual**: SRM filters (3 filters)
- **Frequency Domain**: 8×8 DCT blocks
- **Segmentation**: ASPP with dilations [1, 6, 12, 18]

## Research Context

This implementation supports a thesis research project investigating whether multi-task learning improves forensic representation quality compared to classification-only approaches. The system targets the 84.4% EfficientNet-B7 baseline classification performance while adding spatial localization capabilities.

## Implementation Phases

1. ✅ **Phase 1**: Environment setup and configuration - COMPLETE
2. 🟨 **Phase 2**: Dataset ingestion and validation - PARTIAL (parsing ✅, splitting ⏳)
3. ⏳ **Phase 3**: Face detection and preprocessing - NOT STARTED
4. ⏳ **Phase 4**: Multi-representation feature extraction - NOT STARTED
5. 🟨 **Phase 5**: Model architecture (EfficientNet-B7 + DeepLabV3+) - SIMPLIFIED VERSION COMPLETE
6. 🟨 **Phase 6**: Training pipeline - BASIC LOOP COMPLETE
7. ⏳ **Phase 7**: Evaluation metrics - NOT STARTED
8. ⏳ **Phase 8**: Ablation studies - NOT STARTED
9. ⏳ **Phase 9**: Baseline comparison - NOT STARTED
10. ⏳ **Phase 10**: Deployment (Gradio app) - NOT STARTED

## 🔥 Smoke Test Ready

The implementation now has enough infrastructure to run a **smoke test**!

**Run this**: `smoke_test.ipynb` - Trains on 100 images for 2 epochs to verify all components work together.

**See**: `SMOKE_TEST_READY.md` for complete smoke test guide.
**Status**: `IMPLEMENTATION_STATUS.md` for detailed progress.

## License

This project is for research purposes as part of a thesis implementation.

## Citation

If you use this implementation, please cite:

```
VERITAS: Vision-based EfficientNet-Reinforced Image Tampering Authentication 
and ASPP-Segmentation MTL-Framework
[Thesis Author], [Year]
```

## Contact

For questions or issues, please refer to the thesis documentation or contact the research team.
