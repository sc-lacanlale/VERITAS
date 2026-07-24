# VERITAS Quick Reference Card

## 🚀 Quick Start

### Validate Architecture
```bash
python validate_architecture.py
```

### Run Tests
```bash
python tests/test_full_architecture.py
```

### Create Model
```python
from src.config import Config
from src.models import VERITASModel

config = Config({'model': {'input_resolution': 600}})

# RGB-only model
model = VERITASModel(config, use_multi_representation=False)

# Full multi-representation model
model = VERITASModel(config, use_multi_representation=True)
```

### Run Inference
```python
import torch

x = torch.randn(2, 3, 600, 600)
outputs = model(x)

# Get predictions
predictions = model.predict(x, threshold=0.5)
cls_pred = predictions['classification_pred']
seg_mask = predictions['segmentation_mask']
```

---

## 📂 Key Files

| File | Purpose |
|------|---------|
| `src/models/veritas.py` | Complete VERITAS model |
| `src/models/representation_extractor.py` | Multi-representation extraction |
| `src/models/feature_fusion.py` | Feature fusion module |
| `src/models/aspp.py` | ASPP module |
| `src/models/deeplabv3plus.py` | DeepLabV3+ decoder |
| `src/models/loss.py` | Multi-task loss |
| `validate_architecture.py` | Quick validation script |
| `tests/test_full_architecture.py` | Comprehensive tests |
| `ARCHITECTURE_COMPLETE.md` | Full documentation |
| `IMPLEMENTATION_SUMMARY.md` | What's done & what's next |

---

## 🎯 Model Configurations

### Full Model (Best Performance)
```python
model = VERITASModel(
    config,
    use_multi_representation=True,
    use_full_segmentation=True
)
# Parameters: ~75M | Memory: ~12 GB (batch=4)
```

### RGB-Only (Baseline)
```python
model = VERITASModel(
    config,
    use_multi_representation=False,
    use_full_segmentation=True
)
# Parameters: ~68M | Memory: ~10 GB (batch=4)
```

### Lightweight (Fast Inference)
```python
model = VERITASModel(
    config,
    use_multi_representation=False,
    use_full_segmentation=False
)
# Parameters: ~52M | Memory: ~8 GB (batch=4)
```

---

## ✅ What's Complete

- [x] Project setup & infrastructure
- [x] Dataset pipeline (annotation parsing, masks, splitting)
- [x] Multi-representation extraction (RGB, noise, frequency)
- [x] Feature fusion module
- [x] EfficientNet-B7 backbone with skip connections
- [x] Classification head (global pool + FC)
- [x] ASPP module (multi-scale context)
- [x] DeepLabV3+ decoder (skip connections + upsampling)
- [x] Complete VERITAS model assembly
- [x] Multi-task loss function
- [x] Test suite
- [x] Documentation

---

## ⏳ What's Next

### Immediate (Now)
1. Run `python validate_architecture.py`
2. Run `python tests/test_full_architecture.py`
3. Review `ARCHITECTURE_COMPLETE.md`

### Short Term (Next Tasks)
- Task 17: Architecture validation checkpoint
- Task 19: Training pipeline
- Task 20: Full training (50 epochs)

### Long Term
- Tasks 22-26: Evaluation & metrics
- Tasks 28-31: Ablation studies
- Tasks 33-37: Visualization & deployment

---

## 🐛 Troubleshooting

### Import Errors
```python
# Make sure src is in path
import sys
sys.path.insert(0, 'src')
```

### CUDA Out of Memory
```python
# Reduce batch size or use lightweight model
model = VERITASModel(config, use_full_segmentation=False)
```

### Model Too Large
```python
# Check model size
print(f"Parameters: {model.get_num_parameters():,}")
print(f"Size: {model.get_model_size_mb():.2f} MB")
```

---

## 📊 Model Outputs

### Forward Pass
```python
outputs = model(x)
# outputs['classification_logit']: [B, 1] (pre-sigmoid)
# outputs['segmentation_logits']: [B, 1, 600, 600] (pre-sigmoid)
```

### Predictions
```python
predictions = model.predict(x, threshold=0.5)
# predictions['classification_prob']: [B, 1] in [0, 1]
# predictions['classification_pred']: [B, 1] in {0, 1}
# predictions['segmentation_prob']: [B, 1, H, W] in [0, 1]
# predictions['segmentation_mask']: [B, 1, H, W] in {0, 1}
```

---

## 🔧 Utilities

### Get Model Summary
```python
summary = model.get_architecture_summary()
```

### Freeze/Unfreeze Backbone
```python
model.freeze_backbone()    # For fine-tuning
model.unfreeze_backbone()  # For full training
```

### Get Fusion Weights
```python
from src.models import FeatureFusion
fusion = FeatureFusion(config)
weights = fusion.visualize_weights()
# weights = {'rgb_percent': 33.3, 'noise_percent': 33.3, ...}
```

---

## 💾 Memory Requirements

| GPU | Recommended Config | Batch Size |
|-----|-------------------|------------|
| 16+ GB | Full model | 4 |
| 12-16 GB | RGB-only | 2-4 |
| 8-12 GB | Lightweight | 2-4 |
| <8 GB | Lightweight | 1-2 |

---

## 📚 Documentation Files

| File | Description |
|------|-------------|
| `ARCHITECTURE_COMPLETE.md` | Complete architecture documentation |
| `IMPLEMENTATION_SUMMARY.md` | Implementation progress & next steps |
| `QUICK_REFERENCE.md` | This file - quick reference |
| `WHATS_NEXT.md` | Original planning document |
| `COMPONENTS_COMPLETE.md` | Infrastructure status |
| `SMOKE_TEST_PASSED.md` | Smoke test results |

---

## 🎓 Research Features

### Ablation Studies Supported
- ✅ RGB-only vs Multi-representation
- ✅ Full segmentation vs Lightweight
- ✅ Multi-task vs Classification-only
- ✅ Different loss weights

### Representations
- **RGB**: Standard normalized image
- **Noise**: SRM filters (3×3 kernels)
- **Frequency**: DCT (8×8 blocks, 3 bands)

### Architecture
- **Backbone**: EfficientNet-B7 (66M params)
- **Segmentation**: DeepLabV3+ with ASPP
- **Total**: 70-80M params (full model)

---

## 🔗 Important Paths

```
src/
├── models/
│   ├── veritas.py                    # Main model
│   ├── representation_extractor.py   # Task 10
│   ├── feature_fusion.py             # Task 11
│   ├── aspp.py                       # Task 14
│   ├── deeplabv3plus.py             # Task 15
│   └── loss.py                       # Multi-task loss
├── data/                             # Dataset utilities
└── utils/                            # Utilities

tests/
└── test_full_architecture.py         # Test suite

.kiro/specs/veritas-thesis-implementation/
├── tasks.md                          # Task list
├── requirements.md                   # Requirements
└── design.md                         # Design doc
```

---

## ⚡ Commands Cheat Sheet

```bash
# Validate architecture
python validate_architecture.py

# Run tests
python tests/test_full_architecture.py

# Run with pytest
pytest tests/test_full_architecture.py -v

# Quick smoke test (if you create one)
python smoke_test.py

# Check imports
python -c "from src.models import VERITASModel; print('OK')"
```

---

## 📞 Need Help?

1. **Architecture issues**: See `ARCHITECTURE_COMPLETE.md`
2. **Implementation details**: See `IMPLEMENTATION_SUMMARY.md`
3. **Next steps**: See `WHATS_NEXT.md`
4. **Design decisions**: See `.kiro/specs/veritas-thesis-implementation/design.md`

---

**Status**: ✅ Core architecture complete | 📝 Ready for training pipeline

**Next**: Run validation → Implement training pipeline (Task 19)
