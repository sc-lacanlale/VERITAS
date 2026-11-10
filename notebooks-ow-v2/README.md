# OW v2: corrected original and late-fusion candidate

These are fresh experiments on OpenForensics Wild, starting from **ImageNet B7**.
The completed OW_00 masks supply auxiliary targets; no OpenForensics-trained
classifier or legacy VERITAS weights are loaded into these students.

See the [architecture audit](ARCHITECTURE_AUDIT.md) for remaining limitations,
verified code properties, and the next accuracy experiments.

| Notebook | RGB backbone | SRM/FFT fusion | Segmentation |
| --- | --- | --- | --- |
| `OWV2_01_baseline` | EfficientNet-B7 | none | no |
| `OWV2_02_multistream` | EfficientNet-B7 | corrected early fusion | no |
| `OWV2_03_segmentation` | EfficientNet-B7 | none | DeepLabV3+ |
| `OWV2_04_full` | EfficientNet-B7 | corrected early fusion | DeepLabV3+ |
| `OWV2_05_late_fusion` | EfficientNet-B7 | separate encoders, pooled features | DeepLabV3+ |
| `OWV2_06_compare` | reads all five final outputs | reports accuracy difference vs B7 | no training |

The baseline uses the same binary head as VERITAS. All variants use the same
initialization procedure for that head and the ImageNet backbone.

## What changed

The corrected original retains RGB/SRM/FFT → 9-channel B7 → classification and
segmentation. RGB stays ImageNet-normalized. Auxiliary stem weights start at zero,
preserving the original B7 RGB response rather than replacing it with an average
of unrelated domains. SRM and centered FFT features are bounded; FFT extraction
runs in float32 even under CUDA autocast.

The separate late-fusion candidate keeps B7's RGB stem intact. Small independent
SRM and FFT encoders pool their features, which are projected and added to pooled
RGB features before classification. That projection starts at zero; the auxiliary
encoders begin receiving gradients after the projection's first update. The
segmentation decoder receives spatial RGB backbone features. Frequency coordinates
are never interpreted as image locations by that decoder. This comparison also
changes auxiliary encoder capacity, so it evaluates the complete candidate rather
than isolating fusion depth alone.

Segmentation still helps classification through the shared backbone; neither
version adds explicit mask attention. The auxiliary loss is the average of BCE
and Dice, with float32 Dice reductions and an initial weight of **0.25** versus
classification's 1.0. This is a starting setting to validate, not proof of optimal
task balance. Separate loss terms are logged.

Image flips now occur after letterboxing, aligned with the saved mask even when
padding is odd. Augmentation is seeded per image and epoch. Checkpoints preserve
Torch RNG state for mid-epoch continuation and are saved at optimizer update
boundaries, including the final gradient accumulation group.

## Attach and run

1. Upload a training notebook and attach
   [`ahmedgamal/openforensics-wild-dataset`](https://www.kaggle.com/datasets/ahmedgamal/openforensics-wild-dataset)
   and the [completed OW_00 version 7 output](https://www.kaggle.com/code/clarenceshanley/veritas-ow-00-generate-masks/output?scriptVersionId=366854122).
   Its 190,335 masks cover Train 140,002, Validation 39,428, and Test 10,905.
   The loader accepts both a mounted `ow_masks` folder and Kaggle's `_output_.zip`.
2. Enable a GPU and run the default `run_mode: smoke` first. It uses 64 images per
   class per split, trains one epoch, and must print `TRAIN_SMOKE_VALIDATION_PASSED`.
   This checks execution; its accuracy is not a study result.
3. Start a fresh full run with `run_mode: full`. Detach smoke checkpoint outputs
   and clear the smoke session's working files by starting a new session.
   Checkpoint validation rejects smoke/full, architecture, or protocol mismatches.
4. Full training targets three epochs, one per saved run. For epochs two and three,
   attach the same experiment's latest **full** output, alongside Wild and OW_00.
   If the 11-hour budget interrupts an epoch, attach that partial output to resume
   the same epoch. Keep the seed, data, masks, and training configuration fixed.
   The existing legacy OW jobs and checkpoints are separate experiments.
5. Repeat for all five variants with matching configuration. Attach exactly one
   final output per variant to `OWV2_06_compare`.

Run `python make_ow_v2_notebooks.py` from the repo root to regenerate these six
self-contained notebooks. Shared v2 code lives in
`veritas-app/backend/inference/architecture_v2.py` and `ow_v2_training.py`.
The app dispatches checkpoints by `exp_cfg.implementation_version`; legacy
checkpoints continue using the legacy architecture.

## Accuracy experiment

All five variants select their best checkpoint by **Validation accuracy**, with
the same fixed classification threshold (0.5). Full runs evaluate Test only after
three epochs. Keep Test results out of architecture and hyperparameter decisions.
If tuning the auxiliary loss, learning rate, or other settings, use Validation
and equal tuning budgets for the baseline and candidates; freeze the final
configuration before the final Test comparison. For a different protocol or epoch
budget, update the comparison notebook's explicit checks too.

The comparison verifies final epoch count, Test size, unique experiment names,
and a common protocol hash covering settings, masks, and data indices. It reports
accuracy differences in percentage points. Repeat matched seeds before claiming
VERITAS is reliably better. Extra branches do not guarantee a higher score.
Dice is agreement with the legacy teacher's pseudo masks, not independently
annotated localization accuracy. The existing dataset split is inherited; this
implementation does not establish identity/source separation between splits.

Local tests cover the real B7 forward/backward paths, RGB-preserving initialization,
600px numerical behavior, checkpoint compatibility, mask alignment, and comparison
validation. They use random weights without a download. GPU training and final
accuracy must still be measured on Kaggle.
