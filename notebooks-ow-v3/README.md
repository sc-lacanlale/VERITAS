# OW v3 model improvements and controlled experiments

OW v3 implements the additional architecture and training changes identified in
the [v2 audit](../notebooks-ow-v2/ARCHITECTURE_AUDIT.md). It retains EfficientNet-B7,
binary authenticity classification, and optional DeepLabV3+/ASPP segmentation.
The improved model adds nonlinear residual fusion, average-plus-max evidence
pooling, color residuals, local frequency features, and a separable decoder.
These are candidates for better accuracy, not measured accuracy improvements.

Every student starts from **ImageNet EfficientNet-B7**, with new heads and auxiliary
encoders. Legacy VERITAS/OpenForensics weights are used only indirectly through the
already generated masks. Do not load a legacy or v2 model as a v3 training checkpoint.
The corrected early-fusion and additive late-fusion references are trained again
under the same v3 supervision and optimization protocol as the improved candidate.

The source branch is `codex/ow-v3-improvements`. The [publication and execution
record](PUBLICATION.md) contains verified GitHub/Kaggle links, versions and checks.

## Notebook map

`OWV3_00_mask_audit.ipynb` inspects final Train/Validation mask metadata and sample
overlays without training. `OWV3_01`–`OWV3_10` are ten separately named training
experiments. `OWV3_11_compare.ipynb` compares the four completed core experiments.

| Notebook | Experiment | Streams and fusion | Pooling | Segmentation decoder |
| --- | --- | --- | --- | --- |
| `01_baseline` | `baseline_effb7_ow_v3` | RGB only | average | none |
| `02_multistream` | `multistream_no_seg_ow_v3` | color SRM + local FFT; nonlinear late fusion | average + max | none |
| `03_segmentation` | `seg_no_multistream_ow_v3` | RGB only | average | separable + GroupNorm |
| `04_full` | `full_veritas_ow_v3` | color SRM + local FFT; nonlinear late fusion | average + max | separable + GroupNorm |
| `05_early_reference` | `early_reference_ow_v3` | grayscale SRM + global FFT; corrected early fusion | average | dense |
| `06_additive_reference` | `additive_reference_ow_v3` | grayscale SRM + global FFT; additive late fusion | average | dense |
| `07_nonlinear_ablation` | `nonlinear_ablation_ow_v3` | grayscale SRM + global FFT; nonlinear late fusion | average | dense |
| `08_pooling_ablation` | `pooling_ablation_ow_v3` | grayscale SRM + global FFT; nonlinear late fusion | average + max | dense |
| `09_color_ablation` | `color_ablation_ow_v3` | color SRM + global FFT; nonlinear late fusion | average + max | dense |
| `10_frequency_ablation` | `frequency_ablation_ow_v3` | color SRM + local FFT; nonlinear late fusion | average + max | dense |

The core four form the classification-only baseline, multi-stream-only,
segmentation-only, and full-model comparisons. The stepwise architecture path is
**06 → 07 → 08 → 09 → 10 → 04**: nonlinear fusion, pooling, color residuals, local
frequency features, then the separable decoder package. Use Validation to assess
these choices before a final held-out comparison. Notebook 05 retains the corrected
original early-fusion topology as a separate reference.

Each step changes the named component package, including its parameter count or
input representation where applicable. In particular, the separable decoder also
changes new-decoder normalization from BatchNorm to GroupNorm; a gain cannot be
attributed to depthwise separation alone. Nonlinear and pooling adapters also alter
capacity. Report these differences rather than claiming capacity-matched isolation.

## Changes from v2

| Area | v2 behavior | v3 behavior |
| --- | --- | --- |
| Real localization targets | copied teacher mask | known Real label gives an all-zero target in memory |
| Empty Fake masks | included as segmentation targets | localization unavailable; classification continues |
| Segmentation reduction | BCE/Dice average for every image | full BCE for Real; BCE/Dice average for retained nonempty Fake |
| Letterbox pixels | included in segmentation loss and metrics | explicit content mask excludes padding |
| Late fusion | linear auxiliary projection added before a linear head | nonlinear residual head plus direct RGB classifier path |
| Evidence pooling | global average | optional average-plus-max features |
| SRM | three filters on grayscale | optional three filters for each RGB channel, nine residual maps |
| Frequency | global normalized FFT magnitude | optional local three-band RGB power statistics after removing outer padding |
| Decoder | dense 3×3 convolutions | optional separable ASPP/decoder with GroupNorm |
| Optimization | common parameter LR; B7 BN statistics update | backbone/head LR groups; frozen B7 BN statistics by default |
| LR schedule | epoch-level cosine | per-successful-update warmup and cosine, checkpointed |
| Smoke evaluation | included a small Test evaluation | Validation only |
| Checkpoint provenance | architecture and data protocol | source digest added to the protocol; strict model loading |
| Localization reporting | combined teacher agreement | available Fake teacher agreement and Real positive-mask rates separately |

### Architecture behavior

The RGB backbone and classifier are constructed before variant-specific modules,
using the same initialization procedure across experiments. The corrected early
stem preserves its pretrained RGB weights and initializes auxiliary input weights
to zero. Additive late fusion preserves the v2 reference design. Nonlinear late
fusion retains the direct RGB score and adds a LayerNorm/MLP residual score; only
the residual's final layer starts at zero. Initial RGB predictions are preserved,
and earlier residual layers begin receiving gradients after that output layer learns.

Average-plus-max pooling provides both broad evidence and strong local responses.
It does not require a positive predicted mask. Segmentation continues to use spatial
backbone features at stages 2 and 5; local frequency coordinates are not fed to the
spatial decoder. No hard teacher-mask attention gate was added.

The local frequency branch crops each tensor using its explicit rectangular valid
canvas mask. It uses nonoverlapping 32px patches, patch-mean removal, a Hann taper,
and three radial frequency bands per RGB channel. Incomplete patches use replicate
extension. Nine log-power maps are adaptively pooled to 32×32 before their encoder.
No per-image min/max normalization discards the absolute power scale in this branch.
This is a local FFT implementation, not a reproduction of F3-Net's DCT model.

The separable decoder keeps the ASPP rates and feature widths. At 600px, replacing
its two main dense decoder convolutions changes their theoretical cost from
29.0304G to 3.339G multiply-adds per image, about 8.7× less **for those two layers**.
This is not a measured whole-model speedup, memory reduction, or accuracy gain.

### Supervision and mask quality

The original PNGs remain unchanged. The loss creates zero targets for Real images
from their authenticity labels. For Fake images it retains nonempty teacher masks
unless an optional quality flag excludes them. Nonempty means that a localization
target exists; it does not establish that the mask is accurate or confident.

BCE is averaged over valid content pixels separately for each retained image.
Real images contribute full BCE. Retained nonempty Fake images contribute
`(BCE + dice_weight × Dice) / (1 + dice_weight)`. Those per-image losses are then
averaged, and the auxiliary task receives `lambda_seg`. Defaults are
`lambda_cls=1`, `lambda_seg=0.25`, and `dice_weight=1`. Dice applies only to retained,
nonempty Fake targets, so an all-Real batch no longer spends half its auxiliary
weight on an almost constant empty-target Dice term. All spectral operations and
loss reductions that require it run in float32 under mixed-precision training.

`CONFIG["seg_quality_csv"]` optionally points to an attached CSV with unique,
canonical `rel_key` values and binary `seg_valid` flags, for example:

```csv
rel_key,seg_valid
train/fake/example_a.jpg,0
val/fake/example_b.jpg,1
```

The example names are placeholders; use keys from the actual dataset index. A zero
flag excludes that Fake mask from segmentation, while its classification label is
still used. Unlisted entries default to one. The CSV must not contain `test/` keys.
Real zero supervision remains valid regardless of that optional Fake-quality flag.
Fake masks with no positive valid pixels are ignored even if their flag is one.
The effective flags are included in the protocol hash.

Quality flags should come from a recorded assessment of Train/Validation data.
Mask area alone is not confidence. Binary PNGs discarded the teacher probabilities,
so calibrated confidence weighting requires saved probabilities or new teacher
inference; v3 does not invent confidence from the existing PNGs.

## Attach data and run the quality audit

Authenticated source verification on 2026-11-10 confirmed that version 3 of the
provided `nathanielescuro/ds18-veritas` teacher reads GT masks only for Fake samples
and gates segmentation loss on `has_gt_mask`. Real images therefore did not supply
negative segmentation supervision in that source. This verifies the earlier code
concern; it does not establish the accuracy of the teacher's individual masks.

Attach the same sources to every training experiment:

- [OpenForensics Wild dataset](https://www.kaggle.com/datasets/ahmedgamal/openforensics-wild-dataset).
- [Completed OW00 mask output, version 7](https://www.kaggle.com/code/clarenceshanley/veritas-ow-00-generate-masks/output?scriptVersionId=366854122).
- The optional quality CSV, if used consistently across the study.
- Only that experiment's own latest **full** checkpoint output when continuing.

Verify the input's version in Kaggle rather than assuming its latest version is
the intended one. The completed output contains 190,335 masks: Train 140,002,
Validation 39,428, and Test 10,905. Loaders support both mounted folders and Kaggle's
`_output_.zip` wrapper; checkpoint staging also handles wrapped outputs.

Run `OWV3_00_mask_audit.ipynb` first; CPU is sufficient. It writes
`mask_quality_train_val.csv` with per-class area/empty-rate statistics and
`mask_audit_train_val.png` with up to eight overlays per class per split, 32 total.
It prints `MASK_AUDIT_COMPLETED_TRAIN_VAL_ONLY`. Inspect the overlays and investigate
inconsistent masks before increasing their influence. This checks metadata and
sample plausibility, not independent localization accuracy. Test mask quality is
not used for architecture or quality-filter decisions.

## Smoke then full training

1. Upload the desired self-contained notebook, attach Wild and the completed masks,
   and enable a GPU. Internet access is needed for the first ImageNet B7 download
   unless those weights are already cached. The implementation uses one Torch GPU
   device; selecting a two-GPU Kaggle accelerator does not make it multi-GPU training.
2. Run the default `run_mode="smoke"`. It uses 64 images per class per split and one
   training epoch, selects its checkpoint with Validation, and evaluates Validation
   only. It must print `TRAIN_SMOKE_VALIDATION_PASSED`. Its small-sample accuracy is
   an execution check, not a thesis result. No final Test result is produced.
3. Start a new working session for `run_mode="full"`, with smoke checkpoint outputs
   detached. Full training targets three epochs, one completed epoch per saved run.
   Keep dataset, mask version, source, quality flags, seed, and settings fixed.
4. For continuation, attach that experiment's latest full output and run again.
   The 11-hour budget can save a partial epoch; continue it with the same source
   and configuration. Check `PARTIAL_EPOCH_SAVED`, `EPOCH_COMPLETE`, and the saved
   `logs/run_state.json` rather than inferring completion from a successful save.
5. Test is evaluated only after all three full epochs finish, using the best
   **Validation-accuracy** checkpoint. If training finishes at the time limit,
   `TRAINING_COMPLETE_TEST_PENDING_TIME_BUDGET` means another session is required.
   Attach the completed output and rerun; no further training is needed. Explicit
   `skip_training=True` requires a completed checkpoint.
6. Complete the four core experiments and attach exactly one final output from each
   to `OWV3_11_compare.ipynb`. It rejects incomplete/smoke results, duplicate/missing
   experiments, differing protocol hashes, and an unexpected Test sample count.

Checkpoint provenance rejects legacy/v2 weights, another architecture, smoke/full
mismatches, changed source code, or changed data/optimization protocol. Changing
model or runtime source requires a new experiment rather than continuing old weights.
`resume=False` starts fresh only in a clean working session; final best-checkpoint
loading remains independent of that training-resume switch.

For reproducible CLI packages, run `python prepare_ow_v3_kaggle.py` after generating
the notebooks. It validates and copies code plus private upload metadata into
`kaggle_working/ow-v3-publish`; it never publishes or starts a job. For a full-run
package, use `--only 01 04 --run-mode full`, optionally setting
`--time-budget-hours` and this same experiment's `--resume-source owner/slug/version`.
Review the generated metadata, then use the official Kaggle CLI:

```powershell
& '.\.venv-publish\Scripts\kaggle.exe' kernels push -p FOLDER --no-run
```

Omitting `--no-run` starts a saved run. Quick saves can temporarily occupy Kaggle
CPU conversion slots, and a quick-save status of COMPLETE does not mean training
executed. Require the model's validation/training markers and artifacts. Shortening
the wall-clock budget preserves a partial epoch; it does not change the training
protocol or turn that partial run into a final result.

## Shared optimization and reproducibility

| Setting | Default |
| --- | --- |
| Input resolution | 600×600 |
| Physical batch / accumulation | 4 / 1 |
| Seed | 42 |
| Full epochs / completed epochs per run | 3 / 1 |
| Optimizer | AdamW, weight decay 1e-4 |
| New-module LR / backbone LR | 1e-4 / 1e-5 |
| Warmup | first 5% of planned optimizer updates |
| Subsequent schedule | cosine decay per successful optimizer update |
| B7 BatchNorm | running statistics frozen; affine parameters trainable |
| Classification and mask thresholds | 0.5 / 0.5 |
| Session budget / mid-epoch save interval | 11 hours / 30 minutes |

These are common starting settings, not validation-selected optima. Apply comparable
tuning budgets to B7 and VERITAS. Accumulation changes the number of optimizer
updates, not the physical batch seen by normalization layers.

The dataset applies photometric augmentation to RGB only and flips the RGB canvas,
teacher mask, and valid canvas mask together after letterboxing. Per-image/epoch
augmentation seeds, the epoch sampler, and checkpointed Torch RNG state support
continuation. Checkpoints also preserve optimizer, scaler, per-update scheduler,
partial-step position, and accumulated loss statistics. They are written at update
boundaries, including the final accumulation group. This does not guarantee bitwise
identity across different GPUs, Torch versions, or nondeterministic GPU kernels.

The protocol hash covers common settings, relevant source digests, mask metadata,
and effective data indices/quality flags. Architecture flags remain separate so
matched variants can share a protocol while rejecting each other's checkpoints.
Image-index matching is not proof that the underlying image bytes or source
identities are independent across splits.

## Outputs and interpretation

- `checkpoints/{experiment}_last.pt` resumes the latest progress;
  `{experiment}_best.pt` is selected by Validation accuracy.
- `logs/protocol.json`, `logs/run_state.json`, and `logs/validation_history.csv`
  record provenance, epoch status, loss components, retained/ignored targets, and
  validation results.
- `smoke_results.json` records Validation smoke results.
- `experiment_results.csv` and `predictions/{experiment}_predictions.csv` are final
  full-Test outputs, including paired image keys for later statistical analysis.
- `logs/inference_benchmark.json` records the optional model-only benchmark:
  float32, one warmup batch, three timed repetitions, synchronized CUDA timing,
  and peak allocated CUDA memory. It excludes preprocessing and reports batch
  latency plus per-image time **at the recorded batch size**, not single-image
  end-to-end application latency. Repeat longer measurements on matched hardware
  before making a deployment speed claim.

Classification Accuracy/F1/precision/recall use Wild folder labels. Fake Dice/IoU
are reported only for retained nonempty teacher targets, with the corresponding
sample count. Real predicted positive-pixel fraction and nonempty-mask rate are
reported separately. They must not be combined into a claim of independently
annotated localization accuracy. The comparison reports accuracy differences in
percentage points; repeated matched seeds and paired uncertainty estimates are
needed before claiming reliable superiority.

## App and source layout

`architecture_v3.py` is the canonical model implementation. `ow_v3_training.py`
contains loss/optimizer/scheduler helpers, `ow_v3_runtime.py` contains training and
evaluation, and `make_ow_v3_notebooks.py` embeds those sources into every notebook.
Run `python make_ow_v3_notebooks.py` from the repository root to regenerate them;
each code cell is compiled and the notebook schema is validated before writing.

The app dispatches checkpoints by `exp_cfg.implementation_version`. v3 loads its
recorded architecture strictly and never downloads pretrained weights for inference.
For local-frequency models, both ordinary and streaming inference construct the
valid mask from exact `left/top/nw/nh` geometry. They do not infer padding from black
pixels. Legacy and v2 checkpoints retain their previous architecture dispatch.

## Manuscript alignment and remaining limits

The drafts' EfficientNet-B7/ASPP multi-task objective and controlled baseline
comparisons remain applicable. Their methods/results sections need to distinguish:

- Teacher-generated pseudo masks from polygon-derived ground-truth masks. This OW
  study establishes teacher agreement, not independent manipulation localization.
- Validation accuracy checkpoint selection from draft passages specifying F1.
- Nonlinear/late fusion and the separable decoder as new architecture variants,
  rather than implementation-only corrections to the original early-fusion design.
- Implemented fixed starting settings and explicit ablations from a completed
  hyperparameter optimization study. No such optimization outcome is assumed.

Final Train/Validation masks still need quality assessment; the code does not
establish that every retained Fake target is correct. The largest-face crop can
inherit an image-level label; mixed-face label correctness has not been established
for the Kaggle repackaging. Source/identity separation and near-duplicate leakage
across official splits have not been audited. An external held-out dataset is needed
for a broader generalization claim. These issues are not repaired merely by adding
nonlinear fusion, more channels, or a lighter decoder.

No trained accuracy gain, final localization score, or whole-model speedup is
claimed by this implementation documentation. Record actual run versions, selected
checkpoints, hardware, metrics, and statistical evidence before drawing conclusions.
