# OW v2 architecture and implementation audit

Reviewed 2026-11-10. The most useful next steps are to establish that the auxiliary
targets help classification, then test nonlinear fusion and stronger local evidence.
More branches and a larger parameter count do not establish higher accuracy.

This review combines source inspection, synthetic CPU probes, historical mask
metadata, and primary research. Synthetic probes demonstrate properties of the
implementation; they do not measure classification accuracy. No new GPU training
or external job changes were made during this review.

## 1. Audit the supervision before strengthening the segmentation branch

**Confirmed code behavior:** `ow_v2_training.py:17` accepts the teacher's hard masks
as segmentation targets for every sample, with no label-consistency or confidence
filter. The mask writer thresholds probabilities at 0.5 into binary PNGs, discarding
the teacher's per-pixel confidence. Mask area is not a substitute for confidence.

**Important teacher provenance concern:** the repository's OF teacher dataset reads
GT masks only for fake samples (`make_defacto_notebook.py:888`), emits
`has_gt_mask = used_gt` (`:926`), and trains segmentation only where that flag is
true (`:1276`). Real all-zero masks are constructed but excluded from segmentation
supervision. This establishes the repository implementation. It does not establish
the exact source of the supplied Kaggle teacher checkpoint, version 3.

The only local mask index is an **older, failed OW00 v4 Test-only partial export**:

| Historical subset | Images | Mean positive pixel fraction |
| --- | ---: | ---: |
| Real | 1,268 | 0.535986 |
| Fake | 5,492 | 0.537345 |

Every mask in that historical subset is nonempty. This is a warning about the
intended tampered-region semantics. It is **not a measurement of the completed
v7 Train/Validation masks**, and must not be used to select architecture settings
from Test. The earlier completion checks establish coverage and file validity,
not that the masks locate manipulations correctly.

The final v7 index and images are not available locally. Fetching the supplied
teacher's public notebook export returned Kaggle 403 `kernels.get` permission
denied. Exact teacher-version attribution therefore remains unverified.

**First experiment:** inspect final v7 Train/Validation overlays and per-class mask
statistics. For verified Real inputs, use the known absence of manipulation as
zero segmentation supervision. Keep fake localization as an uncertain auxiliary
target; compare the existing loss with label-consistent, quality-filtered targets
and with classification-only training. Do not replace a missing/uncertain fake
mask with a full-positive face mask. Proper confidence weighting requires saved
teacher probabilities or new teacher inference; binary PNGs cannot recover them.

Do not make the classifier depend strongly on these masks before this check.
Research on [uncertainty-aware pseudo-label learning](https://arxiv.org/abs/2003.03773)
supports treating source-model predictions as noisy targets under domain shift;
its results on other datasets do not establish an improvement for VERITAS.

## 2. Late fusion currently adds independent scores

**Confirmed expressiveness limit:** `architecture_v2.py:84` projects 256 pooled
auxiliary features into 2,560 channels, adds them to RGB, then applies one linear
classifier (`:105`). At evaluation, with dropout disabled, this reduces to:

```text
logit = wᵀ RGB + (wᵀ P) [SRM; FFT] + bias
```

The projection has 655,360 parameters, but its auxiliary scoring contribution can
be collapsed into 256 effective weights. A numerical probe of nonzero projection
weights agrees with the collapsed expression to 7.15e-7. The auxiliary encoders
are nonlinear within each stream; there is no nonlinear interaction between the
pooled streams. Factorization can still affect optimization and regularization.
Additive fusion is a valid reference, not a proven accuracy defect.

**Candidate:** keep the direct RGB classification path and add a small residual
fusion MLP over RGB/SRM/FFT features, using normalization and one nonlinear hidden
layer, or an input-dependent stream gate. Initialize the residual output to zero
to preserve initial RGB predictions. Avoid zero-initializing two consecutive
learned factors, which can prevent either from learning. Compare additive and
nonlinear variants under the same data and optimization budget.

The [high-frequency forgery detection paper](https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Generalizing_Face_Forgery_Detection_With_High-Frequency_Features_CVPR_2021_paper.html)
provides a relevant precedent for interactions between RGB and residual modalities.
The proposed small residual MLP is our experiment, not a reproduction of that model.

## 3. Small local clues can be diluted by pooling

**Confirmed structure; unmeasured accuracy effect:** all three classification
streams use global average pooling. The RGB map at 600px is 19×19; the auxiliary
maps are 75×75 before pooling. Averaging can dilute an activation covering only a
few locations. It can also provide useful regularization, so replacing it is an
ablation rather than an automatic fix.

Segmentation uses encoder stages 2 and 5. A segmentation-only backward pass reaches
stages 0–5, but not the deepest classification stages 6–8. The classifier never
consumes the predicted mask. This is standard shared-trunk auxiliary supervision,
and does not imply that segmentation has no effect on classification.

**Candidate order:** compare average pooling with average-plus-max pooling or GeM.
Only after the mask-quality check, test residual mask-weighted pooling alongside
the global RGB path. Do not remove global features or require a predicted positive
mask for classification. Pooling that uses reliable local evidence is supported
as a research direction by [Multi-Attentional Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Multi-Attentional_Deepfake_Detection_CVPR_2021_paper.html).

## 4. SRM and FFT have specific information limitations

**Chromatic SRM information loss:** `architecture_v2.py:29` converts RGB to gray
before filtering. A synthetic checkerboard with R ±0.1 and
G ∓0.1×0.299/0.587 changes RGB by 0.1 but leaves SRM outputs unchanged in the probe.
RGB and FFT still retain information, so this is not blindness of the entire model.
Test per-color or luminance-plus-chroma residuals within the separate auxiliary
encoder; this changes its input channels and requires fresh training.

**Global FFT nuisance sensitivity:** FFT operates on an already letterboxed image
(`architecture_v2.py:31`). Constant gray content in a square versus a 300×600
rectangle letterboxed to 600px produces mean absolute FFT features of 5.56e-6 and
2.63e-4, about 47× apart, without any manipulation. This establishes sensitivity to
geometry, not evidence that geometry correlates with class in the dataset.

Audit padding fraction, detection fallback, crop scale, and compression on
Train/Validation. Then compare the global FFT with interior/local frequency
features or carefully windowed inputs. Windowing changes the signal too and can
suppress useful edge clues. [F3-Net](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1486_ECCV_2020_paper.php)
is a relevant precedent for local frequency statistics; [SciPy's spectral analysis
documentation](https://docs.scipy.org/doc/scipy/tutorial/signal.html) explains
windowing tradeoffs. Neither establishes which option wins here.

## 5. Empty-mask Dice provides almost no learning signal

**Confirmed loss behavior:** at 600×600, zero logits, and an empty target, the Dice
formula at `ow_v2_training.py:31` has loss approximately 1 with mean absolute logit
gradient 7.716e-18. BCE's corresponding gradient is 1.388889e-6. Averaging BCE and
Dice therefore halves useful BCE supervision for this case while adding an almost
constant Dice penalty. BCE still learns; the entire segmentation loss is not dead.

**Candidate fix:** apply BCE to all valid pixels, and foreground Dice to trusted,
nonempty targets only, with explicit weights and reductions. Handle a batch with
no trusted positives by returning a differentiable zero Dice term. Select the new
weights on Validation because this changes the effective loss balance. This becomes
particularly relevant if Real targets are corrected to all-zero masks.

Also, a static `lambda_seg=0.25` does not establish balanced or aligned gradients.
Measure shared-layer gradient norms and cosine similarity on a small Train sample
before adding adaptive task balancing. [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
describes the risk of conflicting task gradients; conflict has not been measured
on VERITAS data in this review.

## 6. Training and decoder cost deserve controlled experiments

The inherited settings use physical batch 4, one AdamW learning rate for the
pretrained B7 and new heads, and training-mode BatchNorm throughout B7. These are
valid choices, not demonstrated bugs. Test frozen B7 running statistics against
normal BatchNorm; train its affine parameters if appropriate. Test a smaller
backbone LR and larger new-head LR, with a short warmup. Apply comparable tuning
to the B7 baseline. Gradient accumulation does not increase the batch seen by BN.
[Group Normalization](https://arxiv.org/abs/1803.08494) motivates small-batch
normalization experiments, but replacing pretrained B7 BN with GN changes its
pretraining behavior and is not the first intervention proposed here.

Three epochs do not by themselves establish undertraining: one current full epoch
already has about 35,000 optimizer updates at accumulation 1. Use validation curves
and matched tuning budgets to decide whether more training helps.

The two dense decoder convolutions at `architecture.py:127` and `:130` operate at
150×150 and cost **29.0304G MAC per image**. Depthwise-separable replacements at the
same widths would cost **3.339G MAC**, approximately 8.7× less for those two layers.
These are operation counts, not measured wall-clock speedups for the whole model.
[DeepLabv3+](https://arxiv.org/abs/1802.02611) supplies a direct precedent for separable
ASPP/decoder convolutions. A lighter decoder could allow more experiments or a larger
physical batch within Kaggle limits; it could also reduce useful capacity.

ASPP rates 6/12/18 are conventional for output stride 16. At our 38×38 map, rate18
has about 46.8% in-bounds sampling taps averaged over locations. Smaller rates are
a reasonable later ablation, but the conventional configuration is not a bug.

## 7. A concrete checkpoint evaluation bug was fixed

The inherited loader coupled every checkpoint load to `CONFIG["resume"]`. With
`resume=False`, the final evaluation silently kept the last model instead of
loading the validation-selected best model.

`make_ow_v2_notebooks.py` now applies the resume switch only when continuing
training. Evaluation loads its selected checkpoint independently and still checks
architecture/protocol provenance. All five v2 training notebooks were regenerated.
A regression test confirms best-checkpoint loading with training resume disabled.
Legacy teacher behavior and legacy notebook experiments were not changed by this fix.

## 8. Verify the unit of labeling and the final comparison

OW00 selects the largest Haar face (`make_ow_notebooks.py:666`), falling back to
the whole image, then transfers the image's folder label to the saved crop. If a
Fake file contains multiple faces and the largest one is authentic, that crop can
receive an incorrect label. The original [OpenForensics paper](https://openaccess.thecvf.com/content/ICCV2021/html/Le_OpenForensics_Large-Scale_Challenging_Dataset_for_Multi-Face_Forgery_Detection_and_Segmentation_ICCV_2021_paper.html)
addresses multi-face forgery, but this does not prove that the Kaggle repackaging
used here contains such cases. If files are already single-face crops this concern
may not apply. Inspect final Train/Validation examples before proposing multi-face
training or multiple-instance pooling. Also check source/identity duplicates across
splits; the current protocol hash does not establish their independence.

For the final accuracy claim, compare paired predictions on the same held-out
images, report uncertainty in the accuracy difference, and repeat matched seeds.
Use source-level grouping for uncertainty if images share a source. Every final
architecture must be selected using Validation rather than repeated Test rankings.
An independently held-out dataset is needed for a broader generalization claim.

## Experiment order

1. Audit final Train/Validation supervision; compare classification-only with
   label-consistent auxiliary targets and empty-aware loss.
2. Compare additive versus nonlinear residual late fusion with everything else fixed.
3. Compare global average pooling versus one local-evidence pooling alternative.
4. Test color residuals and padding-aware/local frequency features separately.
5. Test normalization/LR changes and a separable decoder within recorded budgets.
6. Combine only changes that help Validation, freeze settings, and run the matched
   B7 comparison on the final held-out data across repeated seeds.

Retain the corrected original and additive late-fusion model as references. None
of the proposed architecture changes has established an accuracy gain yet.

## Reproduce the source-level probes

From the repository root, using a compatible Torch/torchvision environment:

```powershell
& '.\.venv-ow-v2\Scripts\python.exe' diagnostics/ow_v2_architecture_audit.py
& '.\.venv-ow-v2\Scripts\python.exe' -m pytest tests/test_ow_v2.py -q
```

The diagnostics use synthetic inputs and random model weights, make no pretrained
download, and do not train a study model. They print JSON and include assertions
for the properties they verify. Mask-quality conclusions still require the actual
final data and exact teacher provenance.
