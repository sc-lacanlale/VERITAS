# OW v3 publication and execution record

Recorded 2026-11-10. This file is a dated record, not a live job monitor.

## GitHub delivery

- Repository: [sc-lacanlale/VERITAS](https://github.com/sc-lacanlale/VERITAS).
- Branch: [codex/ow-v3-improvements](https://github.com/sc-lacanlale/VERITAS/tree/codex/ow-v3-improvements).
- Review: [pull request 1](https://github.com/sc-lacanlale/VERITAS/pull/1).
- Initial v3 implementation commit: `666ce53ddabb1102b65e7a98da458462bf8b9141`.
- Validation: **116 tests passed**, plus five unittest subtests; three pre-existing
  deprecation warnings. Tests include actual B7 backward/600px inference,
  self-contained notebook execution, aligned geometry, numerical behavior,
  optimizer/scheduler/RNG continuation, best-checkpoint selection, and app regressions.

The branch also retains the separately versioned v2 study and incorporates the
two main-branch updates that arrived after the original checkout. This publication
does not imply that the pull request has been merged into `main`.

## Kaggle publication

All twelve notebooks were saved privately under `clarenceshanley`, initially as
**version 1 without executing cells**. Every saved notebook was read back and its
cell sources, privacy, accelerator preference, and data references were checked.

| Notebook | Kaggle link |
| --- | --- |
| 00 mask audit | [Train/Validation audit](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-00-mask-audit) |
| 01 baseline | [B7 baseline](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-01-baseline) |
| 02 multistream | [Multistream classification](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-02-multistream) |
| 03 segmentation | [RGB segmentation model](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-03-segmentation) |
| 04 full | [Improved VERITAS](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-04-full) |
| 05 early reference | [Corrected early fusion](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-05-early-reference) |
| 06 additive reference | [Additive late fusion](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-06-additive-reference) |
| 07 nonlinear ablation | [Nonlinear fusion](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-07-nonlinear-ablation) |
| 08 pooling ablation | [Local evidence pooling](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-08-pooling-ablation) |
| 09 color ablation | [Color residuals](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-09-color-ablation) |
| 10 frequency ablation | [Local frequency features](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-10-frequency-ablation) |
| 11 comparison | [Four-core-model comparison](https://www.kaggle.com/code/clarenceshanley/veritas-ow-v3-11-compare) |

Kaggle briefly rejected a save because five CPU conversion workers were occupied.
Saving resumed after those workers completed. A successful quick-save conversion
has status COMPLETE but is not a training run. Require execution markers and
artifacts before recording model progress.

## Verified data and teacher provenance

Each training/audit upload requests `ahmedgamal/openforensics-wild-dataset` and
`clarenceshanley/veritas-ow-00-generate-masks/7`. Authenticated v7 logs verify
`SHARD COMPLETE`, `MASK_VALIDATION_PASSED`, 190,335 expected/validated masks and
`resume=false`. The archive size is 587,739,559 bytes. Coverage is Train 140,002,
Validation 39,428 and Test 10,905.

Kaggle's metadata readback canonicalizes the kernel source to `owner/slug`, omitting
the requested version suffix. Version 7 was independently verified and was the
current latest mask version at publication. Retain the explicit `/7` request and
verify source/version and protocol hash when continuing future runs.

The provided teacher source `nathanielescuro/ds18-veritas/3` was retrieved with
authorized access. It reads segmentation GT only for Fake samples and filters
loss by `has_gt_mask`. This verifies the absence of Real segmentation negatives
in that source. The downloaded notebook SHA256 is
`a6e383900c783a8dcf8ac23802f8c3d7aa42b58db64c0bb905692eea724dae3e`.
Its private source/checkpoints and raw datasets are not copied into this repository.
The students still initialize from ImageNet, with teacher knowledge only in the
already generated targets.

## Execution

The CPU audit and representative GPU smoke checks were submitted as **version 2**
of notebooks 00, 01 and 04, with a 1,800-second server limit each. Their outcomes
and subsequent full-run versions are recorded below after verification.

At submission, the account reported about 5.74 GPU hours remaining, with the quota
refresh at **2026-11-14 08:00 Asia/Manila**. Full runs must fit remaining quota and
preserve partial checkpoints. Neither a successful submission nor a partial epoch
is evidence of higher held-out accuracy.
