# VERITAS

Vision-based EfficientNet-Reinforced Image Tampering Authentication and ASPP-Segmentation MTL-Framework.

Four EfficientNet-B7 variants, one experiment per notebook, face-crop level:

| Name | Multi-stream | Segmentation |
| --- | --- | --- |
| `baseline_effb7_*` | no | no |
| `multistream_no_seg_*` | yes | no |
| `seg_no_multistream_*` | no | yes |
| `full_veritas_*` | yes | yes |

| Study | Folder | Files |
| --- | --- | --- |
| OpenForensics Wild | `notebooks-wild/` | `WILD_01` … `WILD_05` |
| OpenForensics Full (GT-mask) | `notebooks-of/` | `OF_01` … `OF_05` |

Wild B0 capacity check: `notebooks-wild-b0/`.

The inference app uses the OpenForensics Full checkpoint (`full_veritas_defacto_best.zip` from `OF_04_full`). Teammates: **`APP_SETUP.md`**.

Kaggle steps: **`KAGGLE_RUN_GUIDE.md`**. `WILD_05_compare` can attach the Full (`*_defacto`) checkpoint to score Wild Test.

## What is in this repo

| Path | Purpose |
| --- | --- |
| `notebooks-wild/` | Wild B7 ablations + compare |
| `notebooks-of/` | OpenForensics Full B7 ablations + compare |
| `notebooks-wild-b0/` | Wild EfficientNet-B0 baseline |
| `make_wild_notebook.py` | Regenerates Wild notebooks |
| `make_defacto_notebook.py` | Regenerates `notebooks-of/` |
| `split_group_notebooks.py` | Regenerates both |
| `KAGGLE_RUN_GUIDE.md` | Kaggle instructions |
| `APP_SETUP.md` | Run the inference app |
| `veritas-app/` | Inference UI + API |

Images are not stored here. Attach the dataset on Kaggle. The ~800 MB zip is not in git.
