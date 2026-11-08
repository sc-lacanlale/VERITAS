# OpenForensics Wild with generated masks

This folder is separate from `notebooks-wild/` and `notebooks-of/`.

`OW_00` runs the current Full checkpoint on every OpenForensics Wild image and saves a mask.
`OW_01` to `OW_04` then train on those Wild images and those masks. No other image set is attached.

The mask is the Full model’s prediction on one face crop, threshold 0.5. It is not a hand-drawn polygon. Dice in the training notebooks is agreement with that prediction. The classification label is still the Wild `Real` or `Fake` folder.

## What to attach

| Notebook | Attach | Writes |
| --- | --- | --- |
| `OW_00_generate_masks.ipynb` | Wild dataset, `full_veritas_defacto_best.pt` or `.zip`, and this notebook’s previous output when resuming | `/kaggle/working/ow_masks/` |
| `OW_01` … `OW_04` | Wild dataset, the finished `ow_masks` output, and that notebook’s own checkpoint on later epochs | `checkpoints/{experiment}_best.pt` and `_last.pt` |
| `OW_05_compare.ipynb` | The `OW_01`–`OW_04` outputs | `ow_compare.csv` |

Wild dataset: `ahmedgamal/openforensics-wild-dataset`.

## How a mask run stays under 12 hours

The Kaggle session limit is 12 hours. `OW_00` stops at **10.5 hours**, writes `index.csv` and `progress.json`, and exits.

1. Save the version.
2. New session. Attach the Wild dataset, the same Full checkpoint, and this notebook’s latest output.
3. Leave `shard_index` and `shard_count` unchanged. Run All.
4. Stop when `progress.json` says `"resume": false`.

Finished images are skipped. The new output copies the earlier PNGs forward, so the latest version of a shard is a complete mask set for that shard.

`manifest.csv` is the full image list. `index.csv` is the finished rows for this shard. Each PNG lives at `masks/{split}/{real or fake}/{image_id}.png`. `image_id` is a hash of `train/fake/filename`, so the same file name in another split does not collide. The crop box in `index.csv` is the box the mask was painted on. Training reads that box and does not detect the face again.

After 64 new images the log prints images per hour. If that estimate needs more than one session, resume as above. To split the work across GPUs instead, set `shard_count` before the first full run and start one notebook per `shard_index`. Training attaches every shard. Do not change `shard_count` after the first `index.csv` exists.

`run_mode: smoke` paints 8 images. `full` walks the shard.

## How training reads it

`OW_01`–`OW_04` start from ImageNet, learning rate 1e-4, both loss weights 1.0, 3 epochs, one epoch per run. They do not load the Full checkpoint’s weights.

A training run stops itself at 11 hours and writes a mid-epoch checkpoint every 30 minutes. Attach that output to continue. The log must show the epoch number going up by one when a full epoch finishes.

`full` refuses to train if any Wild image is still missing a mask. Finish `OW_00` first.
