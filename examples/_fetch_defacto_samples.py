"""Download 5 DEFACTO Face fakes + probe_masks from Kaggle (not the full 42GB)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent / "defacto_raw"
ROOT.mkdir(parents=True, exist_ok=True)
DS = "defactodataset/defactoface"

# IDs known to exist (donor_mask listing confirmed 0.jpg_IN_1.jpg etc.)
IDS = [
    "0.jpg_IN_1.jpg",
    "0.jpg_IN_10.jpg",
    "0.jpg_IN_50.jpg",
    "0.jpg_IN_100.jpg",
    "0.jpg_IN_110.jpg",
]


def dl(api: KaggleApi, rel: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"GET {rel} ...", flush=True)
    try:
        api.dataset_download_file(DS, rel, path=str(dest), quiet=False, force=True)
        print(f"  OK {time.time() - t0:.1f}s", flush=True)
        return True
    except Exception as e:
        print(f"  FAIL {time.time() - t0:.1f}s {e}", flush=True)
        return False


def main() -> int:
    api = KaggleApi()
    api.authenticate()

    # Confirm probe_mask path (donor_mask already worked).
    if not dl(api, "morphing_annotations/probe_mask/0.jpg_IN_1.jpg", ROOT / "masks"):
        print("probe_mask path failed", flush=True)
        return 1

    img_patterns = [
        "morphing_img/{id}.tif",
        "morphing_img/img/{id}.tif",
        "morphing_img/{id}",
        "swapping_img/{id}.tif",
        "swapping_img/img/{id}.tif",
        "swapping_img/{id}",
    ]
    img_pat = None
    for pat in img_patterns:
        rel = pat.format(id=IDS[0])
        if dl(api, rel, ROOT / "imgs"):
            img_pat = pat
            break
    if img_pat is None:
        print("no image path worked", flush=True)
        return 1

    print("using image pattern", img_pat, flush=True)
    kind = "morphing" if img_pat.startswith("morphing") else "swapping"
    mask_pat = f"{kind}_annotations/probe_mask/{{id}}"

    for i in IDS[1:]:
        dl(api, img_pat.format(id=i), ROOT / "imgs")
        dl(api, mask_pat.format(id=i), ROOT / "masks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
