"""Build a 5-fake grid and a matching GT probe_mask overlay grid."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAW = Path(__file__).resolve().parent / "defacto_raw"
OUT = Path(__file__).resolve().parent
IDS = [
    "0.jpg_IN_1.jpg",
    "0.jpg_IN_10.jpg",
    "0.jpg_IN_50.jpg",
    "0.jpg_IN_100.jpg",
    "0.jpg_IN_110.jpg",
]
CELL = 320
PAD = 12
TITLE_H = 44
CAPTION_H = 36
OVERLAY_COLOR = np.array([255, 48, 80], dtype=np.float32)  # red-magenta
ALPHA = 0.48


def load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def load_mask(path: Path, hw: tuple[int, int]) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"cannot read {path}")
    if mask.shape[:2] != hw:
        mask = cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def letterbox(bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = bgr.astype(np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * (1.0 - ALPHA) + OVERLAY_COLOR * ALPHA
    # thin contour so the forged region is obvious
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    cv2.drawContours(out_u8, contours, -1, (255, 220, 40), 2)
    return out_u8


def font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(cells: list[np.ndarray], captions: list[str], title: str, dest: Path) -> None:
    n = len(cells)
    w = PAD + n * (CELL + PAD)
    h = TITLE_H + PAD + CELL + CAPTION_H + PAD
    canvas = Image.new("RGB", (w, h), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    ft = font(22)
    fc = font(14)
    draw.text((PAD, 10), title, fill=(240, 240, 245), font=ft)
    for i, (cell, cap) in enumerate(zip(cells, captions)):
        x = PAD + i * (CELL + PAD)
        y = TITLE_H + PAD
        rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(rgb), (x, y))
        tw = draw.textlength(cap, font=fc)
        draw.text((x + (CELL - tw) / 2, y + CELL + 8), cap, fill=(200, 200, 208), font=fc)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=95)
    print("wrote", dest)


def main() -> None:
    singles = OUT / "defacto_fakes"
    singles.mkdir(exist_ok=True)
    fake_cells = []
    overlay_cells = []
    captions = []
    for idx, stem in enumerate(IDS, start=1):
        img_p = RAW / "imgs" / f"{stem}.tif"
        mask_p = RAW / "masks" / stem
        if not img_p.exists():
            # kaggle may flatten or zip
            matches = list((RAW / "imgs").rglob(f"{stem}*"))
            if not matches:
                raise FileNotFoundError(img_p)
            img_p = matches[0]
        if not mask_p.exists():
            matches = list((RAW / "masks").rglob(stem))
            if not matches:
                raise FileNotFoundError(mask_p)
            mask_p = matches[0]
        bgr = load_bgr(img_p)
        mask = load_mask(mask_p, bgr.shape[:2])
        fake = letterbox(bgr, CELL)
        over = letterbox(overlay(bgr, mask), CELL)
        a, b = stem.replace(".jpg", "").split("_IN_")
        captions.append(f"{idx}. morph {a} → {b}")
        fake_cells.append(fake)
        overlay_cells.append(over)
        cv2.imwrite(str(singles / f"{idx:02d}_{stem}.png"), bgr)
        cv2.imwrite(str(singles / f"{idx:02d}_{stem}_mask_overlay.png"), overlay(bgr, mask))
    compose(
        fake_cells,
        captions,
        "DEFACTO Face — 5 morphing fakes",
        OUT / "defacto_5_fakes.png",
    )
    compose(
        overlay_cells,
        captions,
        "DEFACTO Face — same 5 with probe_mask overlay",
        OUT / "defacto_5_fakes_masks.png",
    )


if __name__ == "__main__":
    main()
