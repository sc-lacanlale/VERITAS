"""One frame with two faces: a second person beside a DEFACTO morph, mask only on the morph."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAW = Path(__file__).resolve().parent / "defacto_raw"
OUT = Path(__file__).resolve().parent
FAKE_IMG = RAW / "imgs" / "0.jpg_IN_1.jpg.tif"
FAKE_MASK = RAW / "masks" / "0.jpg_IN_1.jpg"
OTHER_IMG = RAW / "imgs" / "0.jpg_IN_110.jpg.tif"
CELL = 420
GAP = 36
OVERLAY = np.array([255, 48, 80], dtype=np.float32)


def load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(path)
    return img


def load_mask(path: Path, hw: tuple[int, int]) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(path)
    if mask.shape[:2] != hw:
        mask = cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def fit(bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def paint_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = bgr.astype(np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * 0.52 + OVERLAY * 0.48
    painted = np.clip(out, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(painted, contours, -1, (255, 220, 40), 2)
    return painted


def font(size: int):
    for name in ("C:\\Windows\\Fonts\\arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_pair(left: np.ndarray, right: np.ndarray, title: str, captions: tuple[str, str], dest: Path) -> None:
    pad = 16
    title_h = 48
    cap_h = 36
    w = pad + CELL + GAP + CELL + pad
    h = title_h + pad + CELL + cap_h + pad
    canvas = Image.new("RGB", (w, h), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), title, fill=(240, 240, 245), font=font(22))
    y = title_h + pad
    for i, (img, cap) in enumerate(zip((left, right), captions)):
        x = pad + i * (CELL + GAP)
        canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (x, y))
        tw = draw.textlength(cap, font=font(15))
        draw.text((x + (CELL - tw) / 2, y + CELL + 8), cap, fill=(210, 210, 216), font=font(15))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=95)
    print("wrote", dest)


def main() -> None:
    fake = load_bgr(FAKE_IMG)
    other = load_bgr(OTHER_IMG)
    mask = load_mask(FAKE_MASK, fake.shape[:2])
    left = fit(other, CELL)
    right = fit(fake, CELL)
    right_mask = fit(paint_mask(fake, mask), CELL)
    save_pair(
        left,
        right,
        "One frame, two faces — only one crop is scored",
        ("second face — not the scored crop", "manipulated morph"),
        OUT / "mixed_two_faces.png",
    )
    save_pair(
        left,
        right_mask,
        "Same frame — probe_mask only on the manipulated face",
        ("second face, no mask", "probe_mask on the morph only"),
        OUT / "mixed_two_faces_mask.png",
    )


if __name__ == "__main__":
    main()
