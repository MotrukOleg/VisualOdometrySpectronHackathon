"""Frame loading and Unicode-safe image I/O (cv2.imread fails on non-ASCII paths on Windows)."""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img, params or [])
    if not ok:
        raise IOError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def rescale(img: np.ndarray, s: float) -> np.ndarray:
    if abs(s - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def resize_long(img: np.ndarray, long_side: int | None, upscale: bool = False) -> tuple[np.ndarray, float]:
    """Downscale so that the longer side is at most `long_side` (never upscales unless asked).

    Returns (image, scale) where scale = new / original size.
    """
    if not long_side:
        return img, 1.0
    s = long_side / max(img.shape[:2])
    if s >= 1.0 and not upscale:
        return img, 1.0
    return rescale(img, s), s


class FrameSource:
    """Ordered sequence of frames from a directory (use scripts/extract_frames.py for videos)."""

    def __init__(
        self,
        root: str | Path,
        start: int = 0,
        step: int = 1,
        max_frames: int | None = None,
        resize_long: int | None = None,
        scale: float = 1.0,
        cache_size: int = 64,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Frames directory not found: {self.root}")
        all_files = sorted(
            (p for p in self.root.iterdir() if p.suffix.lower() in IMG_EXTS),
            key=lambda p: natural_key(p.name),
        )
        if not all_files:
            raise FileNotFoundError(f"No images in {self.root}")
        idx = list(range(len(all_files)))[start::step]
        if max_frames:
            idx = idx[:max_frames]
        self.indices = idx  # index in the full sorted list (for GT lookup)
        self.files = [all_files[i] for i in idx]
        self.resize_long = resize_long
        self.scale = scale
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._sizes: dict[int, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def name(self, i: int) -> str:
        return self.files[i].stem

    def load(self, i: int) -> np.ndarray:
        """BGR uint8 frame after resizing (`resize_long` cap, then the explicit `scale` factor)."""
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        img, _ = resize_long(imread(self.files[i]), self.resize_long)
        img = rescale(img, self.scale)
        self._sizes[i] = (img.shape[1], img.shape[0])
        self._cache[i] = img
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img

    def size(self, i: int) -> tuple[int, int]:
        """(width, height) of the resized frame."""
        if i not in self._sizes:
            self.load(i)
        return self._sizes[i]


def load_gt(path: str | Path, source: FrameSource) -> np.ndarray | None:
    """Ground-truth positions aligned with `source`, shape (N, 2), NaN where missing.

    The CSV must contain `x`, `y` (any planar metric/pixel units, e.g. local ENU metres)
    and either `name` (frame file stem) or `frame` (index in the full sorted frame list).
    GPS lat/lon must be converted to a local planar frame beforehand.
    """
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    out = np.full((len(source), 2), np.nan)
    if "name" in df.columns:
        lut = {str(n): (x, y) for n, x, y in zip(df["name"], df["x"], df["y"])}
        for k in range(len(source)):
            if source.name(k) in lut:
                out[k] = lut[source.name(k)]
    elif "frame" in df.columns:
        lut = {int(f): (x, y) for f, x, y in zip(df["frame"], df["x"], df["y"])}
        for k, orig in enumerate(source.indices):
            if orig in lut:
                out[k] = lut[orig]
    else:
        raise ValueError("GT CSV needs a 'name' or 'frame' column")
    return out
