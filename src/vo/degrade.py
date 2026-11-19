"""Imitation of a phone filming a monitor: moire, motion blur, glare, washed-out colours, noise, JPEG, bezel.

Used to build synthetic benchmarks with exact ground truth that look like the Advanced-level videos.
Every effect is driven by the given numpy Generator, so a dataset is reproducible from its seed.
"""
from __future__ import annotations

import cv2
import numpy as np

PRESETS = {
    "none": {},
    "easy": {"blur": (0, 4), "color": 0.5, "noise": 3, "jpeg": (80, 92)},
    "screen": {"moire": (0.25, 0.55), "blur": (0, 12), "glare": 0.5, "color": 1.0, "noise": 5, "jpeg": (50, 75),
               "bezel": 0.25},
}


def moire(img, rng, strength):
    """Screen pixel grid (fine grating) aliased by resampling -> low-frequency moire bands."""
    h, w = img.shape[:2]
    period = rng.uniform(2.2, 3.4)
    ang = rng.uniform(-0.15, 0.15)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * np.cos(ang) + yy * np.sin(ang)
    v = -xx * np.sin(ang) + yy * np.cos(ang)
    grid = 0.5 + 0.25 * (np.cos(2 * np.pi * u / period) + np.cos(2 * np.pi * v / period))
    out = img.astype(np.float32) * (1 - strength + strength * grid[..., None])
    s = rng.uniform(0.55, 0.8)  # camera sampling below the grid frequency -> aliasing
    small = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def motion_blur(img, rng, length):
    if length < 1.5:
        return img
    k = int(np.ceil(length)) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))


def glare(img, rng, strength):
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    sx, sy = rng.uniform(0.15, 0.4) * w, rng.uniform(0.15, 0.4) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    return img + 255 * strength * rng.uniform(0.3, 1.0) * g[..., None]


def screen_color(img, rng, strength):
    """Washed-out, bluish monitor rendering + camera gamma / white balance."""
    x = img / 255.0
    x = 0.5 + (x - 0.5) * (1 - 0.35 * strength)  # lower contrast
    x = x ** rng.uniform(1 - 0.35 * strength, 1 + 0.25 * strength)
    gains = 1 + strength * np.array([rng.uniform(0.0, 0.15), rng.uniform(-0.05, 0.05), rng.uniform(-0.12, 0.0)])
    return np.clip(x * gains, 0, 1) * 255  # BGR: more blue, less red


def bezel(img, rng):
    """Dark monitor frame + desk entering from one side."""
    h, w = img.shape[:2]
    side = rng.integers(4)
    frac = rng.uniform(0.05, 0.2)
    out = img.copy()
    if side in (0, 1):
        n = int(frac * h)
        sl = (slice(0, n), slice(None)) if side == 0 else (slice(h - n, h), slice(None))
    else:
        n = int(frac * w)
        sl = (slice(None), slice(0, n)) if side == 2 else (slice(None), slice(w - n, w))
    out[sl] = rng.uniform(10, 30) + rng.normal(0, 3, out[sl].shape)
    return out


def degrade(img: np.ndarray, rng: np.random.Generator, preset: str | dict = "screen") -> np.ndarray:
    p = PRESETS[preset] if isinstance(preset, str) else preset
    x = img.astype(np.float32)
    if "moire" in p:
        x = moire(x, rng, rng.uniform(*p["moire"]))
    if "color" in p:
        x = screen_color(x, rng, p["color"])
    if "glare" in p and rng.random() < p["glare"]:
        x = glare(x, rng, 0.6)
    if "blur" in p:
        x = motion_blur(x, rng, rng.uniform(*p["blur"]))
    if "bezel" in p and rng.random() < p["bezel"]:
        x = bezel(x, rng)
    if "noise" in p:
        x = x + rng.normal(0, p["noise"], x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if "jpeg" in p:
        q = int(rng.uniform(*p["jpeg"]))
        x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return x
