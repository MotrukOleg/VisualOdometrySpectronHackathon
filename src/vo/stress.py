"""Stress test of frame -> map localisation: how strong a degradation does each method survive?

Every "axis" is one kind of degradation with a severity level (0 = clean). Photometric axes (blur, noise,
clouds, darkness) are applied to the clean frames; geometric axes (altitude, tilt) re-render the frames
from the detailed image of the area, so the ground truth stays exact in every case.

Degradations are deterministic for a given (frame index, level) seed: all methods see identical images.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .degrade import motion_blur
from .synthetic import frame_to_world


def add_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Gaussian sensor noise, sigma in grey levels."""
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def add_clouds(img: np.ndarray, rng: np.random.Generator, coverage: float) -> np.ndarray:
    """Opaque white clouds over `coverage` of the frame (fractal noise thresholded, soft edges)."""
    if coverage <= 0:
        return img
    h, w = img.shape[:2]
    field = np.zeros((h, w), np.float32)
    for cell, amp in ((160, 1.0), (60, 0.5), (20, 0.25)):
        small = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
        field += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    thr = np.quantile(field, 1.0 - coverage)
    alpha = np.clip((field - thr) / 0.08 + 0.5, 0, 1)[..., None]  # soft cloud edge
    cloud = np.full_like(img, 235, np.float32)
    return (img * (1 - alpha) + cloud * alpha).astype(np.uint8)


def darken(img: np.ndarray, rng: np.random.Generator, factor: float) -> np.ndarray:
    """Low light: exposure x factor, then sensor noise and 8-bit quantisation (detail is lost in the dark)."""
    dim = img.astype(np.float32) * factor + rng.normal(0, 3.0, img.shape)
    return np.clip(np.round(dim), 0, 255).astype(np.uint8)


def blur(img: np.ndarray, rng: np.random.Generator, length: float) -> np.ndarray:
    return motion_blur(img, rng, length)


@dataclass
class Axis:
    name: str
    label: str  # x-axis label for plots
    levels: list[float]  # first level = clean
    apply: Callable | None = None  # photometric: (img, rng, level) -> img
    geometric: str | None = None  # "altitude" | "tilt": the frame is re-rendered


AXES = {
    "blur": Axis("blur", "motion blur, px", [0, 5, 10, 20, 30, 45], blur),
    "noise": Axis("noise", "sensor noise sigma, grey levels", [0, 10, 20, 35, 50, 70], add_noise),
    "clouds": Axis("clouds", "frame covered by clouds, %", [0, 0.2, 0.4, 0.6, 0.75, 0.9], add_clouds),
    "darkness": Axis("darkness", "exposure (1 = normal)", [1.0, 0.5, 0.25, 0.12, 0.06, 0.03], darken),
    "altitude": Axis("altitude", "altitude vs. task crops (frame covers x^2 area)", [1, 2, 4, 6, 8],
                     geometric="altitude"),
    "tilt": Axis("tilt", "camera tilt (perspective), x1e-4", [0, 5, 10, 15, 20], geometric="tilt"),
}


_PYRAMID: dict = {}


def _downsampled(detail: np.ndarray, f: int) -> np.ndarray:
    """detail / f with area averaging (cached): warping straight from full resolution would alias."""
    key = (id(detail), f)
    if key not in _PYRAMID:
        _PYRAMID.clear()  # one level at a time is enough and keeps memory flat
        _PYRAMID[key] = detail if f == 1 else cv2.resize(detail, None, fx=1 / f, fy=1 / f,
                                                         interpolation=cv2.INTER_AREA)
    return _PYRAMID[key]


def render_frame(detail: np.ndarray, k: float, row, frame_size: tuple[int, int], altitude: float = 1.0,
                 tilt: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Frame at the GT pose of `row` (map px) cut from the detailed image (k detail px per map px).

    altitude multiplies the ground footprint; tilt is the perspective term of synthetic.frame_to_world.
    The frame centre always projects to the GT centre.
    """
    w, h = frame_size
    f = max(1, int(row["scale"] * k * altitude))  # detail px per frame px, rounded down
    src, kf = _downsampled(detail, f), k / f
    cx, cy = (row["x"] + 0.5) * kf - 0.5, (row["y"] + 0.5) * kf - 0.5
    M = frame_to_world(cx, cy, row["yaw_deg"], row["scale"] * kf * altitude, w, h, tilt)
    return cv2.warpPerspective(src, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)


def degraded(axis: Axis, level: float, clean: np.ndarray, detail: np.ndarray, k: float, row,
             seed: int) -> np.ndarray:
    """The frame of `row` under `axis` at `level`, reproducible from `seed`. The first level is the clean frame."""
    if level == axis.levels[0]:
        return clean
    rng = np.random.default_rng(seed)
    if axis.geometric == "altitude":
        return render_frame(detail, k, row, clean.shape[1::-1], altitude=level)
    if axis.geometric == "tilt":
        t = 1e-4 * level
        ang = rng.uniform(0, 2 * np.pi)  # tilt direction differs per frame
        return render_frame(detail, k, row, clean.shape[1::-1], tilt=(t * np.cos(ang), t * np.sin(ang)))
    return axis.apply(clean, rng, level)
