"""Synthetic nadir flight over a procedural 'aerial' texture, with exact ground truth.

Used to test the pipeline end-to-end before the real dataset is available.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .degrade import degrade
from .io import imread, imwrite


def _noise(h, w, cell, rng, ch=3):
    small = rng.random((h // cell + 2, w // cell + 2, ch)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def make_world(width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H, W = height, width
    base = 0.55 * _noise(H, W, 400, rng) + 0.3 * _noise(H, W, 60, rng) + 0.15 * _noise(H, W, 12, rng)
    palette = np.array([[60, 120, 70], [80, 150, 110], [70, 110, 140], [90, 140, 160], [50, 90, 60]], np.float32)
    w_idx = (base[..., 0] * len(palette)).clip(0, len(palette) - 1e-3)
    img = palette[w_idx.astype(int)] * (0.7 + 0.6 * base[..., 1:2])
    img = img.astype(np.uint8)

    # fields with stripe texture
    for _ in range(W * H // 250_000):
        c = rng.integers([0, 0], [W, H])
        pts = (c + rng.normal(0, 180, (4, 2))).astype(np.int32)
        col = tuple(int(v) for v in rng.integers(40, 200, 3))
        cv2.fillPoly(img, [pts], col)
        x0, y0 = np.maximum(pts.min(0), 0)
        x1, y1 = np.minimum(pts.max(0) + 1, [W, H])
        if x1 <= x0 or y1 <= y0:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [pts - [x0, y0]], 255)
        stripes = np.zeros_like(mask)
        ang = rng.uniform(0, np.pi)
        nrm = np.array([np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2)])
        d = 600 * np.array([np.cos(ang), np.sin(ang)])
        mid = np.array([(x1 - x0) / 2, (y1 - y0) / 2])
        for s in range(-600, 600, int(rng.integers(6, 14))):
            p0 = mid + s * nrm
            cv2.line(stripes, tuple((p0 - d).astype(int)), tuple((p0 + d).astype(int)), 255, 2)
        roi = img[y0:y1, x0:x1]
        roi[(mask > 0) & (stripes > 0)] = np.clip(np.array(col) * 0.8, 0, 255).astype(np.uint8)
    # roads
    for _ in range(max(4, W // 500)):
        pts = np.cumsum(rng.normal(0, 250, (8, 2)), 0) + rng.integers([0, 0], [W, H])
        cv2.polylines(img, [pts.astype(np.int32)], False, (105, 105, 110), int(rng.integers(8, 22)), cv2.LINE_AA)
    # buildings with shadows
    for _ in range(W * H // 20_000):
        c = rng.integers([0, 0], [W, H]).astype(float)
        size = rng.uniform(12, 60, 2)
        ang = rng.uniform(0, 180)
        box = cv2.boxPoints(((c[0], c[1]), (size[0], size[1]), ang)).astype(np.int32)
        cv2.fillPoly(img, [box + 6], (40, 40, 40))
        cv2.fillPoly(img, [box], tuple(int(v) for v in rng.integers(90, 255, 3)))
    # trees
    for _ in range(W * H // 6_000):
        c = tuple(int(v) for v in rng.integers([0, 0], [W, H]))
        cv2.circle(img, c, int(rng.integers(3, 9)), (int(rng.integers(20, 50)), int(rng.integers(60, 110)), 30), -1)
    img = img.astype(np.float32) + rng.normal(0, 6, img.shape)
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)


def frame_to_world(cx, cy, yaw_deg, scale, w, h, tilt=(0.0, 0.0)) -> np.ndarray:
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    S = np.diag([scale, scale, 1.0])
    Tc = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
    P = np.array([[1, 0, 0], [0, 1, 0], [tilt[0], tilt[1], 1]])  # perspective in centred coords
    Tw = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    return Tw @ R @ S @ P @ Tc


def make_flight(out_dir: str | Path, n_frames: int = 150, frame_size=(640, 480), world_size=(4000, 3000),
                turns: float = 1.15, tilt_std: float = 0.0, noise: float = 3.0, seed: int = 0,
                world_path: str | Path | None = None, frame_scale: float = 1.0, degrade_preset: str = "none") -> Path:
    """Elliptic flight (> 1 turn -> loop closure) with the camera yawing along the heading.

    world_path:  use a real image (e.g. data/train/global_map.png) instead of the procedural texture;
                 then GT is in the map pixel frame, which allows exact evaluation of map localisation.
    frame_scale: world pixels per frame pixel (0.25 = frames are 4x zoomed-in w.r.t. the map,
                 as in the provided training crops).

    Writes `frames/*.jpg`, `gt.csv` (frame, name, x, y, yaw_deg, scale) and `world.jpg`.
    """
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = frame_size
    if world_path:
        world = imread(world_path)
        H, W = world.shape[:2]
    else:
        W, H = world_size
        world = make_world(W, H, seed)
        imwrite(out / "world.jpg", world, [1, 90])
    t = np.linspace(0, 2 * np.pi * turns, n_frames)
    rx, ry = 0.34 * W, 0.30 * H
    wob = 0.01 * min(W, H)
    cx = W / 2 + rx * np.cos(t) + 4 * wob * np.sin(5 * t)
    cy = H / 2 + ry * np.sin(t) + 3 * wob * np.cos(3 * t)
    heading = np.degrees(np.arctan2(np.gradient(cy), np.gradient(cx)))
    yaw = heading + 90 + rng.normal(0, 2, n_frames)  # camera 'up' along the flight direction
    scale = frame_scale * (1.0 + 0.08 * np.sin(2.3 * t))  # altitude changes
    rows = []
    for k in range(n_frames):
        tilt = tuple(rng.normal(0, tilt_std, 2)) if tilt_std > 0 else (0.0, 0.0)
        M = frame_to_world(cx[k], cy[k], yaw[k], scale[k], w, h, tilt)
        fr = cv2.warpPerspective(world, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
        fr = fr.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.normal(0, noise, fr.shape)
        fr = np.clip(fr, 0, 255).astype(np.uint8)
        if degrade_preset != "none":
            fr = degrade(fr, rng, degrade_preset)
        name = f"frame_{k:05d}"
        imwrite(out / "frames" / f"{name}.jpg", fr, [1, 92])
        c = M @ np.array([w / 2, h / 2, 1.0])
        rows.append({"frame": k, "name": name, "x": c[0] / c[2], "y": c[1] / c[2],
                     "yaw_deg": yaw[k], "scale": scale[k]})
    pd.DataFrame(rows).to_csv(out / "gt.csv", index=False)
    return out
