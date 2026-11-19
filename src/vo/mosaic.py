"""Warping frames with their global poses into one canvas and blending them."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .geometry import apply

log = logging.getLogger(__name__)


def canvas_transform(poses, sizes, scale: float, max_canvas_px: int) -> tuple[np.ndarray, tuple[int, int]]:
    """C maps world coords -> canvas pixels; returns (C, (W, H))."""
    corners = np.concatenate(
        [apply(G, [[0, 0], [w, 0], [w, h], [0, h]]) for G, (w, h) in zip(poses, sizes)]
    )
    lo, hi = corners.min(0), corners.max(0)
    extent = (hi - lo) * scale
    if extent.max() > max_canvas_px:
        new = scale * max_canvas_px / extent.max()
        log.warning("mosaic scale %.3f -> %.3f to respect max_canvas_px=%d", scale, new, max_canvas_px)
        scale = new
    C = np.array([[scale, 0, -lo[0] * scale], [0, scale, -lo[1] * scale], [0, 0, 1]])
    W, H = np.ceil((hi - lo) * scale).astype(int) + 1
    return C, (int(W), int(H))


def _feather_weight(w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[1:-1, 1:-1] = 255
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return (d / max(d.max(), 1e-6)).astype(np.float32)


def build_mosaic(load, poses, sizes, scale=0.5, max_canvas_px=6000, blend="feather", every=1):
    """`load(k)` returns BGR frame k. Returns (mosaic BGR uint8, C world->canvas)."""
    C, (W, H) = canvas_transform(poses, sizes, scale, max_canvas_px)
    idx = list(range(0, len(poses), max(1, every)))
    if blend == "multiband":
        blender = cv2.detail_MultiBandBlender(0, 5)
        blender.prepare((0, 0, W, H))
    else:
        acc = np.zeros((H, W, 3), np.float32)
        wsum = np.zeros((H, W), np.float32)
    weight_cache: dict[tuple[int, int], np.ndarray] = {}

    for k in idx:
        img = load(k)
        h, w = img.shape[:2]
        M = C @ poses[k]
        box = apply(M, [[0, 0], [w, 0], [w, h], [0, h]])
        x0, y0 = np.floor(box.min(0)).astype(int).clip(0, [W - 1, H - 1])
        x1, y1 = np.ceil(box.max(0)).astype(int).clip(1, [W, H])
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1 or bh <= 1:
            continue
        L = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ M
        warped = cv2.warpPerspective(img, L, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if (w, h) not in weight_cache:
            weight_cache[(w, h)] = _feather_weight(w, h)
        wt = cv2.warpPerspective(weight_cache[(w, h)], L, (bw, bh), flags=cv2.INTER_LINEAR)
        mask = wt > 0
        if blend == "multiband":
            blender.feed(warped.astype(np.int16), (mask * 255).astype(np.uint8), (int(x0), int(y0)))
        elif blend == "none":
            acc[y0:y1, x0:x1][mask] = warped[mask]
            wsum[y0:y1, x0:x1][mask] = 1.0
        elif blend == "feather":
            acc[y0:y1, x0:x1] += warped.astype(np.float32) * wt[..., None]
            wsum[y0:y1, x0:x1] += wt
        else:
            raise ValueError(f"Unknown blend {blend!r}; options: none, feather, multiband")

    if blend == "multiband":
        out, _ = blender.blend(None, None)
        return np.clip(out, 0, 255).astype(np.uint8), C
    if blend == "none":
        return acc.astype(np.uint8), C
    out = acc / np.maximum(wsum, 1e-6)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), C


def overlap_photometric_error(load, poses, pairs, min_overlap: float = 0.1) -> float:
    """Mean absolute grey-level difference in the overlap of frame pairs warped by the *global* poses.

    A ground-truth-free consistency metric: drift and wrong registrations increase it.
    """
    errs = []
    for i, j in pairs:
        a = cv2.cvtColor(load(i), cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(load(j), cv2.COLOR_BGR2GRAY).astype(np.float32)
        T = np.linalg.inv(poses[i]) @ poses[j]  # j -> i
        h, w = a.shape
        bw = cv2.warpPerspective(b, T, (w, h), flags=cv2.INTER_LINEAR)
        m = cv2.warpPerspective(np.ones_like(b), T, (w, h), flags=cv2.INTER_NEAREST) > 0
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if m.mean() < min_overlap:
            continue
        errs.append(float(np.abs(a[m] - bw[m]).mean()))
    return float(np.mean(errs)) if errs else float("nan")
