from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class Preprocessor:
    """Image enhancement applied before feature extraction to BOTH the map and the frames (not to the mosaic)."""

    def __init__(self, clahe: bool = False, clahe_clip: float = 2.0, clahe_grid: int = 8, **_frame_only):
        self.clahe = cv2.createCLAHE(clahe_clip, (clahe_grid, clahe_grid)) if clahe else None

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.clahe is None:
            return img_bgr
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = self.clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def antimoire(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass against the screen-pixel moire pattern of videos filmed from a monitor."""
    if not sigma or sigma <= 0:
        return img_bgr
    return cv2.GaussianBlur(img_bgr, (0, 0), sigma)


def screen_mask(img_bgr: np.ndarray, dark_thr: int = 40, min_area: float = 0.002) -> np.ndarray:
    """True where the pixel is NOT map content: dark monitor bezel / desk regions.

    Large connected dark regions (the bezel is near-black, unlike shadows inside the map, which are small)
    are dilated so that keypoints on the bezel edge are rejected too.
    """
    g = to_gray(img_bgr)
    dark = (cv2.medianBlur(g, 5) < dark_thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > min_area * g.size
    mask = keep[lab].astype(np.uint8)
    if mask.any():
        r = max(3, int(0.01 * max(g.shape)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return mask.astype(bool)


def drop_masked(pts: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Boolean keep-flags for points (x, y) that are outside the mask."""
    if mask is None or len(pts) == 0:
        return np.ones(len(pts), bool)
    h, w = mask.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return ~mask[y, x]
