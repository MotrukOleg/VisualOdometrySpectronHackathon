from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class Preprocessor:
    """Image enhancement applied before feature extraction (not to the mosaic)."""

    def __init__(self, clahe: bool = False, clahe_clip: float = 2.0, clahe_grid: int = 8):
        self.clahe = cv2.createCLAHE(clahe_clip, (clahe_grid, clahe_grid)) if clahe else None

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.clahe is None:
            return img_bgr
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = self.clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
