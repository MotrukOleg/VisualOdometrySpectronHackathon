"""OpenCV detectors/descriptors: SIFT, RootSIFT, ORB, AKAZE, BRISK, KAZE."""
from __future__ import annotations

import cv2
import numpy as np

from ..preprocess import to_gray
from .base import Extractor, Features

BINARY = {"orb", "akaze", "brisk"}


def _create(kind: str, max_kp: int, params: dict):
    if kind in ("sift", "rootsift"):
        return cv2.SIFT_create(nfeatures=max_kp, **params)
    if kind == "orb":
        p = {"scaleFactor": 1.2, "nlevels": 8, "fastThreshold": 10}
        p.update(params)
        return cv2.ORB_create(nfeatures=max_kp, **p)
    if kind == "akaze":
        return cv2.AKAZE_create(**params)
    if kind == "brisk":
        return cv2.BRISK_create(**params)
    if kind == "kaze":
        return cv2.KAZE_create(**params)
    raise ValueError(f"Unknown classical feature type: {kind}")


class ClassicalExtractor(Extractor):
    def __init__(self, kind: str, max_keypoints: int = 4000, params: dict | None = None):
        self.name = kind
        self.kind = kind
        self.max_kp = max_keypoints
        self.det = _create(kind, max_keypoints, params or {})

    def extract(self, img_bgr: np.ndarray) -> Features:
        gray = to_gray(img_bgr)
        kps, desc = self.det.detectAndCompute(gray, None)
        h, w = gray.shape[:2]
        binary = self.kind in BINARY
        if not kps or desc is None:
            d = 32 if binary else 128
            empty = np.zeros((0, d), np.uint8 if binary else np.float32)
            return Features(np.zeros((0, 2), np.float32), empty, np.zeros(0, np.float32), (w, h),
                            "binary" if binary else "float")
        resp = np.array([k.response for k in kps], np.float32)
        if len(kps) > self.max_kp:  # AKAZE/BRISK/KAZE have no nfeatures limit
            keep = np.argsort(-resp)[: self.max_kp]
            kps = [kps[i] for i in keep]
            desc, resp = desc[keep], resp[keep]
        pts = np.array([k.pt for k in kps], np.float32)
        if self.kind == "rootsift":
            desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
            desc = np.sqrt(desc)
        if not binary:
            desc = desc.astype(np.float32)
        extra = {
            "scales": np.array([k.size for k in kps], np.float32),
            "oris": np.deg2rad(np.array([k.angle for k in kps], np.float32)),
        }
        return Features(pts, desc, resp, (w, h), "binary" if binary else "float", extra)
