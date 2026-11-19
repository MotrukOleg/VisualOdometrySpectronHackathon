"""Classical descriptor matchers: brute force and FLANN, with Lowe ratio test and mutual check."""
from __future__ import annotations

import cv2
import numpy as np

from ..features.base import Features
from .base import DescriptorMatcher


def _ratio_filter(knn, ratio: float | None) -> dict[int, tuple[int, float]]:
    """query idx -> (train idx, confidence)."""
    out = {}
    for pair in knn:
        if not pair:
            continue
        m = pair[0]
        if ratio is not None:
            if len(pair) < 2:
                continue
            n = pair[1]
            if m.distance >= ratio * n.distance:
                continue
            conf = 1.0 - m.distance / max(n.distance, 1e-9)
        else:
            conf = 1.0 / (1.0 + m.distance)
        out[m.queryIdx] = (m.trainIdx, conf)
    return out


class _KnnMatcher(DescriptorMatcher):
    def __init__(self, ratio: float | None = 0.75, cross_check: bool = False):
        self.ratio = ratio
        self.cross_check = cross_check

    def _engine(self, f: Features):
        raise NotImplementedError

    def _knn(self, eng, d0, d1):
        k = 2 if self.ratio is not None else 1
        return eng.knnMatch(d0, d1, k=k)

    def match(self, f0: Features, f1: Features):
        eng = self._engine(f0)
        fwd = _ratio_filter(self._knn(eng, f0.desc, f1.desc), self.ratio)
        if self.cross_check:
            bwd = _ratio_filter(self._knn(eng, f1.desc, f0.desc), self.ratio)
            fwd = {q: (t, c) for q, (t, c) in fwd.items() if bwd.get(t, (-1,))[0] == q}
        if not fwd:
            return np.zeros((0, 2), int), np.zeros(0, np.float32)
        q = np.fromiter(fwd.keys(), int)
        t = np.array([v[0] for v in fwd.values()], int)
        c = np.array([v[1] for v in fwd.values()], np.float32)
        return np.stack([q, t], 1), c


class BFMatcher(_KnnMatcher):
    def _engine(self, f: Features):
        norm = cv2.NORM_HAMMING if f.kind == "binary" else cv2.NORM_L2
        return cv2.BFMatcher(norm, crossCheck=False)


class FLANNMatcher(_KnnMatcher):
    """KD-tree for float descriptors, LSH for binary ones."""

    def __init__(self, ratio=0.75, cross_check=False, trees: int = 5, checks: int = 64):
        super().__init__(ratio, cross_check)
        self.trees, self.checks = trees, checks

    def _engine(self, f: Features):
        if f.kind == "binary":
            index = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # FLANN_INDEX_LSH
        else:
            index = dict(algorithm=1, trees=self.trees)  # FLANN_INDEX_KDTREE
        return cv2.FlannBasedMatcher(index, dict(checks=self.checks))

    def _knn(self, eng, d0, d1):
        if d0.dtype != np.uint8:
            d0, d1 = d0.astype(np.float32), d1.astype(np.float32)
        return super()._knn(eng, d0, d1)
