from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from ..features.base import Extractor, Features


@dataclass
class Correspondences:
    pts0: np.ndarray  # (K, 2) float32 in frame i
    pts1: np.ndarray  # (K, 2) float32 in frame j
    conf: np.ndarray  # (K,)
    n_kpts0: int = 0
    n_kpts1: int = 0
    timings: dict = field(default_factory=dict)  # seconds: extract, match

    def __len__(self) -> int:
        return len(self.pts0)


class DescriptorMatcher(ABC):
    """Matches two precomputed feature sets. Returns (K, 2) index pairs and (K,) confidences."""

    @abstractmethod
    def match(self, f0: Features, f1: Features) -> tuple[np.ndarray, np.ndarray]: ...


class PairMatcher(ABC):
    """Common interface used by the pipeline: correspondences between two frames."""

    name: str = "pair"

    @abstractmethod
    def match(self, key0: int, img0: np.ndarray, key1: int, img1: np.ndarray) -> Correspondences: ...


class FeatureMatcher(PairMatcher):
    """Detector/descriptor + descriptor matcher. Features are computed once per frame and cached."""

    def __init__(self, extractor: Extractor, matcher: DescriptorMatcher, preprocess=None, cache_size: int = 512):
        self.extractor = extractor
        self.matcher = matcher
        self.preprocess = preprocess
        self.name = f"{extractor.name}+{type(matcher).__name__}"
        self._cache: OrderedDict[int, Features] = OrderedDict()
        self._cache_size = cache_size
        self.extract_times: list[float] = []

    def features(self, key: int, img: np.ndarray) -> Features:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        t0 = time.perf_counter()
        f = self.extractor.extract(self.preprocess(img) if self.preprocess else img)
        self.extract_times.append(time.perf_counter() - t0)
        self._cache[key] = f
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return f

    def match(self, key0, img0, key1, img1) -> Correspondences:
        f0, f1 = self.features(key0, img0), self.features(key1, img1)
        t0 = time.perf_counter()
        if len(f0) < 2 or len(f1) < 2:
            m, c = np.zeros((0, 2), int), np.zeros(0, np.float32)
        else:
            m, c = self.matcher.match(f0, f1)
        t_match = time.perf_counter() - t0
        return Correspondences(
            f0.kpts[m[:, 0]].astype(np.float32),
            f1.kpts[m[:, 1]].astype(np.float32),
            c.astype(np.float32),
            len(f0),
            len(f1),
            {"match": t_match},
        )


class ImagePairMatcher(PairMatcher):
    """Detector-free / end-to-end matchers operating on the two images directly."""

    def __init__(self, preprocess=None):
        self.preprocess = preprocess

    @abstractmethod
    def match_images(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns pts0 (K,2), pts1 (K,2), conf (K,)."""

    def match(self, key0, img0, key1, img1) -> Correspondences:
        if self.preprocess:
            img0, img1 = self.preprocess(img0), self.preprocess(img1)
        t0 = time.perf_counter()
        p0, p1, c = self.match_images(img0, img1)
        t = time.perf_counter() - t0
        return Correspondences(p0.astype(np.float32), p1.astype(np.float32), c.astype(np.float32),
                               len(p0), len(p1), {"match": t})
