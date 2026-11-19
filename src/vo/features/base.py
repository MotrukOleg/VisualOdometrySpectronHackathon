from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Features:
    kpts: np.ndarray  # (N, 2) float32, pixel coords in the (resized) frame
    desc: np.ndarray | None  # (N, D) float32 or uint8 (binary)
    scores: np.ndarray | None  # (N,)
    image_size: tuple[int, int]  # (width, height)
    kind: str = "float"  # "float" | "binary"
    extra: dict = field(default_factory=dict)  # method-specific arrays (e.g. scales, oris)

    def __len__(self) -> int:
        return len(self.kpts)


class Extractor(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, img_bgr: np.ndarray) -> Features: ...
