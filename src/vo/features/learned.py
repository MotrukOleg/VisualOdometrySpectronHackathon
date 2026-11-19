"""Learned local features through the `lightglue` package: SuperPoint, DISK, ALIKED, SIFT."""
from __future__ import annotations

import cv2
import numpy as np

from .base import Extractor, Features

LG_EXTRACTORS = {"superpoint": "SuperPoint", "disk": "DISK", "aliked": "ALIKED", "lg_sift": "SIFT"}


def _import_lightglue():
    try:
        import lightglue
    except ImportError as e:
        raise ImportError(
            "Learned features need the `lightglue` package: "
            "pip install git+https://github.com/cvg/LightGlue.git"
        ) from e
    return lightglue


class LightGlueExtractor(Extractor):
    """Wraps lightglue.{SuperPoint,DISK,ALIKED,SIFT}. Weights are downloaded on first use."""

    def __init__(self, kind: str, max_keypoints: int = 2048, device: str = "cpu", params: dict | None = None):
        import torch

        lg = _import_lightglue()
        self.name = kind
        self.kind = kind
        self.device = device
        self.torch = torch
        cls = getattr(lg, LG_EXTRACTORS[kind])
        conf = {"max_num_keypoints": max_keypoints}
        conf.update(params or {})
        self.model = cls(**conf).eval().to(device)

    def extract(self, img_bgr: np.ndarray) -> Features:
        torch = self.torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.inference_mode():
            feats = self.model.extract(t, resize=None)  # frames are already resized
        h, w = img_bgr.shape[:2]
        out = {k: v[0].detach().cpu().numpy() for k, v in feats.items() if torch.is_tensor(v)}
        kpts = out.pop("keypoints").astype(np.float32)
        desc = out.pop("descriptors").astype(np.float32)
        scores = out.pop("keypoint_scores", None)
        out.pop("image_size", None)
        return Features(kpts, desc, scores, (w, h), "float", extra=out)
