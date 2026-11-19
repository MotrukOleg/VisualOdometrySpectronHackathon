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


class DeDoDeExtractor(Extractor):
    """DeDoDe v2 (kornia): detector 'L-C4-v2' + descriptor 'B-upright' / 'G-upright'. Match with mutual NN."""

    def __init__(self, max_keypoints: int = 4000, device: str = "cpu", params: dict | None = None):
        import torch
        import kornia.feature as KF

        p = {"detector_weights": "L-C4-v2", "descriptor_weights": "B-upright"}
        p.update(params or {})
        self.name = "dedode"
        self.torch = torch
        self.device = device
        self.n = max_keypoints
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = KF.DeDoDe.from_pretrained(p["detector_weights"], p["descriptor_weights"], amp_dtype=dtype)
        self.model = self.model.eval().to(device)

    def extract(self, img_bgr: np.ndarray) -> Features:
        torch = self.torch
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)[None].to(self.device)
        with torch.inference_mode():
            kpts, scores, desc = self.model(t, n=self.n)
        k = kpts[0].float().cpu().numpy()
        if np.abs(k).max() <= 1.5:  # normalised coordinates -> pixels
            k = (k + 1) * np.array([w, h]) / 2 - 0.5
        d = desc[0].float().cpu().numpy()
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
        return Features(k.astype(np.float32), d.astype(np.float32), scores[0].float().cpu().numpy(), (w, h), "float")
