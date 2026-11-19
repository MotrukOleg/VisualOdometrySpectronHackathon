"""Learned matchers.

* LightGlue (descriptor matcher on cached SuperPoint/DISK/ALIKED/SIFT features)
* LoFTR (kornia, detector-free)
* XFeat / XFeat* (torch.hub verlab/accelerated_features)
* SuperGlue (magicleap/SuperGluePretrainedNetwork cloned into third_party/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from ..features.base import Features
from ..io import resize_long
from ..preprocess import to_gray
from .base import DescriptorMatcher, ImagePairMatcher

ROOT = Path(__file__).resolve().parents[3]
LG_FEATURE_NAMES = {"superpoint": "superpoint", "disk": "disk", "aliked": "aliked", "lg_sift": "sift"}


class LightGlueMatcher(DescriptorMatcher):
    def __init__(self, feature_type: str, device: str = "cpu", params: dict | None = None):
        import torch

        try:
            from lightglue import LightGlue
        except ImportError as e:
            raise ImportError("pip install git+https://github.com/cvg/LightGlue.git") from e
        if feature_type not in LG_FEATURE_NAMES:
            raise ValueError(f"LightGlue supports features {sorted(LG_FEATURE_NAMES)}, got {feature_type!r}")
        self.torch = torch
        self.device = device
        self.model = LightGlue(features=LG_FEATURE_NAMES[feature_type], **(params or {})).eval().to(device)

    def _to_dict(self, f: Features) -> dict:
        t = self.torch
        d = {
            "keypoints": t.from_numpy(f.kpts)[None].to(self.device),
            "descriptors": t.from_numpy(f.desc)[None].to(self.device),
            "image_size": t.tensor([f.image_size], dtype=t.float32, device=self.device),
        }
        for k, v in f.extra.items():
            d[k] = t.from_numpy(np.ascontiguousarray(v))[None].to(self.device)
        return d

    def match(self, f0: Features, f1: Features):
        t = self.torch
        with t.inference_mode():
            out = self.model({"image0": self._to_dict(f0), "image1": self._to_dict(f1)})
        m = out["matches"][0].cpu().numpy().astype(int)
        s = out["scores"][0].cpu().numpy().astype(np.float32)
        return m.reshape(-1, 2), s


class _TorchImageMatcher(ImagePairMatcher):
    """Helper: optional internal downscale (dense matchers are memory-hungry on 4 GB GPUs)."""

    def __init__(self, device="cpu", resize_long_side=None, preprocess=None):
        super().__init__(preprocess)
        import torch

        self.torch = torch
        self.device = device
        self.resize_long_side = resize_long_side

    def _prep(self, img):
        small, s = resize_long(img, self.resize_long_side) if self.resize_long_side else (img, 1.0)
        return small, s


class LoFTRMatcher(_TorchImageMatcher):
    def __init__(self, device="cpu", resize_long_side=None, pretrained="outdoor", min_conf=0.2, preprocess=None):
        super().__init__(device, resize_long_side, preprocess)
        try:
            import kornia.feature as KF
        except ImportError as e:
            raise ImportError("LoFTR needs kornia: pip install kornia") from e
        self.model = KF.LoFTR(pretrained=pretrained).eval().to(device)
        self.min_conf = min_conf

    def _tensor(self, img):
        g = to_gray(img)
        h, w = g.shape
        g = g[: h - h % 8, : w - w % 8]  # LoFTR needs sizes divisible by 8
        return self.torch.from_numpy(g).float().div(255.0)[None, None].to(self.device)

    def match_images(self, img0, img1):
        a, s0 = self._prep(img0)
        b, s1 = self._prep(img1)
        with self.torch.inference_mode():
            out = self.model({"image0": self._tensor(a), "image1": self._tensor(b)})
        p0 = out["keypoints0"].cpu().numpy() / s0
        p1 = out["keypoints1"].cpu().numpy() / s1
        c = out["confidence"].cpu().numpy()
        keep = c >= self.min_conf
        return p0[keep], p1[keep], c[keep]


class XFeatMatcher(_TorchImageMatcher):
    """XFeat sparse (`xfeat`) or semi-dense XFeat* (`xfeat_star`)."""

    def __init__(self, device="cpu", resize_long_side=None, top_k=4096, star=False, preprocess=None):
        super().__init__(device, resize_long_side, preprocess)
        self.model = self.torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k, trust_repo=True
        )
        self.model.dev = self.torch.device(device)
        self.model.net = self.model.net.to(device)
        self.top_k = top_k
        self.star = star

    def match_images(self, img0, img1):
        a, s0 = self._prep(img0)
        b, s1 = self._prep(img1)
        a, b = cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            if self.star:
                p0, p1 = self.model.match_xfeat_star(a, b, top_k=self.top_k)
            else:
                p0, p1 = self.model.match_xfeat(a, b, top_k=self.top_k)
        p0, p1 = np.asarray(p0, np.float32) / s0, np.asarray(p1, np.float32) / s1
        return p0, p1, np.ones(len(p0), np.float32)


class SuperGlueMatcher(_TorchImageMatcher):
    """SuperPoint + SuperGlue from the official repo (non-commercial license).

    git clone https://github.com/magicleap/SuperGluePretrainedNetwork third_party/SuperGluePretrainedNetwork
    """

    def __init__(self, device="cpu", resize_long_side=None, weights="outdoor", max_keypoints=2048,
                 match_threshold=0.2, preprocess=None):
        super().__init__(device, resize_long_side, preprocess)
        repo = ROOT / "third_party" / "SuperGluePretrainedNetwork"
        if not repo.exists():
            raise ImportError(f"SuperGlue repo not found at {repo}; see the class docstring")
        sys.path.insert(0, str(repo))
        from models.matching import Matching  # type: ignore

        conf = {
            "superpoint": {"nms_radius": 4, "keypoint_threshold": 0.005, "max_keypoints": max_keypoints},
            "superglue": {"weights": weights, "sinkhorn_iterations": 20, "match_threshold": match_threshold},
        }
        self.model = Matching(conf).eval().to(device)

    def match_images(self, img0, img1):
        a, s0 = self._prep(img0)
        b, s1 = self._prep(img1)
        t = self.torch
        ta = t.from_numpy(to_gray(a)).float().div(255.0)[None, None].to(self.device)
        tb = t.from_numpy(to_gray(b)).float().div(255.0)[None, None].to(self.device)
        with t.inference_mode():
            pred = self.model({"image0": ta, "image1": tb})
        k0 = pred["keypoints0"][0].cpu().numpy()
        k1 = pred["keypoints1"][0].cpu().numpy()
        m = pred["matches0"][0].cpu().numpy()
        c = pred["matching_scores0"][0].cpu().numpy()
        valid = m > -1
        return k0[valid] / s0, k1[m[valid]] / s1, c[valid]
