"""Robust estimation of the relative 2D transform between two frames.

Convention: T_ij maps pixel coordinates of frame j into frame i (x_i ~ T_ij @ x_j).
Global poses G_k map frame k into the world frame (= frame 0), so G_j = G_i @ T_ij.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ESTIMATORS = {
    "ransac": cv2.RANSAC,
    "lmeds": cv2.LMEDS,
    "magsac": cv2.USAC_MAGSAC,
    "usac_accurate": cv2.USAC_ACCURATE,
    "usac_default": cv2.USAC_DEFAULT,
}
MODELS = ("similarity", "affine", "homography")


@dataclass
class Estimate:
    T: np.ndarray | None  # 3x3
    inliers: np.ndarray  # bool mask over the input correspondences
    n_inliers: int
    reproj_err: float  # mean error of inliers, px
    ok: bool
    reason: str = ""


def to3x3(M: np.ndarray) -> np.ndarray:
    if M.shape == (3, 3):
        return M.astype(np.float64)
    out = np.eye(3)
    out[:2] = M
    return out


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ T.T
    return h[:, :2] / h[:, 2:3]


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (Umeyama) mapping src -> dst, returned as 3x3."""
    from .eval import umeyama

    s, R, t = umeyama(src, dst, with_scale=True)
    T = np.eye(3)
    T[:2, :2] = s * R
    T[:2, 2] = t
    return T


def decompose(T: np.ndarray) -> dict:
    A = T[:2, :2]
    return {
        "tx": float(T[0, 2]),
        "ty": float(T[1, 2]),
        "rot_deg": float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))),
        "scale": float(np.sqrt(abs(np.linalg.det(A)))),
    }


def sanity_check(T: np.ndarray, size: tuple[int, int], cfg: dict) -> str:
    """Empty string if the transform is plausible for consecutive nadir frames."""
    if not np.all(np.isfinite(T)):
        return "non-finite"
    det = np.linalg.det(T[:2, :2])
    if det <= 0:
        return "reflection"
    s = np.sqrt(det)
    if not (cfg.get("min_scale", 0.5) <= s <= cfg.get("max_scale", 2.0)):
        return f"scale={s:.2f}"
    if max(abs(T[2, 0]), abs(T[2, 1])) > cfg.get("max_perspective", 0.002):
        return "perspective"
    w, h = size
    corners = apply(T, [[0, 0], [w, 0], [w, h], [0, h]]).astype(np.float32)
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        return "non-convex"
    return ""


def estimate_transform(pts0: np.ndarray, pts1: np.ndarray, cfg: dict, size: tuple[int, int]) -> Estimate:
    """Estimate T mapping pts1 (frame j) -> pts0 (frame i) with the configured model/estimator."""
    n = len(pts0)
    min_inl = cfg.get("min_inliers", 15)
    empty = np.zeros(n, bool)
    if n < max(min_inl, 4):
        return Estimate(None, empty, 0, np.nan, False, f"too few matches ({n})")

    model = cfg.get("model", "homography")
    method = ESTIMATORS[cfg.get("estimator", "ransac")]
    thr = float(cfg.get("reproj_threshold", 3.0))
    conf = float(cfg.get("confidence", 0.999))
    iters = int(cfg.get("max_iters", 5000))
    src, dst = pts1.astype(np.float32), pts0.astype(np.float32)

    if model == "homography":
        M, mask = cv2.findHomography(src, dst, method, thr, maxIters=iters, confidence=conf)
    elif model == "affine":
        M, mask = cv2.estimateAffine2D(src, dst, method=method, ransacReprojThreshold=thr,
                                       maxIters=iters, confidence=conf)
    elif model == "similarity":
        if method in (cv2.RANSAC, cv2.LMEDS):
            M, mask = cv2.estimateAffinePartial2D(src, dst, method=method, ransacReprojThreshold=thr,
                                                  maxIters=iters, confidence=conf)
        else:
            # OpenCV has no USAC similarity: select inliers with a USAC affine fit, then refit.
            M, mask = cv2.estimateAffine2D(src, dst, method=method, ransacReprojThreshold=thr,
                                           maxIters=iters, confidence=conf)
            if M is not None:
                inl = mask.ravel().astype(bool)
                if inl.sum() >= 3:
                    S = fit_similarity(src[inl], dst[inl])
                    err = np.linalg.norm(apply(S, src) - dst, axis=1)
                    M, mask = S, (err < thr).astype(np.uint8)
    else:
        raise ValueError(f"Unknown motion model {model!r}; options: {MODELS}")

    if M is None or mask is None:
        return Estimate(None, empty, 0, np.nan, False, "estimation failed")
    T = to3x3(M)
    inl = mask.ravel().astype(bool)
    k = int(inl.sum())
    err = float(np.linalg.norm(apply(T, src[inl]) - dst[inl], axis=1).mean()) if k else np.nan
    if k < min_inl:
        return Estimate(T, inl, k, err, False, f"too few inliers ({k})")
    reason = sanity_check(T, size, cfg)
    return Estimate(T, inl, k, err, not reason, reason)
