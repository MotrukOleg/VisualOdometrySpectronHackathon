"""Ground-truth-free quality checks of a localisation run.

1. Frame-to-frame consistency: the relative homography of consecutive frames, estimated independently
   with SIFT, must agree with the two map registrations (H_i @ T_ij ~ H_j). Disagreement in map px.
2. Agreement with a reference trajectory (e.g. the SIFT whole-map baseline): distance between centres.

The frame-to-frame homographies depend only on the dataset, so they are computed once and cached.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import load_config
from .geometry import apply, estimate_transform
from .io import FrameSource
from .matching import build_pair_matcher
from .schema import corners_of

F2F_GEOM = {"model": "homography", "estimator": "magsac", "reproj_threshold": 3.0, "min_inliers": 15,
            "min_scale": 0.3, "max_scale": 3.0, "max_perspective": 0.01}


# Matchers for the frame-to-frame reference. A localisation run checked against f2f of the SAME matcher shares its
# errors (they cancel), so "cross" consistency averages the check over matchers of different families.
F2F_MATCHERS = {
    "sift": ["features.type=sift", "matching.type=flann", "matching.ratio=0.75"],
    "splg": ["features.type=superpoint", "features.max_keypoints=2048", "matching.type=lightglue", "matching.ratio=null"],
    "loftr": ["matching.type=loftr"],
}


def frame_to_frame(frames_dir: str | Path, cache: str | Path, frame_scale: float = 0.35, step: int = 1,
                   matcher: str = "sift") -> dict:
    """{(i, i+1): T (frame i+1 -> frame i, ORIGINAL px)} for consecutive frames of the (sub-sampled) sequence;
    indices refer to the sequence after `step`. `matcher` is a key of F2F_MATCHERS. Cached in an .npz file."""
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache)
        return {(int(i), int(j)): T for i, j, T in zip(z["i"], z["j"], z["T"])}
    frames_dir = str(frames_dir).split("#")[0]
    src = FrameSource(frames_dir, step=step, scale=frame_scale)
    cfg = load_config(None, [f"data.frames={frames_dir}"] + F2F_MATCHERS[matcher])
    device = "cpu"
    if matcher != "sift":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = build_pair_matcher(cfg, device)
    S = np.diag([frame_scale, frame_scale, 1.0])
    out = {}
    for i in range(len(src) - 1):
        c = matcher.match(i, src.load(i), i + 1, src.load(i + 1))
        est = estimate_transform(c.pts0, c.pts1, F2F_GEOM, src.size(i + 1))
        if est.ok:
            out[(i, i + 1)] = np.linalg.inv(S) @ est.T @ S
    cache.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(out)
    np.savez(cache, i=[k[0] for k in keys], j=[k[1] for k in keys], T=np.stack([out[k] for k in keys]))
    return out


def consistency(traj: pd.DataFrame, f2f: dict, frame_size: tuple[int, int]) -> dict:
    w, h = frame_size
    cs = corners_of(traj)
    st = traj["status"].to_numpy()
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = []
    for (i, j), T in f2f.items():
        if j >= len(st) or st[i] != "map" or st[j] != "map":  # run may cover only a prefix of the frames
            continue
        Hi = cv2.getPerspectiveTransform(src, cs[i].astype(np.float32))
        p = apply(Hi @ T, [[w / 2, h / 2]])[0]
        d.append(np.linalg.norm(p - traj[["x", "y"]].iloc[j].to_numpy(float)))
    d = np.array(d)
    if not len(d):
        return {"cons_pairs": 0}
    return {"cons_pairs": len(d), "cons_median_px": float(np.median(d)), "cons_p95_px": float(np.percentile(d, 95)),
            "cons_max_px": float(d.max())}


def agreement(traj: pd.DataFrame, ref: pd.DataFrame) -> dict:
    ref = ref.set_index("name").reindex(traj["name"])  # align by frame name (runs may cover a subset)
    a = traj[["x", "y"]].to_numpy(float)
    b = ref[["x", "y"]].to_numpy(float)
    ok = np.all(np.isfinite(a), 1) & np.all(np.isfinite(b), 1) & traj["status"].isin(["map", "interp"]).to_numpy()
    if not ok.any():
        return {}
    e = np.linalg.norm(a[ok] - b[ok], axis=1)
    return {"ref_dev_median_px": float(np.median(e)), "ref_dev_p95_px": float(np.percentile(e, 95)),
            "ref_dev_max_px": float(e.max()), "ref_far_frames": int((e > 25).sum())}
