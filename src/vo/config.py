"""Experiment configuration: YAML files with defaults, inheritance (`base:`) and CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "unnamed",
    "device": "auto",  # auto | cuda | cpu
    "seed": 0,
    "data": {
        "frames": "data/synthetic/frames",  # directory with images
        "gt": None,  # optional CSV with columns frame|name, x, y
        "start": 0,
        "step": 1,  # take every k-th frame
        "max_frames": None,
        "resize_long": 1280,  # cap on the longer image side (only downscales; None = keep)
        "scale": 1.0,  # extra explicit scale factor (e.g. 0.25 to match the map resolution)
    },
    "preprocess": {
        "clahe": False,  # map + frames
        "clahe_clip": 2.0,
        "clahe_grid": 8,
        "blur_sigma": 0.0,  # frames only: Gaussian low-pass against screen moire (px of the prepared frame)
        "screen_mask": False,  # frames only: drop matches on the dark monitor bezel / desk
    },
    "features": {
        "type": "sift",  # sift|rootsift|orb|akaze|brisk|kaze|superpoint|disk|aliked|lg_sift
        "max_keypoints": 4000,
        "params": {},  # passed to the OpenCV/LightGlue constructor
    },
    "matching": {
        "type": "flann",  # bf|flann|lightglue|loftr|xfeat|xfeat_star|superglue
        "ratio": 0.75,  # Lowe ratio test (None = disabled)
        "cross_check": False,  # mutual nearest neighbours
        "resize_long": None,  # extra downscale for dense matchers (memory)
        "params": {},
    },
    "geometry": {
        "model": "homography",  # similarity | affine | homography
        "estimator": "ransac",  # ransac | magsac | lmeds | usac_accurate | usac_default
        "reproj_threshold": 3.0,
        "confidence": 0.999,
        "max_iters": 5000,
        "min_inliers": 15,
        "min_scale": 0.5,  # sanity limits for a relative transform
        "max_scale": 2.0,
        "max_perspective": 0.002,  # max |h31|,|h32| (in 1/px)
    },
    "graph": {
        "fallback_skips": [2, 3],  # if i-1 -> i fails, try i-k -> i
        "extra_skips": [],  # always add i-k -> i edges (used by the optimiser)
        "loop_closure": False,
        "loop_min_gap": 20,  # frames between loop candidates
        "loop_radius": 0.5,  # candidate radius as a fraction of the frame diagonal
        "loop_stride": 2,  # check every k-th frame for loops
        "loop_max_candidates": 200,
        "loop_min_inliers": 40,
        "optimize": False,
        "opt_points_per_edge": 40,
        "opt_loss": "huber",
        "opt_f_scale": 2.0,
    },
    "mosaic": {
        "enabled": True,
        "scale": 0.5,  # mosaic resolution relative to the resized frames
        "max_canvas_px": 6000,  # longer canvas side limit
        "blend": "feather",  # none | feather | multiband
        "every": 1,  # use every k-th frame in the mosaic
    },
    "eval": {
        "rpe_delta": 10,  # frames
        "photometric_every": 5,
        "success_thresholds": [5, 10, 25],  # map px, for frame-to-map localisation
    },
    # ---- frame-to-map localisation (Standard / Advanced level) ----
    "map": {
        "path": "data/train/global_map.png",
        "scale": 1.0,  # downscale of the map before feature extraction
        "max_keypoints": 60000,  # keypoint budget for the whole map
    },
    "localize": {
        # global: every frame vs the whole map (one FLANN index; classical descriptors only)
        # track:  frame vs a map ROI around the predicted position (any PairMatcher incl. learned / dense);
        #         the first frame and frames after a loss are initialised by the global SIFT search
        # tiles:  frame vs every overlapping map tile, best tile wins (search-strategy ablation)
        "mode": "global",
        "frame_scale": 0.5,  # frames are resized before matching (map is ~4x coarser than the crops)
        "min_inliers": 20,
        "reproj_threshold": 5.0,  # map px (oblique views -> homography is approximate)
        "min_scale": 0.02,  # sanity limits for the frame->map scale (map px per frame px)
        "max_scale": 5.0,
        "max_perspective": 0.01,
    },
    "tracking": {
        "roi_factor": 1.6,  # ROI side = factor * predicted footprint diagonal (map px)
        "roi_retry_factor": 3.0,  # second attempt with a larger ROI before global re-initialisation
        "min_roi": 256,
        "work_scale": 1.0,  # matching resolution relative to the map (2 = map ROI and frame upsampled x2)
        "auto_scale": True,  # resize the frame with the predicted scale so it matches the map resolution
        "rotation": "predicted",  # none | predicted (derotate by predicted heading) | test (try 4 rotations)
        "motion_model": "constant_velocity",  # or 'static'
        "tile_factor": 1.6,  # tiles mode: tile side = factor * footprint diagonal, stride = side / 2
    },
    "smoothing": {
        "outlier_factor": 4.0,  # a fix is an outlier if it is > factor * median step from its neighbours
        "outlier_min_px": 50.0,
        "interpolate": True,  # fill failed / rejected frames
        "savgol_window": 0,  # Savitzky-Golay window (odd, frames); 0 = no smoothing
        "savgol_order": 2,
    },
    "viz": {
        "footprints": True,  # draw the outline of every frame on the map
        "label_every": 0,  # 0 = auto
        "zoom_margin": 150,  # px around the route for route_zoom.png
    },
    "output": {
        "dir": "experiments/results",
        "save_mosaic": True,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(_load_yaml((path.parent / base).resolve()), cfg)
    return cfg


def parse_override(item: str) -> dict:
    """'a.b.c=value' -> {'a': {'b': {'c': value}}}; the value is parsed as YAML."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"Override must look like key=value, got {item!r}")
    value = yaml.safe_load(raw)
    out: dict = {}
    node = out
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        cfg = deep_merge(cfg, _load_yaml(Path(path).resolve()))
    for item in overrides or []:
        cfg = deep_merge(cfg, parse_override(item))
    return cfg


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
