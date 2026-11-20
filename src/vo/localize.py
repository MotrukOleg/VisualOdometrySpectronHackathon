"""Frame-to-map localisation: register every UAV frame to the global satellite map.

Baseline (Standard level): local features of the whole map are extracted once; every frame is
resized (`localize.frame_scale`), its features are matched against the map (FLANN / BF + ratio test)
and a homography frame -> map is estimated with a robust estimator (MAGSAC++ by default).

The result for a frame is a `MapFix`: homography in ORIGINAL pixels (frame px -> map px), the frame
centre, its footprint (4 corners), heading and scale on the map.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .config import resolve_device
from .eval import localization_metrics
from .features import build_extractor
from .features.base import Features
from .geometry import apply, estimate_transform
from .io import FrameSource, imread, imwrite, load_gt, rescale
from .preprocess import Preprocessor, antimoire, drop_masked, screen_mask
from .schema import CORNER_COLS, write_trajectory
from .filtering import postprocess

log = logging.getLogger(__name__)


@dataclass
class MapFix:
    ok: bool
    H: np.ndarray | None = None  # 3x3, original frame px -> original map px
    center: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    corners: np.ndarray = field(default_factory=lambda: np.full((4, 2), np.nan))
    heading_deg: float = np.nan
    scale: float = np.nan  # map px per frame px
    n_matches: int = 0
    n_inliers: int = 0
    reproj_err: float = np.nan  # mean inlier error, map px
    reason: str = ""
    timings: dict = field(default_factory=dict)
    source: str = "global"  # global | init | roi | roi_retry | reinit | tiles | lost


def describe_homography(H: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Centre, corners, heading (0 = map up, clockwise) and scale of a frame placed by H."""
    corners = apply(H, [[0, 0], [w, 0], [w, h], [0, h]])
    center = apply(H, [[w / 2, h / 2]])[0]
    up = apply(H, [[w / 2, 0]])[0] - center  # direction of the frame's top edge
    heading = float(np.degrees(np.arctan2(up[0], -up[1])) % 360)
    # mean scale = sqrt(footprint area / frame area); valid for perspective homographies too
    x, y = corners[:, 0], corners[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    scale = float(np.sqrt(area / (w * h)))
    return center, corners, heading, scale


class _MapDescriptorIndex:
    """Nearest-neighbour index over the map descriptors, built once (FLANN KD-tree / LSH or brute force)."""

    def __init__(self, f: Features, kind: str, ratio: float | None, checks: int = 64, trees: int = 5):
        self.ratio = ratio
        self.desc = f.desc
        if kind == "bf":
            norm = cv2.NORM_HAMMING if f.kind == "binary" else cv2.NORM_L2
            self.m = cv2.BFMatcher(norm)
            self.m.add([f.desc])
        elif kind == "flann":
            if f.kind == "binary":
                index = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            else:
                index = dict(algorithm=1, trees=trees)
            self.m = cv2.FlannBasedMatcher(index, dict(checks=checks))
            self.m.add([f.desc if f.kind == "binary" else f.desc.astype(np.float32)])
        else:
            raise ValueError(f"map matching supports 'flann' and 'bf' (learned matchers: improvement phase), got {kind!r}")
        self.m.train()

    def match(self, d: np.ndarray) -> np.ndarray:
        """(K, 2) indices (frame, map)."""
        if d is None or len(d) < 2:
            return np.zeros((0, 2), int)
        if d.dtype != np.uint8:
            d = d.astype(np.float32)
        k = 2 if self.ratio else 1
        out = []
        for pair in self.m.knnMatch(d, k=k):
            if not pair:
                continue
            if self.ratio:
                if len(pair) < 2 or pair[0].distance >= self.ratio * pair[1].distance:
                    continue
            out.append((pair[0].queryIdx, pair[0].trainIdx))
        return np.array(out, int).reshape(-1, 2)


class MapLocalizer:
    def __init__(self, map_bgr: np.ndarray, cfg: dict, device: str = "cpu"):
        self.cfg = cfg
        self.lc = cfg["localize"]
        self.pre = Preprocessor(**cfg["preprocess"])
        self.map_scale = float(cfg["map"]["scale"])
        self.map_size = (map_bgr.shape[1], map_bgr.shape[0])
        self.frame_extractor = build_extractor(cfg["features"], device)
        map_extractor = build_extractor({**cfg["features"], "max_keypoints": cfg["map"]["max_keypoints"]}, device)
        t0 = time.perf_counter()
        self.map_feats = map_extractor.extract(self.pre(rescale(map_bgr, self.map_scale)))
        self.t_map = time.perf_counter() - t0
        m = cfg["matching"]
        self.index = _MapDescriptorIndex(self.map_feats, m["type"], m.get("ratio"), **(m.get("params") or {}))
        log.info("map: %dx%d, %d keypoints (%s) in %.1fs", *self.map_size, len(self.map_feats),
                 cfg["features"]["type"], self.t_map)
        g = dict(cfg["geometry"])
        g.update(min_inliers=self.lc["min_inliers"], reproj_threshold=self.lc["reproj_threshold"],
                 min_scale=self.lc["min_scale"], max_scale=self.lc["max_scale"],
                 max_perspective=self.lc["max_perspective"], model="homography")
        self.geom = g

    def localize(self, frame_bgr: np.ndarray) -> MapFix:
        h, w = frame_bgr.shape[:2]
        fs, ms = float(self.lc["frame_scale"]), self.map_scale
        t0 = time.perf_counter()
        pp = self.cfg["preprocess"]
        small = antimoire(rescale(frame_bgr, fs), pp.get("blur_sigma", 0.0))
        f = self.frame_extractor.extract(self.pre(small))
        t1 = time.perf_counter()
        m = self.index.match(f.desc) if len(f) else np.zeros((0, 2), int)
        if pp.get("screen_mask") and len(m):
            m = m[drop_masked(f.kpts[m[:, 0]], screen_mask(small))]
        t2 = time.perf_counter()
        pts_f = f.kpts[m[:, 0]] if len(m) else np.zeros((0, 2), np.float32)
        pts_m = self.map_feats.kpts[m[:, 1]] if len(m) else np.zeros((0, 2), np.float32)
        # Estimate in ORIGINAL coordinates so that thresholds are in map px.
        est = estimate_transform(pts_m / ms, pts_f / fs, self.geom, (w, h))
        t3 = time.perf_counter()
        timings = {"extract": t1 - t0, "match": t2 - t1, "geom": t3 - t2}
        fix = MapFix(False, est.T, n_matches=len(m), n_inliers=est.n_inliers, reproj_err=est.reproj_err,
                     reason=est.reason, timings=timings)
        if est.T is None:
            return fix
        center, corners, heading, scale = describe_homography(est.T, w, h)
        inside = (0 <= center[0] <= self.map_size[0]) and (0 <= center[1] <= self.map_size[1])
        fix.center, fix.corners, fix.heading_deg, fix.scale = center, corners, heading, scale
        fix.ok = est.ok and inside
        if est.ok and not inside:
            fix.reason = "centre outside the map"
        return fix


def TrackLocalizer_scale_probe(map_bgr, cfg, frame) -> float:
    """Frame->map scale from one global SIFT fix (used by the tiles mode, which then ignores the position)."""
    import copy as _copy

    g = _copy.deepcopy(cfg)
    g["features"] = {"type": "sift", "max_keypoints": 4000, "params": {}}
    g["matching"] = {"type": "flann", "ratio": 0.75, "cross_check": False, "params": {}}
    fix = MapLocalizer(map_bgr, g, "cpu").localize(frame)
    if not fix.ok:
        raise RuntimeError("tiles mode: cannot estimate the frame scale from the first frame")
    return fix.scale


def _fix_row(k: int, name: str, orig_idx: int, fix: MapFix) -> dict:
    row = {"frame": orig_idx, "name": name, "x": fix.center[0], "y": fix.center[1],
           "heading_deg": fix.heading_deg, "scale": fix.scale, "n_matches": fix.n_matches,
           "n_inliers": fix.n_inliers, "reproj_err": fix.reproj_err,
           "status": "map" if fix.ok else "lost", "reason": fix.reason, "source": fix.source}
    row.update({c: v for c, v in zip(CORNER_COLS, fix.corners.ravel())})
    row.update({f"t_{a}": b for a, b in fix.timings.items()})
    return row


def run_localization(cfg: dict, out_dir: str | Path | None = None, progress=None) -> dict:
    """Localise every frame on the map, post-process, draw the route; returns metrics.

    `progress(event, payload)` (optional) is called with "start" (frame count, map size, GT), "frame" (the raw
    per-frame row, before post-processing) and "done" (final trajectory + metrics). It lets a UI stream the run;
    an exception raised inside it aborts the run.
    """
    from .viz import render_route

    t_start = time.perf_counter()
    device = resolve_device(cfg["device"])
    run_id = f"{cfg['name']}_{datetime.now():%Y%m%d-%H%M%S}"
    out = Path(out_dir) if out_dir else Path(cfg["output"]["dir"]) / run_id
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    d = cfg["data"]
    map_bgr = imread(cfg["map"]["path"])
    source = FrameSource(d["frames"], d["start"], d["step"], d["max_frames"], resize_long=None)
    log.info("localising %d frames from %s on %s (%s)", len(source), d["frames"], cfg["map"]["path"], device)
    mode = cfg["localize"].get("mode", "global")
    torch = sys.modules.get("torch")
    if mode == "global":
        loc = MapLocalizer(map_bgr, cfg, device)
    elif mode == "track":
        from .track import TrackLocalizer

        loc = TrackLocalizer(map_bgr, cfg, device)
        prior = cfg["tracking"].get("init_prior")
        if prior is not None:
            if prior == "gt":
                g0 = load_gt(d.get("gt"), source)
                if g0 is None or not np.all(np.isfinite(g0[0])):
                    raise ValueError("tracking.init_prior=gt needs a GT position for the first frame")
                prior = g0[0]
            loc.set_prior(prior, float(cfg["localize"]["frame_scale"]), float(cfg["tracking"].get("init_heading", 0)))
            log.info("track: start prior %s", np.round(np.asarray(prior, float), 1))
    elif mode == "tiles":
        from .track import TileLocalizer

        f0 = source.load(0)  # expected scale from one global SIFT fix (position is NOT used)
        tmp = TrackLocalizer_scale_probe(map_bgr, cfg, f0)
        loc = TileLocalizer(map_bgr, cfg, device, tmp, (f0.shape[1], f0.shape[0]))
    else:
        raise ValueError(f"unknown localize.mode {mode!r}")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gt = load_gt(d.get("gt"), source)
    if progress:
        progress("start", {"run_id": run_id, "n_frames": len(source), "names": [source.name(k) for k in range(len(source))],
                           "map_size": [map_bgr.shape[1], map_bgr.shape[0]], "mode": mode, "device": device,
                           "gt": gt.tolist() if gt is not None else None})
    rows = []
    for k in tqdm(range(len(source)), desc="localise", unit="frame"):
        fix = loc.localize(source.load(k))
        rows.append(_fix_row(k, source.name(k), source.indices[k], fix))
        if progress:
            progress("frame", rows[-1])
    raw = pd.DataFrame(rows)
    traj = postprocess(raw, cfg["smoothing"])
    write_trajectory(traj, out / "trajectory.csv")

    n = len(traj)
    metrics = {
        "name": cfg["name"], "run_id": run_id, "features": cfg["features"]["type"],
        "matcher": cfg["matching"]["type"], "estimator": cfg["geometry"]["estimator"],
        "frame_scale": cfg["localize"]["frame_scale"], "device": device, "n_frames": n,
        "n_registered": int((raw["status"] == "map").sum()),
        "n_rejected": int(traj["rejected"].sum()),
        "n_interp": int((traj["status"] == "interp").sum()),
        "n_lost": int((traj["status"] == "lost").sum()),
        "mean_inliers": float(raw.loc[raw["status"] == "map", "n_inliers"].mean()),
        "median_inliers": float(raw.loc[raw["status"] == "map", "n_inliers"].median()),
        "mean_reproj_err": float(raw.loc[raw["status"] == "map", "reproj_err"].mean()),
        "mode": mode, "map_keypoints": len(getattr(loc, "map_feats", ())), "t_map_features_s": getattr(loc, "t_map", 0.0),
        "sources": json.dumps(raw["source"].value_counts().to_dict()),
        "vram_peak_mb": (torch.cuda.max_memory_allocated() / 2**20
                         if torch is not None and torch.cuda.is_available() else 0.0),
        "t_extract_ms": 1000 * float(raw["t_extract"].mean()), "t_match_ms": 1000 * float(raw["t_match"].mean()),
        "t_geom_ms": 1000 * float(raw["t_geom"].mean()),
    }
    metrics["registered_rate"] = metrics["n_registered"] / max(n, 1)
    metrics["fps"] = n / float((raw["t_extract"] + raw["t_match"] + raw["t_geom"]).sum())

    if gt is not None:
        metrics.update(localization_metrics(traj, gt, cfg["eval"]["success_thresholds"]))

    render_route(map_bgr, traj, out / "route.png", out / "route_zoom.png", cfg["viz"],
                 title=f"{cfg['name']}: {metrics['n_registered']}/{n} frames registered", gt=gt)
    metrics["t_total_s"] = time.perf_counter() - t_start
    metrics["out_dir"] = str(out)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)
    if progress:
        progress("done", {"metrics": metrics, "trajectory": traj[["x", "y", "status", *CORNER_COLS]].to_dict("records")})
    log.info("registered %d/%d, rejected %d, interpolated %d, lost %d -> %s", metrics["n_registered"], n,
             metrics["n_rejected"], metrics["n_interp"], metrics["n_lost"], out)
    return metrics
