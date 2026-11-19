"""Frame-to-map localisation with ANY pair matcher (classical, learned, detector-free).

TrackLocalizer
    For every frame the pose is predicted from the previous fixes (constant velocity), the frame is
    resized to the map resolution with the predicted scale (`auto_scale`) and optionally rotated
    north-up with the predicted heading (`rotation: predicted`), then matched against a map ROI around
    the predicted position. If that fails, a larger ROI is tried, then the global SIFT search
    (`localize.MapLocalizer`) re-initialises the track. The first frame is always initialised globally.

TileLocalizer
    Search-strategy ablation: the frame is matched against every overlapping map tile; the tile with the
    most inliers wins. No prior is used, so it is slow but independent of tracking.

Both return `localize.MapFix` with the homography in ORIGINAL frame px -> map px.
"""
from __future__ import annotations

import copy
import logging
import time

import cv2
import numpy as np

from .geometry import apply, estimate_transform
from .matching import build_pair_matcher
from .preprocess import antimoire, drop_masked, screen_mask

log = logging.getLogger(__name__)


def _geom_cfg(cfg: dict) -> dict:
    lc = cfg["localize"]
    g = dict(cfg["geometry"])
    g.update(min_inliers=lc["min_inliers"], reproj_threshold=lc["reproj_threshold"], min_scale=lc["min_scale"],
             max_scale=lc["max_scale"], max_perspective=lc["max_perspective"], model="homography")
    return g


def prepare_frame(frame: np.ndarray, scale: float, heading_deg: float, pp: dict):
    """Resize + rotate (north-up) the frame. Returns (image, A 3x3: original frame px -> prepared px, mask)."""
    h, w = frame.shape[:2]
    t = np.radians(heading_deg)
    c, s = np.cos(t), np.sin(t)
    # rotation by +heading maps the frame 'up' axis onto the map direction of the frame top (see describe_homography)
    A = np.array([[scale * c, -scale * s, 0], [scale * s, scale * c, 0], [0, 0, 1.0]])
    corners = apply(A, [[0, 0], [w, 0], [w, h], [0, h]])
    lo, hi = corners.min(0), corners.max(0)
    A[:2, 2] = -lo
    size = tuple(int(v) for v in np.ceil(hi - lo))
    img = cv2.warpAffine(frame, A[:2], size, flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), A[:2], size, flags=cv2.INTER_NEAREST) > 0
    img = antimoire(img, pp.get("blur_sigma", 0.0))
    mask = ~valid  # border introduced by the rotation
    if pp.get("screen_mask"):
        mask |= screen_mask(img)
    if mask.any():  # grow a little: the edge of the black border creates spurious keypoints
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return img, A, mask


class _Base:
    def __init__(self, map_bgr: np.ndarray, cfg: dict, device: str):
        self.map = map_bgr
        self.map_size = (map_bgr.shape[1], map_bgr.shape[0])
        self.cfg = cfg
        self.tc = cfg["tracking"]
        self.geom = _geom_cfg(cfg)
        self.matcher = build_pair_matcher(cfg, device)
        self.device = device
        self._key = 0
        self._global = None
        self.t_map = 0.0
        self.map_feats = ()

    def global_localizer(self):
        """Global SIFT search (init / re-init), built lazily."""
        if self._global is None:
            from .localize import MapLocalizer

            g = copy.deepcopy(self.cfg)
            g["features"] = {"type": "sift", "max_keypoints": 4000, "params": {}}
            g["matching"] = {"type": "flann", "ratio": 0.75, "cross_check": False, "params": {}}
            g["geometry"]["estimator"] = "magsac"
            g["preprocess"] = {**g["preprocess"], "clahe": False}
            g["localize"] = {**g["localize"], "frame_scale": self.cfg["localize"].get("init_frame_scale",
                                                                                     g["localize"]["frame_scale"])}
            self._global = MapLocalizer(self.map, g, "cpu")
        return self._global

    def _next_key(self):
        self._key += 1
        return self._key

    def match_region(self, frame, x0, y0, x1, y1, scale, heading):
        """Match the prepared frame against map[y0:y1, x0:x1]. Returns (MapFix-like dict, timings)."""
        from .localize import MapFix, describe_homography

        ws = float(self.tc.get("work_scale", 1.0))
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        img, A, mask = prepare_frame(frame, scale * ws, heading, self.cfg["preprocess"])
        roi = self.map[y0:y1, x0:x1]
        if abs(ws - 1) > 1e-6:
            roi = cv2.resize(roi, None, fx=ws, fy=ws, interpolation=cv2.INTER_LINEAR if ws > 1 else cv2.INTER_AREA)
        t1 = time.perf_counter()
        corr = self.matcher.match(self._next_key(), roi, self._next_key(), img)
        t2 = time.perf_counter()
        keep = drop_masked(corr.pts1, mask)
        pts_map = corr.pts0[keep] / ws + np.array([x0, y0], np.float32)
        pts_frame = apply(np.linalg.inv(A), corr.pts1[keep]).astype(np.float32)
        est = estimate_transform(pts_map, pts_frame, self.geom, (w, h))
        t3 = time.perf_counter()
        fix = MapFix(False, est.T, n_matches=int(keep.sum()), n_inliers=est.n_inliers, reproj_err=est.reproj_err,
                     reason=est.reason, timings={"extract": t1 - t0, "match": t2 - t1, "geom": t3 - t2})
        if est.T is None:
            return fix
        center, corners, hd, sc = describe_homography(est.T, w, h)
        fix.center, fix.corners, fix.heading_deg, fix.scale = center, corners, hd, sc
        inside = (x0 <= center[0] <= x1) and (y0 <= center[1] <= y1)
        fix.ok = est.ok and inside
        if est.ok and not inside:
            fix.reason = "centre outside the search region"
        return fix


class TrackLocalizer(_Base):
    def __init__(self, map_bgr, cfg, device):
        super().__init__(map_bgr, cfg, device)
        self.history: list = []  # (index, MapFix) of good fixes
        self.k = -1
        self.sources: dict[str, int] = {}

    def _predict(self):
        (i1, f1) = self.history[-1]
        center = f1.center.copy()
        if self.tc.get("motion_model") == "constant_velocity" and len(self.history) >= 2:
            (i0, f0) = self.history[-2]
            if i1 > i0:
                center = f1.center + (f1.center - f0.center) * (self.k - i1) / (i1 - i0)
        return center, f1.scale, f1.heading_deg

    def _roi(self, center, scale, w, h, factor):
        side = max(self.tc["min_roi"], factor * scale * np.hypot(w, h))
        x0, y0 = np.floor(center - side / 2).astype(int)
        x1, y1 = np.ceil(center + side / 2).astype(int)
        W, H = self.map_size
        return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

    def _rotations(self, heading):
        mode = self.tc.get("rotation", "predicted")
        if mode == "none":
            return [0.0]
        if mode == "test":
            return [0.0, 90.0, 180.0, 270.0]
        return [heading]

    def localize(self, frame):
        self.k += 1
        h, w = frame.shape[:2]
        fix, source = None, "init"
        if self.history:
            center, scale, heading = self._predict()
            # auto_scale: frame resized to the map resolution with the predicted scale;
            # otherwise a fixed resize factor (localize.frame_scale), as in the global mode
            fscale = scale if self.tc.get("auto_scale", True) else float(self.cfg["localize"]["frame_scale"])
            for factor in (self.tc["roi_factor"], self.tc["roi_retry_factor"]):
                roi = self._roi(center, scale, w, h, factor)
                cands = [self.match_region(frame, *roi, fscale, r) for r in self._rotations(heading)]
                best = max(cands, key=lambda f: (f.ok, f.n_inliers))
                tsum = {k: sum(c.timings[k] for c in cands) for k in cands[0].timings}
                best.timings = tsum
                if best.ok:
                    fix, source = best, "roi" if factor == self.tc["roi_factor"] else "roi_retry"
                    break
            if fix is None:
                source = "reinit"
        if fix is None:
            t0 = time.perf_counter()
            fix = self.global_localizer().localize(frame)
            fix.timings = {"extract": 0.0, "match": time.perf_counter() - t0, "geom": 0.0}
        fix.source = source if fix.ok else "lost"
        self.sources[fix.source] = self.sources.get(fix.source, 0) + 1
        if fix.ok:
            self.history.append((self.k, fix))
            self.history = self.history[-3:]
        return fix


class TileLocalizer(_Base):
    """No prior: frame vs every overlapping tile (scale from `localize.frame_scale`, no derotation)."""

    def __init__(self, map_bgr, cfg, device, expected_scale: float, frame_size):
        super().__init__(map_bgr, cfg, device)
        w, h = frame_size
        side = int(self.tc["tile_factor"] * expected_scale * np.hypot(w, h))
        W, H = self.map_size
        side = min(side, W, H)
        step = max(1, side // 2)
        xs = list(range(0, max(1, W - side) + 1, step)) + [W - side]
        ys = list(range(0, max(1, H - side) + 1, step)) + [H - side]
        self.tiles = sorted({(x, y, x + side, y + side) for x in xs for y in ys})
        self.scale = expected_scale
        self.sources = {"tiles": 0}
        log.info("tiles mode: %d tiles of %d px", len(self.tiles), side)

    def localize(self, frame):
        cands = [self.match_region(frame, *t, self.scale, 0.0) for t in self.tiles]
        best = max(cands, key=lambda f: (f.ok, f.n_inliers))
        best.timings = {k: sum(c.timings[k] for c in cands) for k in cands[0].timings}
        best.source = "tiles" if best.ok else "lost"
        self.sources["tiles"] += int(best.ok)
        return best
