"""Stage 1 of island localisation: chain the UAV frames into "islands" with frame-to-frame SIFT + RANSAC.

Frame k is linked to k-1 (and, if that fails, to k-2 ... k-`bridge`) by keypoint matching and a robust similarity /
homography fit. Frames connected by accepted links form an island; every frame of an island gets G_k (3x3, frame px
-> island reference px = px of the island's first frame). Frames without texture (forest, water) or without overlap
break the chain and start a new island. Same sensor and time, so this works where frame-to-map matching does not.

Convention (as in geometry.py): T_ij maps pixels of frame j into frame i, G_j = G_i @ T_ij.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np

from .geometry import estimate_transform
from .matching import build_pair_matcher

log = logging.getLogger(__name__)

ISLAND_DEFAULTS = {
    "features": {"type": "sift", "max_keypoints": 4000, "params": {}},
    "matching": {"type": "flann", "ratio": 0.75, "cross_check": False, "params": {}},
    "geometry": {"model": "similarity", "estimator": "magsac", "reproj_threshold": 3.0, "min_inliers": 15,
                 "min_scale": 0.7, "max_scale": 1.4},
    "resize_long": 1024,  # frames are matched at this size (positions are returned in ORIGINAL frame px)
    "bridge": 2,  # try k-1, then k-2 ... k-bridge when the previous link fails
}


@dataclass
class Island:
    frames: list[int]  # frame indices (into the FrameSource), increasing
    G: dict[int, np.ndarray] = field(default_factory=dict)  # frame px -> island reference px

    def centers(self, size_of) -> np.ndarray:
        """Frame centres in island reference px; size_of(k) -> (w, h) of frame k (original px)."""
        out = []
        for k in self.frames:
            w, h = size_of(k)
            p = self.G[k] @ np.array([w / 2, h / 2, 1.0])
            out.append(p[:2] / p[2])
        return np.array(out)


def _merge(cfg_islands: dict | None) -> dict:
    c = copy.deepcopy(ISLAND_DEFAULTS)
    for k, v in (cfg_islands or {}).items():
        c[k] = {**c[k], **v} if isinstance(v, dict) and isinstance(c.get(k), dict) else v
    return c


def build_islands(source, base_cfg: dict, device: str = "cpu"):
    """Returns (islands, links): links = list of dicts (i, j, ok, n_inliers, reason) for every tried pair."""
    ic = _merge(base_cfg.get("islands"))
    cfg = copy.deepcopy(base_cfg)
    cfg["features"], cfg["matching"] = ic["features"], ic["matching"]
    matcher = build_pair_matcher(cfg, device)
    geom = {**cfg["geometry"], **ic["geometry"]}
    rl = ic.get("resize_long")

    def load(k):
        img = source.load(k)
        h, w = img.shape[:2]
        s = min(1.0, rl / max(h, w)) if rl else 1.0
        if s < 1.0:
            import cv2

            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        return img, s, (w, h)

    n = len(source)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cache, edges, links = {}, {}, []
    for k in range(n):
        cache[k] = load(k)
        for d in range(1, int(ic["bridge"]) + 1):
            i = k - d
            if i < 0:
                break
            (im_i, s_i, _), (im_k, s_k, size_k) = cache[i], cache[k]
            corr = matcher.match(i, im_i, k, im_k)
            est = estimate_transform(corr.pts0 / s_i, corr.pts1 / s_k, geom, size_k)
            links.append({"i": i, "j": k, "ok": bool(est.ok), "n_matches": len(corr.pts0),
                          "n_inliers": int(est.n_inliers), "reason": est.reason})
            if est.ok:
                edges[(i, k)] = est.T  # k -> i
                parent[find(k)] = find(i)
                break
        cache.pop(k - int(ic["bridge"]), None)

    comps: dict[int, list[int]] = {}
    for k in range(n):
        comps.setdefault(find(k), []).append(k)
    adj: dict[int, list] = {}
    for (i, j), T in edges.items():
        adj.setdefault(i, []).append((j, T))  # G_j = G_i @ T
        adj.setdefault(j, []).append((i, np.linalg.inv(T)))  # G_i = G_j @ inv(T)
    islands = []
    for members in sorted(comps.values(), key=lambda m: m[0]):
        isl = Island(sorted(members))
        isl.G[isl.frames[0]] = np.eye(3)
        stack = [isl.frames[0]]
        while stack:
            a = stack.pop()
            for b, T in adj.get(a, []):
                if b not in isl.G:
                    isl.G[b] = isl.G[a] @ T
                    stack.append(b)
        islands.append(isl)
    sizes = [len(i.frames) for i in islands]
    log.info("islands: %d frames -> %d islands (largest %d, singletons %d)", n, len(islands), max(sizes),
             sum(s == 1 for s in sizes))
    return islands, links
