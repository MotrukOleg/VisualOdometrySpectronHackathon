"""Pose graph: chaining relative transforms, loop-closure candidates and global optimisation."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from .geometry import apply

log = logging.getLogger(__name__)


@dataclass
class Edge:
    i: int
    j: int
    T: np.ndarray  # maps frame j -> frame i
    pts_i: np.ndarray  # inlier correspondences (subsampled) used by the optimiser
    pts_j: np.ndarray
    n_inliers: int
    kind: str  # seq | skip | loop


def subsample(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.arange(n) if n <= k else np.sort(rng.choice(n, k, replace=False))


def frame_centers(poses: list[np.ndarray], sizes: list[tuple[int, int]]) -> np.ndarray:
    return np.array([apply(G, [[w / 2, h / 2]])[0] for G, (w, h) in zip(poses, sizes)])


def loop_candidates(poses, sizes, min_gap: int, radius: float, stride: int, max_candidates: int,
                    existing: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Pairs (i, j), j - i >= min_gap, whose estimated centres are closer than radius * diagonal."""
    c = frame_centers(poses, sizes)
    diag = np.array([np.hypot(w, h) for w, h in sizes])
    scale = np.array([np.sqrt(abs(np.linalg.det(G[:2, :2]))) for G in poses])
    cands = []
    for j in range(min_gap, len(poses), max(1, stride)):
        prev = np.arange(0, j - min_gap + 1)
        d = np.linalg.norm(c[prev] - c[j], axis=1)
        k = int(np.argmin(d))
        lim = radius * diag[j] * scale[j]
        if d[k] < lim and (prev[k], j) not in existing:
            cands.append((d[k] / lim, int(prev[k]), j))
    cands.sort()
    return [(i, j) for _, i, j in cands[:max_candidates]]


# ---------------------------------------------------------------- optimisation
_NPARAM = {"similarity": 4, "affine": 6, "homography": 8}


def _pose_to_x(G: np.ndarray, model: str) -> np.ndarray:
    G = G / G[2, 2]
    if model == "similarity":
        return np.array([(G[0, 0] + G[1, 1]) / 2, (G[1, 0] - G[0, 1]) / 2, G[0, 2], G[1, 2]])
    if model == "affine":
        return G[:2].ravel()
    return G.ravel()[:8]


def _x_to_poses(x: np.ndarray, model: str, n: int) -> np.ndarray:
    p = x.reshape(n, _NPARAM[model])
    G = np.zeros((n, 3, 3))
    G[:, 2, 2] = 1.0
    if model == "similarity":
        a, b, tx, ty = p.T
        G[:, 0, 0], G[:, 0, 1], G[:, 0, 2] = a, -b, tx
        G[:, 1, 0], G[:, 1, 1], G[:, 1, 2] = b, a, ty
    elif model == "affine":
        G[:, :2] = p.reshape(n, 2, 3)
    else:
        G.reshape(n, 9)[:, :8] = p
    return G


def _project(G: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.einsum("nij,nj->ni", G, np.hstack([pts, np.ones((len(pts), 1))]))
    return h[:, :2] / h[:, 2:3]


def _stack(edges: list[Edge]):
    ei = np.concatenate([np.full(len(e.pts_i), e.i) for e in edges])
    ej = np.concatenate([np.full(len(e.pts_j), e.j) for e in edges])
    pi = np.concatenate([e.pts_i for e in edges]).astype(np.float64)
    pj = np.concatenate([e.pts_j for e in edges]).astype(np.float64)
    return ei, ej, pi, pj


def edge_residuals(poses: list[np.ndarray], edges: list[Edge]) -> dict[str, float]:
    """Mean world-frame disagreement (px) of edge correspondences, grouped by edge kind."""
    out: dict[str, list[float]] = {}
    for e in edges:
        if len(e.pts_i) == 0:
            continue
        r = np.linalg.norm(apply(poses[e.i], e.pts_i) - apply(poses[e.j], e.pts_j), axis=1)
        out.setdefault(e.kind, []).append(float(r.mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}


def optimize(poses: list[np.ndarray], edges: list[Edge], model: str, loss: str = "huber",
             f_scale: float = 2.0, max_nfev: int = 200) -> list[np.ndarray]:
    """Minimise ||G_i(p_i) - G_j(p_j)|| over all edge correspondences; G_0 is fixed (gauge)."""
    edges = [e for e in edges if len(e.pts_i) > 0]
    n = len(poses)
    if n < 2 or not edges:
        return poses
    k = _NPARAM[model]
    G0 = poses[0]
    x0 = np.concatenate([_pose_to_x(G, model) for G in poses[1:]])
    ei, ej, pi, pj = _stack(edges)
    m = len(ei)

    def fun(x):
        G = np.concatenate([G0[None], _x_to_poses(x, model, n - 1)])
        return (_project(G[ei], pi) - _project(G[ej], pj)).ravel()

    rr, cc = [], []
    rows = np.arange(m)
    for idx in (ei, ej):
        sel = idx > 0
        r, c0 = rows[sel], (idx[sel] - 1) * k
        for d in range(2):
            for q in range(k):
                rr.append(2 * r + d)
                cc.append(c0 + q)
    rr, cc = np.concatenate(rr), np.concatenate(cc)
    sp = coo_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(2 * m, (n - 1) * k)).tocsr()
    sp.data[:] = 1  # duplicates (i == j never happens, but be safe)
    r0 = np.abs(fun(x0)).mean()
    res = least_squares(fun, x0, jac_sparsity=sp, loss=loss, f_scale=f_scale, x_scale="jac",
                        method="trf", max_nfev=max_nfev)
    log.info("pose optimisation: mean |r| %.3f -> %.3f px (%d residuals, %s)",
             r0, np.abs(res.fun).mean(), m, res.message)
    G = _x_to_poses(res.x, model, n - 1)
    return [G0] + [G[q] for q in range(n - 1)]
