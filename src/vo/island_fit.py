"""Stage 3 of island localisation: place a whole island on the map from noisy per-frame retrieval candidates.

Every frame k of an island has a centre q_k in island reference px (stage 1) and top-K candidate map positions
p_k,m with similarities (stage 2). We look for the similarity S (island px -> map px) that sends as many frame
centres as possible onto one of their candidates (RANSAC over the candidate pairs):

    * heading known (e.g. north-up frames): one (frame, candidate) pair fixes the translation -> all are tried;
    * heading unknown: two pairs from different frames give scale + rotation + translation (random sampling).

The scale must stay within `scale_tol` of the expected frame -> map scale. Inliers: frames whose S(q_k) lies within
`r_inlier` of a candidate. The winner is refined by a least-squares similarity (Umeyama) on its inliers. A single
wrong candidate per frame is random over the map, so a consistent placement of several frames is strong evidence
even when the per-frame top-1 is mostly wrong.
"""
from __future__ import annotations

import numpy as np


def umeyama2d(src: np.ndarray, dst: np.ndarray, scale: float | None = None, rot: float | None = None):
    """Least-squares similarity dst ~ s R src + t; optionally with fixed scale and/or rotation (rad). Returns 3x3."""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    if rot is None:
        if len(src) == 1:
            th = 0.0
        else:
            U, _, Vt = np.linalg.svd(B.T @ A)
            D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
            Rm = U @ D @ Vt
            th = np.arctan2(Rm[1, 0], Rm[0, 0])
    else:
        th = rot
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if scale is None:
        den = (A ** 2).sum()
        s = float(np.trace(B.T @ (A @ R.T)) / den) if den > 1e-9 else 1.0
    else:
        s = scale
    t = md - s * R @ ms
    S = np.eye(3)
    S[:2, :2], S[:2, 2] = s * R, t
    return S


def _apply(S, q):
    return q @ S[:2, :2].T + S[:2, 2]


def _score(S, q, cands, sims, r):
    """Per frame: index of the nearest candidate within r (or -1); total score = #inliers + 0.01 * sum(sim)."""
    P = _apply(S, q)
    best, score = np.full(len(q), -1), 0.0
    for k in range(len(q)):
        d = np.hypot(*(cands[k] - P[k]).T)
        j = int(np.argmin(d))
        if d[j] <= r:
            best[k] = j
            score += 1.0 + 0.01 * float(sims[k][j])
    return score, best


def fit_island(q: np.ndarray, cands: list[np.ndarray], sims: list[np.ndarray], scale0: float, r_inlier: float,
               heading_deg: float | None = None, scale_tol: float = 1.3, iters: int = 3000, seed: int = 0,
               n_alt: int = 0) -> dict:
    """q: (n, 2) island-frame centres; cands[k]: (m_k, 2) map candidates of frame k, sims[k]: (m_k,) similarities.

    Returns dict(S, inliers (bool n), n_inliers, score, scale, heading_deg, alternatives): `alternatives` holds up to
    `n_alt` next-best hypotheses (S, score) whose island centre is > r_inlier away from every better one - for a
    precise matcher to verify when the best one is not confirmed."""
    n = len(q)
    rng = np.random.default_rng(seed)
    hyps = []
    if heading_deg is not None:
        th = np.radians(heading_deg)
        R = scale0 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for k in range(n):
            for p in cands[k]:
                S = np.eye(3)
                S[:2, :2], S[:2, 2] = R, p - R @ q[k]
                hyps.append(S)
    else:
        for _ in range(iters if n > 1 else 0):
            k1, k2 = rng.choice(n, 2, replace=False)
            if np.hypot(*(q[k1] - q[k2])) < 1e-6:
                continue
            p1 = cands[k1][rng.integers(len(cands[k1]))]
            p2 = cands[k2][rng.integers(len(cands[k2]))]
            S = umeyama2d(np.array([q[k1], q[k2]]), np.array([p1, p2]))
            s = np.sqrt(abs(np.linalg.det(S[:2, :2])))
            if scale0 / scale_tol <= s <= scale0 * scale_tol:
                hyps.append(S)
        if n == 1 or not hyps:  # single frame: its top candidate with the expected scale, heading 0
            for p in cands[0][:1] if n == 1 else [c[0] for c in cands]:
                S = np.eye(3)
                S[:2, :2] = scale0 * np.eye(2)
                S[:2, 2] = p - scale0 * q[0]
                hyps.append(S)
    scored = []
    for S in hyps:
        sc, b = _score(S, q, cands, sims, r_inlier)
        scored.append((sc, S, b))
    scored.sort(key=lambda t: -t[0])
    qc = q.mean(0)
    alts, seen = [], []
    for sc_, S_, _ in scored:
        c = _apply(S_, qc[None])[0]
        if all(np.hypot(*(c - o)) > r_inlier for o in seen):
            seen.append(c)
            alts.append((S_, sc_))
        if len(alts) > n_alt:
            break
    best = scored[0] if scored else (-1.0, None, None)
    sc, S, b = best
    if S is None:
        return {"S": None, "inliers": np.zeros(n, bool), "n_inliers": 0, "score": 0.0}
    inl = b >= 0
    if inl.sum() >= 2:  # refine on the inlier pairs (heading kept fixed if it was given)
        src = q[inl]
        dst = np.array([cands[k][b[k]] for k in np.flatnonzero(inl)])
        S2 = umeyama2d(src, dst, rot=None if heading_deg is None else np.radians(heading_deg))
        s2 = np.sqrt(abs(np.linalg.det(S2[:2, :2])))
        if scale0 / scale_tol <= s2 <= scale0 * scale_tol:
            sc2, b2 = _score(S2, q, cands, sims, r_inlier)
            if sc2 >= sc:
                S, sc, b, inl = S2, sc2, b2, b2 >= 0
    return {"S": S, "inliers": inl, "n_inliers": int(inl.sum()), "score": float(sc), "alternatives": alts[1:],
            "scale": float(np.sqrt(abs(np.linalg.det(S[:2, :2])))),
            "heading_deg": float(np.degrees(np.arctan2(S[1, 0], S[0, 0])))}
