import numpy as np
import pytest

from vo.eval import umeyama
from vo.geometry import apply, estimate_transform
from vo.graph import Edge, edge_residuals, optimize

CFG = {"reproj_threshold": 2.0, "min_inliers": 10, "confidence": 0.999, "max_iters": 2000}


def _similarity(theta=0.3, s=1.1, t=(25.0, -40.0)):
    c, n = np.cos(theta), np.sin(theta)
    return np.array([[s * c, -s * n, t[0]], [s * n, s * c, t[1]], [0, 0, 1]])


@pytest.mark.parametrize("model", ["similarity", "affine", "homography"])
@pytest.mark.parametrize("estimator", ["ransac", "magsac"])
def test_estimate_recovers_transform_with_outliers(model, estimator):
    rng = np.random.default_rng(0)
    T = _similarity()
    pts1 = rng.uniform(0, 600, (300, 2))
    pts0 = apply(T, pts1) + rng.normal(0, 0.3, (300, 2))
    pts0[:60] = rng.uniform(0, 600, (60, 2))  # 20 % outliers
    est = estimate_transform(pts0, pts1, {**CFG, "model": model, "estimator": estimator}, (640, 480))
    assert est.ok, est.reason
    assert est.n_inliers >= 220
    probe = np.array([[0, 0], [640, 480], [320, 240]], float)
    assert np.abs(apply(est.T, probe) - apply(T, probe)).max() < 2.0


def test_too_few_matches_is_rejected():
    est = estimate_transform(np.zeros((5, 2)), np.zeros((5, 2)), {**CFG, "model": "homography"}, (640, 480))
    assert not est.ok


def test_umeyama_exact():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 2))
    T = _similarity(0.7, 2.5, (3, 4))
    s, R, t = umeyama(src, apply(T, src))
    assert s == pytest.approx(2.5)
    assert np.allclose((s * R @ src.T).T + t, apply(T, src))


def test_optimize_reduces_loop_inconsistency():
    rng = np.random.default_rng(2)
    n = 6
    true = [np.eye(3)] + [_similarity(0.05 * k, 1.0, (100.0 * k, 10.0 * k)) for k in range(1, n)]
    edges = []
    for i, j in [(k, k + 1) for k in range(n - 1)] + [(0, n - 1)]:
        pj = rng.uniform(0, 400, (30, 2))
        pi = apply(np.linalg.inv(true[i]) @ true[j], pj)
        edges.append(Edge(i, j, np.eye(3), pi, pj, 30, "loop" if j - i > 1 else "seq"))
    noisy = [true[0]] + [G @ _similarity(0.02, 1.01, (5, -5)) for G in true[1:]]  # drifted start
    before = edge_residuals(noisy, edges)
    after = edge_residuals(optimize(noisy, edges, "similarity"), edges)
    assert after["loop"] < 0.05 * before["loop"]
    assert after["seq"] < 1e-3


def _tilted_far_from_origin():
    """Frame (640x640) -> map homography: scale 0.25, 150 deg, mild perspective, frame ~3000 px from origin."""
    a, s = np.radians(150), 0.25
    R = np.array([[s * np.cos(a), -s * np.sin(a), 0], [s * np.sin(a), s * np.cos(a), 0], [0, 0, 1]])
    P = np.array([[1, 0, 0], [0, 1, 0], [2.5e-4, 1.2e-4, 1]])  # tilt, in centred frame px
    Tc = np.array([[1, 0, -320], [0, 1, -320], [0, 0, 1]])
    Tw = np.array([[1, 0, 3050], [0, 1, 960], [0, 0, 1]])
    H = Tw @ R @ P @ Tc
    return H / H[2, 2]


def test_sanity_accepts_tilted_homography_far_from_origin():
    from vo.geometry import sanity_check

    H = _tilted_far_from_origin()
    assert np.linalg.det(H[:2, :2]) < 0  # the old affine-only test called this a reflection
    assert sanity_check(H, (640, 640), {"min_scale": 0.02, "max_scale": 5.0, "max_perspective": 0.01}) == ""


def test_sanity_rejects_real_reflection():
    from vo.geometry import sanity_check

    H = _tilted_far_from_origin() @ np.diag([-1.0, 1.0, 1.0])  # mirror the frame
    H /= H[2, 2]
    assert sanity_check(H, (640, 640), {"min_scale": 0.02, "max_scale": 5.0, "max_perspective": 0.01}) == "reflection"
