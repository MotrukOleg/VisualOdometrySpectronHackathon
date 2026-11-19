import numpy as np
import pytest

from vo.stress import AXES, add_clouds, degraded, render_frame


@pytest.fixture
def detail():
    # smooth ramp: blue channel = x / 8, green = y / 8, so a pixel tells where it was cut from
    yy, xx = np.mgrid[0:1600, 0:1600]
    img = np.zeros((1600, 1600, 3), np.uint8)
    img[..., 0], img[..., 1] = xx // 8, yy // 8
    return img


ROW = {"x": 200.0, "y": 150.0, "yaw_deg": 30.0, "scale": 0.25}  # map px; detail has k = 4 px per map px


@pytest.mark.parametrize("altitude,tilt", [(1, (0, 0)), (2, (0, 0)), (3, (0, 0)), (1, (5e-4, -5e-4))])
def test_render_keeps_gt_centre(detail, altitude, tilt):
    fr = render_frame(detail, 4.0, ROW, (64, 64), altitude, tilt)
    cx, cy = (ROW["x"] + 0.5) * 4 - 0.5, (ROW["y"] + 0.5) * 4 - 0.5
    b, g = fr[32, 32, :2].astype(float)
    assert b == pytest.approx(cx / 8, abs=1.5) and g == pytest.approx(cy / 8, abs=1.5)


def test_altitude_widens_footprint(detail):
    lo = render_frame(detail, 4.0, ROW, (64, 64), 1)
    hi = render_frame(detail, 4.0, ROW, (64, 64), 3)
    span = lambda f: np.ptp(f[..., 0].astype(float)) + np.ptp(f[..., 1].astype(float))  # noqa: E731
    assert span(hi) == pytest.approx(3 * span(lo), rel=0.15)


@pytest.mark.parametrize("name", list(AXES))
def test_degradations_are_deterministic(detail, name):
    axis = AXES[name]
    clean = render_frame(detail, 4.0, ROW, (96, 96))
    level = axis.levels[-1]
    a = degraded(axis, level, clean, detail, 4.0, ROW, seed=7)
    b = degraded(axis, level, clean, detail, 4.0, ROW, seed=7)
    assert a.shape == clean.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert degraded(axis, axis.levels[0], clean, detail, 4.0, ROW, seed=7) is clean  # level 0 = untouched


def test_cloud_coverage():
    img = np.zeros((300, 300, 3), np.uint8)
    for cov in (0.2, 0.6):
        out = add_clouds(img, np.random.default_rng(0), cov)
        assert (out[..., 0] > 117).mean() == pytest.approx(cov, abs=0.05)
