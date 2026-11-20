import numpy as np

from vo.spoofing import detect, spoof, to_enu


def _flight(n=100, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.stack([np.arange(n) * 50.0, np.zeros(n)], 1)
    visual = truth + rng.normal(0, 3, truth.shape)
    visual[rng.random(n) < 0.6] = np.nan  # most frames have no visual fix
    return truth, visual


def test_to_enu_scale():
    e = to_enu([30.001], [50.0], 30.0, 50.0)[0]
    assert abs(e[0] - 71.5) < 0.5 and abs(e[1]) < 1e-9  # 0.001 deg of longitude at 50 N ~ 71.5 m


def test_spoof_ramps_and_saturates():
    s = spoof(np.zeros((10, 2)), start=3, rate_m=10, max_m=25, bearing_deg=90)
    assert np.allclose(s[:3], 0)
    assert np.allclose(s[3:, 0], [10, 20, 25, 25, 25, 25, 25]) and np.allclose(s[:, 1], 0, atol=1e-9)


def test_no_false_alarm_on_honest_gps():
    truth, visual = _flight()
    assert detect(truth, visual).alarm is None


def test_detects_spoofing_and_navigates_by_vision():
    truth, visual = _flight()
    gps = spoof(truth, start=40, rate_m=3, max_m=300)
    det = detect(gps, visual, threshold_m=15, confirm=3)
    assert det.alarm is not None and 40 < det.alarm < 60
    after = slice(det.alarm, None)
    err_vis = np.linalg.norm(det.position[after] - truth[after], axis=1)
    err_gps = np.linalg.norm(gps[after] - truth[after], axis=1)
    assert np.median(err_vis) < 20 < np.median(err_gps)


def test_single_outlier_does_not_alarm():
    truth, visual = _flight()
    k = np.flatnonzero(np.isfinite(visual[:, 0]))[5]
    visual[k] += 500
    assert detect(truth, visual).alarm is None
