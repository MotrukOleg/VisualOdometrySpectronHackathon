"""Frame-to-map localisation on a synthetic flight with exact ground truth."""
import numpy as np
import pandas as pd

from vo.config import load_config
from vo.filtering import postprocess
from vo.io import imwrite
from vo.localize import run_localization
from vo.schema import CORNER_COLS, read_trajectory
from vo.synthetic import make_flight, make_world


def test_localize_synthetic_flight(tmp_path):
    world = make_world(1600, 1200, seed=5)
    imwrite(tmp_path / "map.png", world)
    # frames are 2x zoomed-in w.r.t. the map (0.5 map px per frame px), with small perspective;
    # the procedural map is weakly textured (~4k SIFT points), hence the lower min_inliers
    data = make_flight(tmp_path / "syn", n_frames=25, frame_size=(400, 300), turns=0.8, tilt_std=2e-5, seed=5,
                       world_path=tmp_path / "map.png", frame_scale=0.5)
    cfg = load_config(None, [
        f"map.path={tmp_path / 'map.png'}", f"data.frames={data / 'frames'}", f"data.gt={data / 'gt.csv'}",
        "localize.frame_scale=0.5", "localize.min_inliers=10", "geometry.estimator=magsac", f"output.dir={tmp_path / 'out'}",
    ])
    m = run_localization(cfg)
    assert m["n_registered"] == 25
    assert m["center_err_median"] < 2.0
    assert m["success@10px"] == 1.0
    traj = read_trajectory(tmp_path / "out" / m["run_id"] / "trajectory.csv")
    gt = pd.read_csv(data / "gt.csv")
    # heading follows the synthetic yaw (frame 'up' rotated by yaw clockwise in image coords)
    dh = (traj["heading_deg"].to_numpy() - gt["yaw_deg"].to_numpy() + 180) % 360 - 180
    assert np.abs(dh).max() < 2.0
    # footprint scale ~ frame_scale
    assert np.allclose(traj["scale"], gt["scale"], rtol=0.05)


def _traj(xy, status):
    n = len(xy)
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "heading_deg": np.zeros(n), "scale": np.ones(n),
                       "status": status})
    for c in CORNER_COLS:
        df[c] = df["x"] if c.endswith("x") else df["y"]
    return df


def test_outlier_rejection_and_interpolation():
    xy = np.stack([np.arange(10) * 100.0, np.zeros(10)], 1)
    xy[4] = [400.0, 900.0]  # wrong registration far from the route
    status = ["map"] * 10
    status[7] = "lost"
    cfg = {"outlier_factor": 4.0, "outlier_min_px": 50.0, "interpolate": True, "savgol_window": 0,
           "savgol_order": 2}
    out = postprocess(_traj(xy, status), cfg)
    assert bool(out["rejected"].iloc[4])
    assert list(out["status"].iloc[[4, 7]]) == ["interp", "interp"]
    assert np.allclose(out[["x", "y"]].to_numpy(), np.stack([np.arange(10) * 100.0, np.zeros(10)], 1))


def test_track_with_start_prior_needs_no_global_search(tmp_path, monkeypatch):
    """tracking.init_prior=gt: the first frame is found in a ROI around the known start, never globally."""
    from vo.track import TrackLocalizer

    world = make_world(1600, 1200, seed=7)
    imwrite(tmp_path / "map.png", world)
    data = make_flight(tmp_path / "syn", n_frames=12, frame_size=(400, 300), turns=0.0, tilt_std=2e-5, seed=7,
                       world_path=tmp_path / "map.png", frame_scale=0.5)

    class NoGlobal:  # the global SIFT search always fails (as on a real cross-sensor flight)
        def localize(self, frame):
            from vo.localize import MapFix
            return MapFix(False, None, reason="global search disabled")

    calls = []
    monkeypatch.setattr(TrackLocalizer, "global_localizer", lambda self: calls.append(1) or NoGlobal())
    cfg = load_config(None, [
        f"map.path={tmp_path / 'map.png'}", f"data.frames={data / 'frames'}", f"data.gt={data / 'gt.csv'}",
        "localize.mode=track", "localize.frame_scale=0.5", "localize.min_inliers=10", "tracking.init_prior=gt",
        "tracking.rotation=none", f"output.dir={tmp_path / 'out'}",
    ])
    m = run_localization(cfg)
    traj = read_trajectory(tmp_path / "out" / m["run_id"] / "trajectory.csv")
    assert traj["source"].iloc[0] == "roi"  # first frame found around the prior, not by a global search
    assert m["n_registered"] >= 10 and len(calls) <= 2  # the weak procedural texture may drop a frame
    assert m["center_err_median"] < 2.0
