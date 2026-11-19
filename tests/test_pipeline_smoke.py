"""End-to-end smoke test on a tiny synthetic flight with exact ground truth."""
import json

from vo.config import load_config
from vo.pipeline import run
from vo.synthetic import make_flight


def test_baseline_pipeline_on_synthetic(tmp_path):
    data = make_flight(tmp_path / "syn", n_frames=50, frame_size=(400, 300), world_size=(1600, 1200),
                       turns=1.1, seed=3)
    cfg = load_config(None, [
        f"data.frames={data / 'frames'}",
        f"data.gt={data / 'gt.csv'}",
        "data.resize_long=null",
        f"output.dir={tmp_path / 'out'}",
        "mosaic.scale=0.5",
        "graph.loop_closure=true",
        "graph.loop_min_gap=10",
        "graph.loop_min_inliers=20",
        "graph.optimize=true",
        "geometry.model=similarity",
    ])
    m = run(cfg)
    assert m["n_lost"] == 0
    assert m["seq_fail_rate"] == 0.0
    assert m["ate_pct_of_path"] < 2.0
    out = tmp_path / "out" / m["run_id"]
    for f in ("pairs.csv", "trajectory.csv", "metrics.json", "mosaic_trajectory.jpg", "trajectory.png"):
        assert (out / f).exists(), f
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["n_frames"] == 50
