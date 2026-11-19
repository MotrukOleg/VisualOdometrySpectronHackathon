"""Reconstruct the UAV route on the satellite map (task entry point).

Reads the region map and a series of frames from the given paths, registers every frame to the map,
and writes: trajectory.csv (positions, headings, frame footprints), route.png / route_zoom.png
(map with the route, START/FINISH, order of points, frame outlines) and metrics.json.

Examples:
    python scripts/localize.py --map data/train/global_map.png --frames data/train/crops --out results/standard
    python scripts/localize.py --config configs/map/baseline_sift.yaml --set localize.frame_scale=0.35
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.benchmark import append_summary  # noqa: E402
from vo.config import load_config  # noqa: E402
from vo.localize import run_localization  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "map" / "baseline_sift.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--map", help="path to the satellite map image (overrides map.path)")
    ap.add_argument("--frames", help="directory with UAV frames (overrides data.frames)")
    ap.add_argument("--gt", help="optional ground-truth CSV (name/frame, x, y in map pixels)")
    ap.add_argument("--out", help="output directory (default: experiments/results/<name>_<timestamp>)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--summary", default="experiments/map_summary.csv")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = list(a.set)
    if a.map:
        overrides.append(f"map.path={a.map}")
    if a.frames:
        overrides.append(f"data.frames={a.frames}")
    if a.gt:
        overrides.append(f"data.gt={a.gt}")
    cfg = load_config(a.config, overrides)
    metrics = run_localization(cfg, a.out)
    append_summary(metrics, a.summary)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
