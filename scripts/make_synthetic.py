"""Generate a synthetic nadir flight with ground truth (for testing before the real dataset)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.synthetic import make_flight  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--frame-size", default="640x480")
    ap.add_argument("--world-size", default="4000x3000")
    ap.add_argument("--turns", type=float, default=1.15)
    ap.add_argument("--tilt-std", type=float, default=0.0, help="random perspective (~1e-5 .. 1e-4)")
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", default=None, help="real map image to fly over (e.g. data/train/global_map.png)")
    ap.add_argument("--frame-scale", type=float, default=1.0, help="map px per frame px (0.25 = 4x zoom)")
    a = ap.parse_args()
    fs = tuple(int(v) for v in a.frame_size.split("x"))
    ws = tuple(int(v) for v in a.world_size.split("x"))
    out = make_flight(a.out, a.frames, fs, ws, a.turns, a.tilt_std, a.noise, a.seed,
                      a.world, a.frame_scale)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
