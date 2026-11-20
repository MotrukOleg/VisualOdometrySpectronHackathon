"""Mosaic of the UAV frames on the satellite map, from a localisation run (bonus figure).

Every registered frame is warped onto the map with its own frame -> map homography (from the footprint corners in
trajectory.csv) and averaged where frames overlap; the result is shown over a dimmed map, cropped to the route.

    python scripts/make_map_mosaic.py --run experiments/results/final_crops --frames data/train/crops \
        --out docs/presentation/figures/6_mosaic_on_map.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vo.io import FrameSource, imread, imwrite  # noqa: E402
from vo.schema import corners_of, read_trajectory  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-width", type=int, default=2400)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = yaml.safe_load(open(run / "config.yaml", encoding="utf-8"))
    m = imread(ROOT / cfg["map"]["path"]).astype(np.float32)
    traj = read_trajectory(run / "trajectory.csv")
    src = FrameSource(ROOT / a.frames)
    names = {src.name(i): i for i in range(len(src))}
    acc = np.zeros_like(m)
    wsum = np.zeros(m.shape[:2], np.float32)
    cs = corners_of(traj)
    for k, r in traj.iterrows():
        if r["status"] != "map" or str(r["name"]) not in names:
            continue
        f = src.load(names[str(r["name"])]).astype(np.float32)
        h, w = f.shape[:2]
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), cs[k].astype(np.float32))
        # feathered weight: 1 in the centre, 0 at the frame border -> seamless blending of overlaps
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        wt = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]) / (0.5 * min(w, h))
        warp = cv2.warpPerspective(f, H, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR)
        ww = cv2.warpPerspective(np.clip(wt, 0, 1), H, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR)
        acc += warp * ww[..., None]
        wsum += ww
    covered = wsum > 1e-3
    mos = acc / np.maximum(wsum, 1e-3)[..., None]
    out = (m * 0.35).copy()  # dimmed map as context
    out[covered] = mos[covered]
    ys, xs = np.nonzero(covered)
    pad = 60
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad)
    out = np.clip(out[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    s = min(1.0, a.max_width / out.shape[1])
    imwrite(a.out, cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    print(f"{int(covered.sum())} map px covered by {int((traj['status'] == 'map').sum())} frames -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
