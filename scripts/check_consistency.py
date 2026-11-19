"""Ground-truth-free check of a localisation run: frame-to-frame vs frame-to-map consistency.

For consecutive frames i, j=i+1 the relative homography T_ij (frame j -> frame i) is estimated
independently from frame-to-frame matches. If both map registrations H_i, H_j are correct, then
H_i @ T_ij must place frame j where H_j does. The disagreement (map px) of the frame centre and corners
is reported per pair; large values flag wrong registrations.

Example:
    python scripts/check_consistency.py experiments/results/standard_baseline_sift
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.config import load_config  # noqa: E402
from vo.geometry import apply, estimate_transform  # noqa: E402
from vo.io import FrameSource  # noqa: E402
from vo.matching import build_pair_matcher  # noqa: E402
from vo.schema import corners_of, read_trajectory  # noqa: E402


def homography_from_corners(corners: np.ndarray, w: int, h: int) -> np.ndarray:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(src, corners.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--frame-scale", type=float, default=0.5, help="downscale for frame-to-frame matching")
    a = ap.parse_args()
    run = Path(a.run_dir)
    cfg = load_config(run / "config.yaml")
    cfg = load_config(None, [f"data.frames={cfg['data']['frames']}", "features.type=sift",
                             "matching.type=flann", "matching.ratio=0.75"])
    traj = read_trajectory(run / "trajectory.csv")
    src = FrameSource(cfg["data"]["frames"], scale=a.frame_scale)
    matcher = build_pair_matcher(cfg, "cpu")
    geom = {"model": "homography", "estimator": "magsac", "reproj_threshold": 3.0, "min_inliers": 15,
            "min_scale": 0.3, "max_scale": 3.0, "max_perspective": 0.01}
    cs = corners_of(traj)
    w0, h0 = src.size(0)
    s = a.frame_scale
    S = np.diag([s, s, 1.0])  # original frame px -> scaled frame px
    rows = []
    for i in range(len(src) - 1):
        j = i + 1
        if not (traj["status"].iloc[i] == "map" and traj["status"].iloc[j] == "map"):
            continue
        c = matcher.match(i, src.load(i), j, src.load(j))
        est = estimate_transform(c.pts0, c.pts1, geom, src.size(j))
        if not est.ok:
            rows.append((i, j, np.nan, np.nan, est.n_inliers))
            continue
        T = np.linalg.inv(S) @ est.T @ S  # frame j -> frame i, original px
        w, h = w0 / s, h0 / s
        Hi = homography_from_corners(cs[i], w, h)
        pred = apply(Hi @ T, [[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]])
        ref = np.vstack([cs[j], traj[["x", "y"]].iloc[j].to_numpy(float)[None]])
        d = np.linalg.norm(pred - ref, axis=1)
        rows.append((i, j, float(d[4]), float(d[:4].mean()), est.n_inliers))
    arr = np.array([r[2] for r in rows], float)
    corn = np.array([r[3] for r in rows], float)
    out = {
        "pairs": len(rows),
        "pairs_failed_f2f": int(np.isnan(arr).sum()),
        "center_disagreement_median_px": float(np.nanmedian(arr)),
        "center_disagreement_mean_px": float(np.nanmean(arr)),
        "center_disagreement_max_px": float(np.nanmax(arr)),
        "corner_disagreement_mean_px": float(np.nanmean(corn)),
        "worst_pairs": [(r[0], r[1], round(r[2], 2)) for r in sorted(rows, key=lambda r: -np.nan_to_num(r[2]))[:5]],
    }
    with open(run / "consistency.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
