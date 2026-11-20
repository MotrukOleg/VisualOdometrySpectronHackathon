"""Run matchers on the exact-GT pair benchmark (scripts/make_pair_benchmark.py) and score them.

For every valid pair (img0, img1, shift dx, dy) and every method config, the method's pair matcher finds
correspondences and one fixed robust estimator (homography + MAGSAC++, the same for all methods) fits T(img1 -> img0).
Error = mean distance of the 4 corners of img1 mapped by the estimate vs by the ground truth (px).

Variants:
    plain  the pair as is (orthophoto vs orthophoto, pure shift)
    geo    img1 additionally rotated (+-15 deg) and scaled (0.75-1.3); seeded per pair, ground truth composed exactly
    hard   geo + degraded like a phone filming a screen (moire, blur, glare, colour, JPEG, bezel; src/vo/degrade.py)

Output: one row per (method, variant, pair) in --out, plus a summary table on stdout.

    python scripts/run_pair_benchmark.py --configs "configs/map/methods/track_*.yaml" --variants plain hard
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vo.config import load_config  # noqa: E402
from vo.degrade import degrade  # noqa: E402
from vo.geometry import apply, estimate_transform  # noqa: E402
from vo.io import imread  # noqa: E402
from vo.matching import build_pair_matcher  # noqa: E402

PAIR_GEOM = {"model": "homography", "estimator": "magsac", "reproj_threshold": 3.0, "min_inliers": 12,
             "min_scale": 0.3, "max_scale": 3.0, "max_perspective": 0.01}


def hard_view(img, seed, degraded=True):
    """Rotated / scaled (/ degraded) copy of img and the 3x3 map img -> copy."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.75, 1.3))
    M3 = np.vstack([M, [0, 0, 1]])
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (degrade(out, rng, "screen") if degraded else out), M3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(ROOT / "experiments/pairs/finetune_pos_crops/pairs.csv"))
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["plain", "geo", "hard"])
    ap.add_argument("--max-pairs", type=int, default=0, help="limit valid pairs (0 = all)")
    ap.add_argument("--stride", type=int, default=1, help="then keep every k-th pair (a nested subset of --max-pairs)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pairs = pd.read_csv(a.pairs)
    pairs = pairs[pairs["valid"]].reset_index(drop=True)
    if a.max_pairs:
        per = a.max_pairs // pairs["seq"].nunique()
        pairs = pd.concat([g.iloc[np.linspace(0, len(g) - 1, min(len(g), per)).astype(int)]
                           for _, g in pairs.groupby("seq")]).reset_index(drop=True)
    pairs["k"] = np.arange(len(pairs))  # index in the --max-pairs selection: seeds the geo/hard views (nested subsets)
    pairs = pairs.iloc[:: a.stride].reset_index(drop=True)
    out = Path(a.out or Path(a.pairs).with_name("results.csv"))
    device = a.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = sorted({p for pat in a.configs for p in (glob.glob(pat) or [pat])})
    imgs = {}

    def load(p):
        if p not in imgs:
            imgs[p] = imread(ROOT / p)
        return imgs[p]

    for cpath in configs:
        cfg = load_config(cpath, [])
        name = cfg["name"]
        try:
            matcher = build_pair_matcher(cfg, device)
        except Exception as e:  # missing weights / package: record and go on
            print(f"{name}: cannot build ({type(e).__name__}: {e})", flush=True)
            continue
        for vi, var in enumerate(a.variants):
            rows = []
            for k, r in pairs.iterrows():
                A, B = load(r["img0"]), load(r["img1"])
                S = np.array([[1, 0, r["dx"]], [0, 1, r["dy"]], [0, 0, 1]], float)
                if var in ("geo", "hard"):
                    B, M3 = hard_view(B, 1000 + int(r["k"]), degraded=var == "hard")
                    T_true = S @ np.linalg.inv(M3)
                else:
                    T_true = S
                h, w = B.shape[:2]
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                t0 = time.perf_counter()
                try:
                    c = matcher.match(10 * (2 * k) + vi, A, 10 * (2 * k + 1) + vi, B)
                    est = estimate_transform(c.pts0, c.pts1, PAIR_GEOM, (w, h))
                    err = float(np.linalg.norm(apply(est.T, corners) - apply(T_true, corners), axis=1).mean()) \
                        if est.ok else np.inf
                    n_in, n_m, reason = est.n_inliers, len(c.pts0), est.reason
                except Exception as e:
                    err, n_in, n_m, reason = np.inf, 0, 0, f"{type(e).__name__}: {e}"[:120]
                rows.append({"method": name, "config": Path(cpath).stem, "variant": var, "seq": r["seq"],
                             "img0": r["img0"], "err_px": err, "inliers": n_in, "matches": n_m,
                             "ms": 1000 * (time.perf_counter() - t0), "reason": reason})
            df = pd.DataFrame(rows)
            e = df["err_px"]
            print(f"{name:40s} {var:5s} n={len(df)} ok={np.isfinite(e).mean():.3f} @1px={(e <= 1).mean():.3f} "
                  f"@3px={(e <= 3).mean():.3f} @10px={(e <= 10).mean():.3f} med={np.median(e):.2f} "
                  f"inl={df['inliers'].mean():.0f} ms={df['ms'].median():.0f}", flush=True)
            df.to_csv(out, mode="a", header=not out.exists(), index=False)
        imgs.clear() if len(imgs) > 4000 else None
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
