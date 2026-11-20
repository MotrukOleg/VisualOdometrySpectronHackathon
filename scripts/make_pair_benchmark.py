"""Pair benchmark with exact ground truth from overlapping crops of one north-up orthophoto.

Consecutive crops of a sequence (e.g. data/_finetune_pos_crops/<seq>/*.jpg) are cut from the same raster, so the
true transform between them is a pure integer shift. It is found WITHOUT any feature matcher (no bias towards a
method): the mean squared pixel difference over the overlap is computed for every integer shift with FFT
correlations, and the minimum is accepted as ground truth only if it is

    * near-identical:  RMSE <= --max-rmse grey levels (JPEG noise only),
    * unique:          every shift farther than 4 px from it has RMSE >= --min-ratio x best + 2,
    * large enough:    overlap >= --min-overlap of the crop.

Pairs failing that (no overlap, flat water / sand where the shift is ambiguous) are kept in the CSV with a reason and
left out of the benchmark.

Output: experiments/pairs/<name>/pairs.csv  (seq, img0, img1, dx, dy, rmse, overlap, valid, reason)
Convention: point (x, y) of img1 lies at (x + dx, y + dy) in img0, i.e. T(img1 -> img0) = [[1,0,dx],[0,1,dy]].

    python scripts/make_pair_benchmark.py data/_finetune_pos_crops --per-seq 150
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vo.io import imread, natural_key  # noqa: E402


def _corr(a, b):
    """out[dy + hb - 1, dx + wb - 1] = sum_y,x a[y, x] * b[y - dy, x - dx]."""
    return fftconvolve(a, b[::-1, ::-1], mode="full")


def exact_shift(A, B, min_overlap=0.15, max_rmse=4.0, min_ratio=2.0):
    a = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = b.shape
    oa, ob = np.ones_like(a), np.ones_like(b)
    n = _corr(oa, ob)
    ssd = _corr(a * a, ob) + _corr(oa, b * b) - 2 * _corr(a, b)
    ok = n >= min_overlap * h * w
    mse = np.where(ok, np.maximum(ssd, 0) / np.maximum(n, 1), np.inf)
    iy, ix = np.unravel_index(np.argmin(mse), mse.shape)
    best = float(np.sqrt(mse[iy, ix]))
    dy, dx = iy - (h - 1), ix - (w - 1)
    yy, xx = np.ogrid[: mse.shape[0], : mse.shape[1]]
    far = (np.abs(yy - iy) > 4) | (np.abs(xx - ix) > 4)
    second = float(np.sqrt(np.min(np.where(far, mse, np.inf))))
    overlap = float(n[iy, ix] / (h * w))
    if not np.isfinite(best):
        return dict(dx=np.nan, dy=np.nan, rmse=np.nan, rmse2=np.nan, overlap=0.0, valid=False, reason="no overlap")
    reason = ""
    if best > max_rmse:
        reason = "no exact match"
    elif second < min_ratio * best + 2:
        reason = "ambiguous"
    return dict(dx=int(dx), dy=int(dy), rmse=best, rmse2=second, overlap=overlap, valid=not reason, reason=reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder with one sub-folder of crops per sequence")
    ap.add_argument("--per-seq", type=int, default=150, help="consecutive pairs sampled evenly per sequence")
    ap.add_argument("--name", default=None, help="output name (default: folder name)")
    ap.add_argument("--max-rmse", type=float, default=4.0)
    ap.add_argument("--min-ratio", type=float, default=2.0)
    ap.add_argument("--min-overlap", type=float, default=0.15)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    out = ROOT / "experiments" / "pairs" / (a.name or root.name.strip("_"))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted((p for p in seq.iterdir() if p.suffix.lower() in (".jpg", ".png")), key=lambda p: natural_key(p.name))
        idx = np.unique(np.linspace(0, len(files) - 2, min(a.per_seq, len(files) - 1)).round().astype(int))
        for i in idx:
            r = exact_shift(imread(files[i]), imread(files[i + 1]), a.min_overlap, a.max_rmse, a.min_ratio)
            rows.append({"seq": seq.name, "img0": str(files[i].relative_to(ROOT)).replace("\\", "/"),
                         "img1": str(files[i + 1].relative_to(ROOT)).replace("\\", "/"), **r})
        g = pd.DataFrame([r for r in rows if r["seq"] == seq.name])
        print(f"{seq.name}: {len(files)} crops, {len(g)} pairs, valid {int(g['valid'].sum())}, "
              f"rejected {g['reason'].value_counts().to_dict() if (~g['valid']).any() else {}}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pairs.csv", index=False)
    print(f"saved {out / 'pairs.csv'}: {int(df['valid'].sum())}/{len(df)} valid pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
