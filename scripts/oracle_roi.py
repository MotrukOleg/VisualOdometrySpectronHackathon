"""Oracle-prior diagnostic: can a matcher register a frame at all if it is told roughly where to look?

Every frame is matched against a map window centred on its ground-truth position (the side is
`--roi-factor` x the expected footprint diagonal), with the method's own matcher and the same geometry checks as
tracking. Tracking failures cannot cascade here, so this is an upper bound for "matcher + map" on hard data
(real flights, other sensor / year). Writes per-frame rows (trajectory-like: name, x, y, n_inliers, ok, err_px,
err_m) and one summary line per method.

    python scripts/oracle_roi.py --map data/real/visloc_24/map.jpg --frames data/real/visloc_24/flight/frames \
        --gt data/real/visloc_24/flight/gt.csv --geo data/real/visloc_24/geo.json \
        --configs configs/map/methods/track_sift_lg.yaml configs/map/methods/track_loftr.yaml --scale 0.3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.config import load_config, resolve_device  # noqa: E402
from vo.io import FrameSource, imread, load_gt  # noqa: E402
from vo.track import _Base  # noqa: E402


def _save_summary(path: Path, summary: list):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    new = pd.DataFrame(summary)
    if len(old):
        old = old[~old["method"].isin(new["method"])]
    pd.concat([old, new]).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--geo", help="geo.json with meters_per_px (errors in metres)")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--scale", type=float, required=True, help="expected frame -> map scale (map px per frame px)")
    ap.add_argument("--roi-factor", type=float, default=1.6)
    ap.add_argument("--heading", type=float, default=0.0, help="assumed heading (deg); 0 = frames are north-up")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default="experiments/oracle_roi")
    a = ap.parse_args()

    mpp = json.loads(Path(a.geo).read_text(encoding="utf-8"))["meters_per_px"] if a.geo else float("nan")
    map_bgr = imread(a.map)
    W, H = map_bgr.shape[1], map_bgr.shape[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for cpath in a.configs:
        cfg = load_config(cpath, [f"data.frames={a.frames}", f"data.step={a.step}", *a.set])
        name = Path(cpath).stem
        src = FrameSource(a.frames, 0, a.step, None, resize_long=None)
        gt = load_gt(a.gt, src)
        try:
            base = _Base(map_bgr, cfg, resolve_device(cfg["device"]))
        except Exception as e:  # missing package / weights
            print(f"{name}: cannot build matcher: {e}")
            continue
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rows, t0 = [], time.perf_counter()
        for k in range(len(src)):
            f = src.load(k)
            h, w = f.shape[:2]
            side = max(256, a.roi_factor * a.scale * np.hypot(w, h))
            cx, cy = gt[k]
            x0, y0 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
            x1, y1 = min(W, int(cx + side / 2)), min(H, int(cy + side / 2))
            try:
                fix = base.match_region(f, x0, y0, x1, y1, a.scale, a.heading)
            except Exception as e:  # OOM etc.: count as a failure, keep going
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                rows.append({"name": src.name(k), "ok": False, "reason": f"error: {type(e).__name__}"})
                continue
            err = float(np.hypot(*(fix.center - gt[k]))) if fix.ok else np.nan
            rows.append({"name": src.name(k), "gt_x": cx, "gt_y": cy, "x": fix.center[0], "y": fix.center[1],
                         "ok": bool(fix.ok), "n_matches": fix.n_matches, "n_inliers": fix.n_inliers,
                         "scale": fix.scale, "heading_deg": fix.heading_deg, "err_px": err, "err_m": err * mpp,
                         "reason": fix.reason})
        dt = time.perf_counter() - t0
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.3f")
        ok = df["ok"].astype(bool)
        e = df.loc[ok, "err_m"]
        s = {"method": name, "frames": len(df), "registered": int(ok.sum()),
             "registered_%": round(100 * ok.mean(), 1),
             "le_5m_%": round(100 * (df["err_m"] <= 5).mean(), 1),
             "le_15m_%": round(100 * (df["err_m"] <= 15).mean(), 1),
             "err_m_median": round(float(e.median()), 2) if len(e) else np.nan,
             "err_m_p90": round(float(e.quantile(0.9)), 2) if len(e) else np.nan,
             "inliers_median_all": float(df["n_inliers"].median()) if "n_inliers" in df else 0.0,
             "inliers_median_ok": float(df.loc[ok, "n_inliers"].median()) if ok.any() else np.nan,
             "scale_median_ok": round(float(df.loc[ok, "scale"].median()), 3) if ok.any() else np.nan,
             "s_per_frame": round(dt / max(len(df), 1), 3),
             "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch is not None
             and torch.cuda.is_available() else 0}
        summary.append(s)
        print(json.dumps(s, ensure_ascii=False), flush=True)
        _save_summary(out / "summary.csv", summary)  # after every method: a CUDA crash later keeps the rest
        del base
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"-> {out / 'summary.csv'}")

if __name__ == "__main__":
    main()
