"""Tile retrieval quality: is the right map tile among the top-k for every frame?

A frame is resized to the map resolution with `--scale`, centre-cropped to a square, rotated by each of
`--rotations` and embedded; the map tiles (side = that square, stride = `--stride-frac` x side) are ranked by
cosine similarity. A hit at k = one of the top-k tile centres is within `--hit-frac` x side of the GT position
(the precise matcher searches a ROI ~2 x the tile around a candidate, so 0.5 leaves the frame fully inside it).

Frames can be split by the oracle-prior diagnostic (scripts/oracle_roi.py): "matchable" = registered by at least one
method when told the GT position, "unmatchable" = the rest (forest etc.).

    python scripts/eval_retrieval.py --map data/real/visloc_24/map.jpg --frames data/real/visloc_24/flight/frames \
        --gt data/real/visloc_24/flight/gt.csv --geo data/real/visloc_24/geo.json --scale 0.23 \
        --oracle-dir experiments/oracle_roi/visloc_24 --embedders thumb dinov2:pooling=cls dinov2:pooling=cls_mean
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.config import resolve_device  # noqa: E402
from vo.io import FrameSource, imread, load_gt  # noqa: E402
from vo.retrieval import TileIndex, build_embedder, query_images  # noqa: E402


def parse_spec(spec: str) -> dict:
    """'dinov2:pooling=cls,gray=1,model=facebook/dinov2-base' -> retrieval config dict."""
    kind, _, rest = spec.partition(":")
    rc = {"embedder": kind}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=", 1)
        rc[k] = int(v) if v.lstrip("-").isdigit() else (v in ("1", "true", "True") if k == "gray" else v)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--geo")
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--embedders", nargs="+", default=["thumb", "dinov2"])
    ap.add_argument("--rotations", nargs="+", type=int, default=[0])
    ap.add_argument("--stride-frac", type=float, default=0.33)
    ap.add_argument("--hit-frac", type=float, default=0.5)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--oracle-dir", help="per-method CSVs of scripts/oracle_roi.py -> matchable / unmatchable split")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--tag", default="", help="dataset label in the output")
    ap.add_argument("--out", default="experiments/retrieval")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    map_bgr = imread(a.map)
    src = FrameSource(a.frames, 0, a.step, None, resize_long=None)
    gt = load_gt(a.gt, src)
    mpp = json.loads(Path(a.geo).read_text(encoding="utf-8"))["meters_per_px"] if a.geo else float("nan")
    f0 = src.load(0)
    side = int(round(min(f0.shape[:2]) * a.scale))
    matchable = None
    if a.oracle_dir:
        oks = [pd.read_csv(p).set_index("name")["ok"].astype(bool) for p in glob.glob(f"{a.oracle_dir}/track_*.csv")]
        u = pd.concat(oks, axis=1).any(axis=1)
        matchable = np.array([bool(u.get(src.name(k), False)) for k in range(len(src))])
    device = resolve_device(a.device)
    tag = a.tag or Path(a.frames).parent.parent.name
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = [src.load(k) for k in range(len(src))]
    rows = []
    for spec in a.embedders:
        rc = parse_spec(spec)
        emb = build_embedder(rc, device)
        t0 = time.perf_counter()
        idx = TileIndex(map_bgr, side, max(1, int(a.stride_frac * side)), emb, cache_dir="data/cache/retrieval",
                        map_id=str(Path(a.map).resolve()))
        t_index = time.perf_counter() - t0
        t0 = time.perf_counter()
        per = []
        for k, f in enumerate(frames):
            q = emb.embed(query_images(f, a.scale, a.rotations))
            top, _ = idx.query(q, max(a.ks))
            d = np.hypot(*(idx.centers[top] - gt[k]).T)
            per.append({"name": src.name(k), **{f"hit@{kk}": bool((d[:kk] <= a.hit_frac * side).any()) for kk in a.ks},
                        "top1_err_m": float(d[0] * mpp), "top1_err_px": float(d[0])})
        t_q = (time.perf_counter() - t0) / len(frames)
        df = pd.DataFrame(per)
        df.to_csv(out / f"{tag}_{emb.name.replace('/', '_')}.csv", index=False, float_format="%.2f")
        groups = {"all": np.ones(len(df), bool)}
        if matchable is not None:
            groups.update(matchable=matchable, unmatchable=~matchable)
        for g, m in groups.items():
            r = {"dataset": tag, "embedder": emb.name, "rotations": "/".join(map(str, a.rotations)), "frames": g,
                 "n": int(m.sum()), "tiles": len(idx.boxes), "tile_px": side}
            for kk in a.ks:
                r[f"recall@{kk}_%"] = round(100 * df.loc[m, f"hit@{kk}"].mean(), 1) if m.any() else np.nan
            r["top1_err_m_median"] = round(float(df.loc[m, "top1_err_m"].median()), 1) if m.any() else np.nan
            r["ms_query"] = round(1000 * t_q, 1)
            r["s_index"] = round(t_index, 1)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
        del emb
    sp = out / "summary.csv"
    new = pd.DataFrame(rows)
    old = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    if len(old):
        keep = ~old.set_index(["dataset", "embedder", "rotations", "frames"]).index.isin(
            new.set_index(["dataset", "embedder", "rotations", "frames"]).index)
        old = old[keep]
    pd.concat([old, new]).to_csv(sp, index=False)
    print(f"-> {sp}")


if __name__ == "__main__":
    main()
