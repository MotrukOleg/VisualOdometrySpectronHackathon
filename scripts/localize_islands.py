"""Three-stage localisation for flights where frame-to-map matching fails on most frames (forest, other sensor):

  1. islands   - consecutive UAV frames chained by SIFT + RANSAC (vo.islands); a texture-less / non-overlapping
                 frame starts a new island.
  2. retrieval - the map is cut into frame-sized tiles; every frame is embedded (DINOv2, optionally fine-tuned, or a
                 classical thumbnail) and its top-K most similar tiles are the candidate positions (vo.retrieval).
  3. fit       - per island, the similarity island -> map that puts most frame centres onto one of their candidates
                 (RANSAC, vo.island_fit). Optional 3b (--verify): a precise matcher (any track_* config) registers
                 each frame in a ROI around its predicted position; the island is refit on those precise fixes.

Outputs (--out): positions.csv (name, island, x, y, status, err_m), metrics.json, islands.png.

    python scripts/localize_islands.py --map data/real/visloc_24/map.jpg --frames data/real/visloc_24/flight/frames \
        --gt data/real/visloc_24/flight/gt.csv --geo data/real/visloc_24/geo.json --scale 0.23 --heading 0 \
        --embedder thumb --verify configs/map/methods/track_xfeat.yaml --out experiments/islands/visloc_24_thumb
"""
import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.config import load_config, resolve_device  # noqa: E402
from vo.io import FrameSource, imread, load_gt  # noqa: E402
from vo.island_fit import fit_island  # noqa: E402
from vo.islands import build_islands  # noqa: E402
from vo.retrieval import TileIndex, build_embedder, query_images  # noqa: E402

log = logging.getLogger("islands")


def parse_spec(spec: str) -> dict:
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
    ap.add_argument("--gt")
    ap.add_argument("--geo")
    ap.add_argument("--scale", type=float, required=True, help="expected frame -> map scale (map px per frame px)")
    ap.add_argument("--heading", type=float, default=None, help="known heading of the frames (0 = north-up)")
    ap.add_argument("--embedder", default="thumb", help="thumb | dinov2[:pooling=cls,...] | dinov2:weights=<ckpt>")
    ap.add_argument("--k", type=int, default=10, help="candidates per frame")
    ap.add_argument("--stride-frac", type=float, default=0.33)
    ap.add_argument("--r-inlier-frac", type=float, default=0.5, help="inlier radius as a fraction of the tile side")
    ap.add_argument("--verify", help="track_* config of the precise matcher for stage 3b")
    ap.add_argument("--n-alt", type=int, default=10, help="stage 3b: next-best island placements to verify when the "
                    "best one gets no precise fix")
    ap.add_argument("--min-island-inliers", type=int, default=2, help="islands with fewer retrieval inliers are "
                    "'unplaced' (unless a precise fix verifies them)")
    ap.add_argument("--islands-cache", help="pickle with stage-1 islands (written if missing)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.verify, [f"data.frames={a.frames}", *a.set])
    device = resolve_device(cfg["device"])
    src = FrameSource(a.frames, 0, 1, None, resize_long=None)
    sizes = {}

    def size_of(k):
        if k not in sizes:
            f = src.load(k)
            sizes[k] = (f.shape[1], f.shape[0])
        return sizes[k]

    t = {}
    # ---- stage 1: islands
    t0 = time.perf_counter()
    cache = Path(a.islands_cache) if a.islands_cache else out / "islands.pkl"
    if cache.exists():
        islands, links = pickle.loads(cache.read_bytes())
    else:
        islands, links = build_islands(src, cfg, device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps((islands, links)))
    t["islands_s"] = time.perf_counter() - t0

    # ---- stage 2: retrieval candidates
    t0 = time.perf_counter()
    map_bgr = imread(a.map)
    side = int(round(min(size_of(0)) * a.scale))
    emb = build_embedder(parse_spec(a.embedder), device)
    index = TileIndex(map_bgr, side, max(1, int(a.stride_frac * side)), emb, cache_dir="data/cache/retrieval",
                      map_id=str(Path(a.map).resolve()))
    rots = [0] if a.heading is not None else [0, 90, 180, 270]
    cands, sims = {}, {}
    for k in range(len(src)):
        q = emb.embed(query_images(src.load(k), a.scale, rots))
        top, s = index.query(q, a.k)
        cands[k], sims[k] = index.centers[top].astype(float), s[top]
    t["retrieval_s"] = time.perf_counter() - t0

    # ---- stage 3: fit islands
    t0 = time.perf_counter()
    r_in = a.r_inlier_frac * side
    fits = []
    for ii, isl in enumerate(islands):
        q = isl.centers(size_of)
        f = fit_island(q, [cands[k] for k in isl.frames], [sims[k] for k in isl.frames], a.scale, r_in,
                       a.heading, n_alt=a.n_alt if a.verify else 0)
        f["q"] = q
        fits.append(f)
    t["fit_s"] = time.perf_counter() - t0

    # ---- stage 3b: precise verification around the predicted positions
    fixes = {}
    if a.verify:
        from vo.track import _Base

        t0 = time.perf_counter()
        base = _Base(map_bgr, cfg, device)
        H, W = map_bgr.shape[:2]
        def verify(isl, S):
            """Precise fixes of the island frames around their positions predicted by S."""
            out = {}
            for k in isl.frames:
                M = S @ isl.G[k]
                w, h = size_of(k)
                c = (M @ [w / 2, h / 2, 1])[:2]
                sc = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
                hd = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                half = cfg["tracking"]["roi_factor"] * sc * np.hypot(w, h) / 2
                x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
                x1, y1 = int(min(W, c[0] + half)), int(min(H, c[1] + half))
                if x1 - x0 < 64 or y1 - y0 < 64:
                    continue
                fx = base.match_region(src.load(k), x0, y0, x1, y1, sc, hd)
                if fx.ok:
                    out[k] = fx.center.astype(float)
            return out

        for isl, f in zip(islands, fits):
            if f["S"] is None:
                continue
            got = verify(isl, f["S"])
            for S_alt, _ in ([] if got else f.get("alternatives", [])):  # best placement unconfirmed: try the next
                got = verify(isl, S_alt)
                # an alternative has weaker retrieval support: one precise fix may be a false match (forest ROI),
                # so it needs >= 2 fixes that agree with one placement
                g = (fit_island(f["q"][[isl.frames.index(k) for k in got]], [got[k][None] for k in got],
                                [np.ones(1) for _ in got], a.scale, r_inlier=15.0, heading_deg=a.heading)
                     if len(got) >= 2 else {"n_inliers": 0})
                if g["n_inliers"] >= 2:
                    f["S"], f["alt_used"] = S_alt, True
                    break
                got = {}
            fixes.update(got)
        # refit every island on its precise fixes (RANSAC with one candidate per frame, tight radius)
        for isl, f in zip(islands, fits):
            ks = [k for k in isl.frames if k in fixes]
            if not ks:
                continue
            idx = [isl.frames.index(k) for k in ks]
            g = fit_island(f["q"][idx], [fixes[k][None] for k in ks], [np.ones(1) for _ in ks], a.scale,
                           r_inlier=15.0, heading_deg=a.heading)
            need = 2 if f.get("alt_used") or a.heading is None else 1
            if g["S"] is not None and g["n_inliers"] >= need:
                f.update(S=g["S"], verified=g["n_inliers"])
        t["verify_s"] = time.perf_counter() - t0

    # ---- positions
    gt = load_gt(a.gt, src) if a.gt else None
    mpp = json.loads(Path(a.geo).read_text(encoding="utf-8"))["meters_per_px"] if a.geo else float("nan")
    rows = []
    for ii, (isl, f) in enumerate(zip(islands, fits)):
        placed = f["S"] is not None and (f.get("verified", 0) > 0 or f["n_inliers"] >= a.min_island_inliers
                                         or len(isl.frames) == 1 and f["n_inliers"] >= 1)
        for j, k in enumerate(isl.frames):
            x = y = np.nan
            status = "unplaced"
            if placed:
                x, y = _apply_pt(f["S"], f["q"][j])
                status = "island_verified" if f.get("verified", 0) > 0 else "island"
            if k in fixes and placed and np.hypot(fixes[k][0] - x, fixes[k][1] - y) <= 15.0:
                (x, y), status = fixes[k], "map"
            r = {"name": src.name(k), "island": ii, "island_size": len(isl.frames), "x": x, "y": y,
                 "status": status, "island_inliers": f["n_inliers"], "verified": f.get("verified", 0)}
            if gt is not None:
                r.update(gt_x=gt[k][0], gt_y=gt[k][1], err_m=float(np.hypot(x - gt[k][0], y - gt[k][1]) * mpp))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "positions.csv", index=False, float_format="%.3f")

    m = {"embedder": emb.name, "verify": Path(a.verify).stem if a.verify else None, "frames": len(df),
         "islands": len(islands), "placed_%": round(100 * (df.status != "unplaced").mean(), 1),
         "status": df.status.value_counts().to_dict(), **{k: round(v, 1) for k, v in t.items()}}
    if gt is not None:
        e = df.err_m
        for thr in (5, 15, 50):
            m[f"le_{thr}m_%"] = round(100 * (e <= thr).mean(), 1)
        m["err_m_median_placed"] = round(float(e.median()), 2) if e.notna().any() else None
        isl_ok = df.groupby("island").apply(lambda g: bool((g.err_m.median() <= 50)), include_groups=False)
        m["islands_correct"] = int(isl_ok.sum())
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, default=float))
    _plot(map_bgr, df, out / "islands.png", m)


def _apply_pt(S, q):
    return S[:2, :2] @ q + S[:2, 2]


def _plot(map_bgr, df, path, m):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    H, W = map_bgr.shape[:2]
    f = 1600 / W
    img = cv2.cvtColor(cv2.resize(map_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(16, 16 * H / W + 1), dpi=110)
    ax.imshow(img)
    if "gt_x" in df:
        ax.plot(df.gt_x * f, df.gt_y * f, "-", color="white", lw=1.5, alpha=0.8, label="GPS (GT)")
    style = {"map": ("#1baf7a", "precise fix"), "island_verified": ("#2a78d6", "island, verified"),
             "island": ("#e07a1f", "island, retrieval only")}
    for st, (col, lab) in style.items():
        d = df[df.status == st]
        if len(d):
            ax.scatter(d.x * f, d.y * f, s=22, color=col, edgecolor="white", linewidth=0.5, label=f"{lab} ({len(d)})",
                       zorder=3)
    ax.set_axis_off()
    ax.legend(loc="lower right", fontsize=13, framealpha=0.9)
    e = m["embedder"]
    e = ("DINOv2, fine-tuned" if "ft=" in e else "DINOv2" if e.startswith("dinov2") else "thumbnail")
    v = {"track_xfeat": "XFeat", "track_loftr": "LoFTR", "track_sift_lg": "SIFT+LightGlue"}.get(m.get("verify"),
                                                                                              m.get("verify"))
    title = f"Islands + {e} retrieval" + (f" + {v} check" if v else "")
    if "le_15m_%" in m:
        title += f":  {m['le_15m_%']}% frames within 15 m, {m['placed_%']}% placed"
    ax.set_title(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
