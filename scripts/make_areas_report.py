"""Report of the any-area benchmark (scripts/make_area.py + scripts/run_map_benchmark.py --tag areas).

Reads the benchmark CSVs, keeps the latest run per (method, area), and writes
    experiments/areas_report.md                         tables: success@5px, registered, median error, fps
    docs/presentation/figures/4_areas_heatmap.png       method x area, success@5px
    docs/presentation/figures/4_areas_routes.png        GT route vs. key methods on the hardest areas

    python scripts/make_areas_report.py experiments/map_benchmark.csv experiments/map_benchmark_areas_*.csv
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vo.io import imread  # noqa: E402

FIG = ROOT / "docs" / "presentation" / "figures"
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
AREAS = ["area_lviv", "area_kyiv", "area_carpathians", "area_oleshky", "area_gorgany", "area_askania", "area_tylihul",
         "area_kyiv_sea", "area_poltava"]
SHORT = {"area_lviv": "Lviv\ncity", "area_kyiv": "Kyiv\ncity+river", "area_carpathians": "Carpathians\nforest+resort",
         "area_oleshky": "Oleshky\nsand dunes", "area_gorgany": "Gorgany\nforest", "area_askania": "Askania\nsteppe",
         "area_tylihul": "Tylihul\nestuary", "area_kyiv_sea": "Kyiv sea\nwater", "area_poltava": "Poltava\nfields"}
EXCLUDED = {"efficient_loftr_track"}  # transformers port, coarse-only matches (see map_report.md)
ROUTE_METHODS = [("map_baseline_sift", "SIFT+FLANN, whole map (baseline)", "#e07a1f"),
                 ("sift_lightglue_track", "SIFT+LightGlue, tracking", "#2a78d6"),
                 ("disk_lightglue_track", "DISK+LightGlue, tracking", "#1baf7a"),
                 ("loftr_track", "LoFTR, tracking", "#c4359b")]
ROUTE_AREAS = ["area_poltava", "area_kyiv_sea", "area_tylihul", "area_gorgany"]


def load(paths):
    d = pd.concat([pd.read_csv(p) for p in paths if Path(p).exists()], ignore_index=True)
    d = d[d["dataset"].astype(str).isin(AREAS) & d["tag"].isin(["areas", "reference"])]
    d["time"] = pd.to_datetime(d["time"])
    d = d.sort_values("time").drop_duplicates(["label", "dataset"], keep="last")
    return d[(d["status"] == "ok") & ~d["label"].isin(EXCLUDED)]


def find_traj(label, ds):
    tag = "reference" if label == "map_baseline_sift" else "areas"
    cands = [p for root in (ROOT / "experiments/results/bench", ROOT / "experiments/remote_snapshot/results/bench")
             for p in [root / tag / label / ds / "trajectory.csv"] if p.exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def md_table(t, fmt):
    cols = list(t.columns)
    out = "| method | " + " | ".join(SHORT[c].replace("\n", " ") for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for i, r in t.iterrows():
        out += f"| {i} | " + " | ".join("–" if pd.isna(v) else fmt(v) for v in r) + " |\n"
    return out


def heatmap(s5, path):
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(s5) + 2.6), dpi=150, facecolor=SURF)
    v = s5.to_numpy(float) * 100
    ax.imshow(np.nan_to_num(v, nan=-1), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            t = "–" if np.isnan(v[i, j]) else f"{v[i, j]:.0f}"
            ax.text(j, i, t, ha="center", va="center", fontsize=9, color="white" if v[i, j] > 60 else INK)
    ax.set_xticks(range(len(s5.columns)), [SHORT[c] for c in s5.columns], fontsize=9, color=INK)
    ax.set_yticks(range(len(s5)), [m.replace("_track", "").replace("_global", " (global)")
                                   .replace("map_baseline_sift", "SIFT baseline (global)") for m in s5.index],
                  fontsize=10, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.suptitle("Any-area benchmark: share of frames within 5 map px of the ground truth, %", x=0.02, ha="left",
                 fontsize=14, color=INK)
    ax.set_title("Esri map at zoom 17, frames from zoom 18–19 (scale 0.25); areas ordered from easy to hard; "
                 "methods sorted by mean", loc="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def routes(path):
    """Small multiples: rows = hard areas, columns = key methods. Green = within 5 px of GT, red x = wrong fix,
    gaps = frames the method did not register."""
    fig, axes = plt.subplots(len(ROUTE_AREAS), len(ROUTE_METHODS), figsize=(18, 3.4 * len(ROUTE_AREAS) + 1.2),
                             dpi=130, facecolor=SURF, squeeze=False)
    dsets = yaml.safe_load(open(ROOT / "configs/datasets.yaml", encoding="utf-8"))
    for i, ds in enumerate(ROUTE_AREAS):
        cfg = dsets[ds]
        m = imread(ROOT / cfg["map"])[..., ::-1]
        gt = pd.read_csv(ROOT / cfg["gt"])
        pad = 120
        x0, x1 = max(0, gt["x"].min() - pad), min(m.shape[1], gt["x"].max() + pad)
        y0, y1 = max(0, gt["y"].min() - pad), min(m.shape[0], gt["y"].max() + pad)
        for j, (label, name, _) in enumerate(ROUTE_METHODS):
            ax = axes[i, j]
            ax.imshow(m)
            ax.plot(gt["x"], gt["y"], color="white", lw=1.5, alpha=0.8, zorder=2)
            p = find_traj(label, ds)
            txt = "no run"
            if p is not None:
                t = pd.read_csv(p).merge(gt[["name", "x", "y"]], on="name", suffixes=("", "_gt"))
                reg = t[t["status"] == "map"]
                e = np.hypot(reg["x"] - reg["x_gt"], reg["y"] - reg["y_gt"])
                good, bad = reg[e <= 5], reg[e > 5]
                ax.scatter(good["x"], good["y"], s=14, color="#1baf7a", edgecolor="white", linewidth=0.4, zorder=3)
                ax.scatter(bad["x"].clip(x0, x1), bad["y"].clip(y0, y1), s=40, marker="x", color="#e5484d",
                           linewidth=1.6, zorder=4)
                txt = f"{100 * len(good) / len(t):.0f}% within 5 px, {len(bad)} wrong"
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(txt, fontsize=10, color=INK, loc="left")
            if i == 0:
                ax.text(0, 1.22, name, transform=ax.transAxes, fontsize=12, color=INK, fontweight="bold")
            if j == 0:
                ax.set_ylabel(SHORT[ds].replace("\n", " — "), fontsize=12, color=INK)
    fig.suptitle("Hard areas: where each method keeps the drone on the map. White = true route, green = fix within "
                 "5 px, red × = wrong fix, gap = frame lost", x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    a = ap.parse_args()
    files = [p for pat in a.csv for p in (glob.glob(pat) or [pat])]
    d = load(files)
    piv = lambda col: d.pivot(index="label", columns="dataset", values=col).reindex(columns=AREAS)  # noqa: E731
    s5, reg, err, fps = piv("success@5px"), piv("registered_rate"), piv("center_err_median"), piv("fps")
    order = s5.mean(axis=1).sort_values(ascending=False).index
    s5, reg, err, fps = s5.loc[order], reg.loc[order], err.loc[order], fps.loc[order]
    md = ["# Any-area benchmark\n",
          "Map: Esri World Imagery at zoom 17 (~0.8 m/px, 2.8×1.4 km); UAV frames rendered from zoom 19 "
          "(zoom 18 ×2 where 19 is missing: carpathians, poltava and the 5 hard areas), frame→map scale 0.25, exact "
          "GT. 80 frames (first four areas) or 50 frames (hard areas). success@5px = share of ALL frames within 5 map "
          "px (≈4 m) of the GT, unregistered frames count as failures.\n",
          "\n## success@5px\n", md_table(s5, lambda v: f"{100 * v:.0f}%"),
          "\n## registered (status map)\n", md_table(reg, lambda v: f"{100 * v:.0f}%"),
          "\n## median centre error, map px (registered + interpolated frames)\n", md_table(err, lambda v: f"{v:.2f}"),
          "\n## speed, frames per second (local GTX 1650 / shared RTX 5060 Ti — not comparable across machines)\n",
          md_table(fps, lambda v: f"{v:.1f}")]
    out = ROOT / "experiments" / "areas_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    heatmap(s5, FIG / "4_areas_heatmap.png")
    routes(FIG / "4_areas_routes.png")
    print(f"report: {out}; {len(d)} runs, {len(order)} methods")
    print((s5 * 100).round(0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
