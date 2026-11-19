"""Stress curves: success rate of frame -> map localisation vs. strength of each degradation, per method.

Uses an area built by scripts/make_area.py (map + detailed image + clean flight with GT). Every method localises
the same degraded frames (global search over the whole map, no tracking), so the curves compare the
matchers themselves. A frame counts as a success if its centre is within --threshold map px of the GT.

Writes experiments/stress/<area>/{frames.csv, summary.csv} and, for the slides,
docs/presentation/figures/4_stress_<area>.png (curves) and 4_stress_<area>_examples.jpg (what the frames look like).

Examples:
    python scripts/run_stress.py --area lviv_old_town
    python scripts/run_stress.py --area lviv_old_town --axes blur clouds --frames 10      # quick look
    python scripts/run_stress.py --area lviv_old_town --methods configs/map/methods/brisk_bf.yaml
    python scripts/run_stress.py --area lviv_old_town --plot-only                         # redraw from summary.csv
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vo.config import load_config, resolve_device  # noqa: E402
from vo.io import imread, imwrite  # noqa: E402
from vo.localize import MapLocalizer  # noqa: E402
from vo.stress import AXES, degraded  # noqa: E402
from vo.tiles import GeoRef  # noqa: E402

DEFAULT_METHODS = [f"configs/map/methods/{m}.yaml" for m in ("sift_flann", "orb_bf", "akaze_bf")]

# Reference palette (dataviz skill), categorical slots 1-3: validated for all pairs incl. colour-blind vision.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"


def method_label(cfg: dict) -> str:
    return {"sift": "SIFT", "rootsift": "RootSIFT", "orb": "ORB", "akaze": "AKAZE", "brisk": "BRISK"}.get(
        cfg["features"]["type"], cfg["features"]["type"])


def level_text(axis, v) -> str:
    if axis.name == "clouds":
        return f"{100 * v:.0f}"
    if axis.name == "altitude":
        return f"x{v:g}"
    return f"{v:g}"


def plot_curves(summary: pd.DataFrame, axes: list, methods: list[str], title: str, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(axes)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, grid = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.9 * rows + 0.9), dpi=130, squeeze=False,
                             facecolor=SURFACE)
    for ax, axis in zip(grid.flat, axes):
        ax.set_facecolor(SURFACE)
        x = np.arange(len(axis.levels))
        for i, m in enumerate(methods):
            s = summary[(summary["method"] == m) & (summary["axis"] == axis.name)].sort_values("level_idx")
            dodge = 0.07 * (i - (len(methods) - 1) / 2)  # equal values would hide one line under another
            ax.plot(s["level_idx"] + dodge, 100 * s["success"], color=COLOURS[i % 3], marker=MARKERS[i % 3], lw=2,
                    ms=8, markeredgecolor=SURFACE, markeredgewidth=2, label=m, zorder=3)
        ax.set_title(axis.name, color=INK, fontsize=15, loc="left", fontweight="bold")
        ax.set_xticks(x, [level_text(axis, v) for v in axis.levels])
        ax.set_xlabel(axis.label, color=INK_2, fontsize=11)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, lw=1)
        ax.tick_params(colors=INK_2, labelsize=11, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    for ax in grid.flat[n:]:
        ax.set_visible(False)
    for ax in grid[:, 0]:
        ax.set_ylabel("frames localised, %", color=INK_2, fontsize=11)
    handles, labels = grid.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(methods), frameon=False, fontsize=13,
               labelcolor=INK, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(title, x=0.01, y=0.985, ha="left", color=INK, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def examples_strip(frames: dict, axes: list, path: Path, thumb: int = 220):
    """One row per axis, one column per level: the same frame as the methods see it."""
    ncol = max(len(a.levels) for a in axes)
    rows = []
    for axis in axes:
        cells = []
        for j in range(ncol):
            if j < len(axis.levels):
                c = cv2.resize(frames[(axis.name, j)], (thumb, thumb), interpolation=cv2.INTER_AREA)
                cv2.rectangle(c, (0, 0), (thumb, 26), (20, 20, 20), -1)
                cv2.putText(c, f"{axis.name} {level_text(axis, axis.levels[j])}", (6, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                c = np.full((thumb, thumb, 3), 252, np.uint8)
            cells.append(cv2.copyMakeBorder(c, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(252, 252, 251)))
        rows.append(np.hstack(cells))
    imwrite(path, np.vstack(rows), [1, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", required=True, help="name of an area in data/areas/ (scripts/make_area.py)")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="map-localisation configs")
    ap.add_argument("--axes", nargs="+", default=list(AXES), choices=list(AXES))
    ap.add_argument("--frames", type=int, default=30, help="frames per level, evenly spread over the flight")
    ap.add_argument("--threshold", type=float, default=5.0, help="success radius, map px")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for every method")
    ap.add_argument("--out", help="default experiments/stress/<area>")
    ap.add_argument("--figures", default=str(ROOT / "docs" / "presentation" / "figures"))
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from an existing summary.csv")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    area = ROOT / "data" / "areas" / a.area
    out = Path(a.out or ROOT / "experiments" / "stress" / a.area)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(a.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    m_per_px = GeoRef.load(area / "geo.json").meters_per_px()
    title = f"Where localisation breaks: {a.area} (success = error < {a.threshold * m_per_px:.0f} m)"
    if a.plot_only:
        summary = pd.read_csv(out / "summary.csv")
        axes = [AXES[n] for n in dict.fromkeys(summary["axis"]) if n in a.axes]
        plot_curves(summary, axes, list(dict.fromkeys(summary["method"])), title, fig_dir / f"4_stress_{a.area}.png")
        return
    map_bgr = imread(area / "map.jpg")
    detail = imread(area / "detail.jpg")
    k = detail.shape[1] / map_bgr.shape[1]
    gt = pd.read_csv(area / "flight" / "gt.csv")
    picks = np.unique(np.linspace(0, len(gt) - 1, a.frames).round().astype(int))
    clean = {i: imread(area / "flight" / "frames" / f"{gt.name[i]}.jpg") for i in picks}
    axes = [AXES[n] for n in a.axes]
    print(f"{a.area}: {len(picks)} frames x {sum(len(x.levels) for x in axes)} levels, success = error < "
          f"{a.threshold:g} map px ({a.threshold * m_per_px:.1f} m)")

    rows, examples = [], {}
    for mpath in a.methods:
        cfg = load_config(mpath, [f"map.path={area / 'map.jpg'}", *a.set])
        name = method_label(cfg)
        t0 = time.perf_counter()
        loc = MapLocalizer(map_bgr, cfg, resolve_device(cfg["device"]))
        print(f"{name}: map features in {time.perf_counter() - t0:.1f}s")
        for ai, axis in enumerate(axes):
            for li, level in enumerate(axis.levels):
                for i in picks:
                    img = degraded(axis, level, clean[i], detail, k, gt.iloc[i], seed=hash((ai, li, int(i))) % 2**32)
                    if i == picks[len(picks) // 2]:
                        examples[(axis.name, li)] = img
                    t = time.perf_counter()
                    fix = loc.localize(img)
                    err = float(np.hypot(*(fix.center - gt.loc[i, ["x", "y"]].to_numpy(float)))) if fix.ok else np.nan
                    rows.append({"method": name, "axis": axis.name, "level_idx": li, "level": level, "frame": int(i),
                                 "ok": bool(fix.ok), "err_px": err, "n_inliers": fix.n_inliers,
                                 "t_s": time.perf_counter() - t})
                s = pd.DataFrame(rows[-len(picks):])
                print(f"  {axis.name:9s} {level_text(axis, level):>5s}: "
                      f"{100 * (s.err_px < a.threshold).mean():5.1f} % localised")

    frames = pd.DataFrame(rows)
    frames.to_csv(out / "frames.csv", index=False)
    frames["success"] = frames["err_px"] < a.threshold  # NaN (not registered) -> False
    summary = (frames.groupby(["method", "axis", "level_idx", "level"], sort=False)
               .agg(success=("success", "mean"), registered=("ok", "mean"), err_median_px=("err_px", "median"),
                    inliers_median=("n_inliers", "median"), ms_per_frame=("t_s", lambda v: 1000 * v.mean()))
               .reset_index())
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    plot_curves(summary, axes, list(dict.fromkeys(frames["method"])), title, fig_dir / f"4_stress_{a.area}.png")
    examples_strip(examples, axes, fig_dir / f"4_stress_{a.area}_examples.jpg")
    print(f"written: {out / 'summary.csv'}, {fig_dir / f'4_stress_{a.area}.png'}")


if __name__ == "__main__":
    main()
