"""Comparison report of the map-localisation benchmark: tables (Markdown) + figures for the slides.

Reads one or more benchmark CSVs (local + the GPU machine), keeps the latest row per (tag, label, dataset), and writes
    experiments/map_report.md
    docs/presentation/figures/3_methods_accuracy.png, 3_methods_pareto.png, 3_ablations.png

Ranking (lower is better): synthetic-hard success@5px (desc), synthetic-hard median error, real-data consistency.

Example:
    python scripts/make_map_report.py experiments/map_benchmark.csv experiments/map_benchmark_remote.csv
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "presentation" / "figures"
SURF, INK, INK2, S1, S2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#e07a1f", "#e4e3df"
REAL = ["crops", "video0061", "video0062"]
NEW_VIDEO = "video214816"
# kept in the tables for transparency, left out of the ranking and the figures (implementation problem, see note)
EXCLUDED = {"efficient_loftr_track": "transformers port: coarse-only matches (8-px grid, no sub-pixel refinement) and a "
                                     "fixed 640x480 resize of both images; see efficient_loftr_official_track"}


def _fallback_share(row) -> float:
    """Share of frames whose fix came from the global SIFT re-initialisation, not from the method itself."""
    try:
        src = json.loads(row["sources"])
    except (TypeError, ValueError):
        return np.nan
    n = sum(src.values())
    return src.get("reinit", 0) / n if n else np.nan


def load(paths):
    # machine: the local CSV (experiments/map_benchmark.csv) was produced on the dev PC, every other one on the GPU
    # server; speeds of the two are not directly comparable
    df = pd.concat([pd.read_csv(p).assign(machine="local" if Path(p).name == "map_benchmark.csv" else "gpu")
                    for p in paths if Path(p).exists()], ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates(["tag", "label", "dataset"], keep="last")
    # every 2nd keyframe ("...s2") stands in for the full video when a method was only run on the subsampled set
    df["subsampled"] = df["dataset"].str.endswith("s2")
    df["dataset"] = df["dataset"].str.replace(r"s2$", "", regex=True)
    df = df.sort_values("subsampled").drop_duplicates(["tag", "label", "dataset"], keep="first")
    df["fallback"] = df.apply(_fallback_share, axis=1)
    return df


def pivot(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (tag, label) with the key metrics of every dataset."""
    rows = []
    for (tag, label), g in df.groupby(["tag", "label"], sort=False):
        r = {"tag": tag, "label": label, "config": g["config"].iloc[0], "variant": g["variant"].fillna("").iloc[0]}
        by = g.set_index("dataset")
        for ds in by.index:
            x = by.loc[ds]
            r[f"{ds}.reg"] = x.get("registered_rate")
            r[f"{ds}.fps"] = x.get("fps")
            r[f"{ds}.sub"] = bool(x.get("subsampled", False))
            r[f"{ds}.fallback"] = x.get("fallback")
            for k in ("center_err_median", "center_err_mean", "center_err_max", "success@5px", "success@10px",
                      "cons_median_px", "cons_p95_px", "ref_dev_median_px", "ref_dev_p95_px", "ref_far_frames",
                      "mean_inliers", "vram_peak_mb"):
                if k in x and pd.notna(x[k]):
                    r[f"{ds}.{k}"] = x[k]
        real = [r.get(f"{d}.cons_median_px") for d in REAL if pd.notna(r.get(f"{d}.cons_median_px", np.nan))]
        r["real.cons_median_px"] = float(np.mean(real)) if real else np.nan
        fps = [r.get(f"{d}.fps") for d in by.index if pd.notna(r.get(f"{d}.fps", np.nan))]
        r["fps_mean"] = float(np.mean(fps)) if fps else np.nan
        regs = [r.get(f"{d}.reg") for d in by.index]
        r["reg_min"] = float(np.nanmin(regs)) if regs else np.nan
        vr = [r.get(f"{d}.vram_peak_mb", np.nan) for d in by.index]
        r["vram_max_mb"] = float(np.nanmax(vr)) if np.isfinite(vr).any() else np.nan
        fb = [r.get(f"{d}.fallback", np.nan) for d in by.index]
        r["fallback_max"] = float(np.nanmax(fb)) if np.isfinite(fb).any() else np.nan
        r["datasets"] = ",".join(sorted(by.index))
        r["machine"] = g["machine"].mode().iloc[0]
        r["videos_sub"] = "s2" if any(r.get(f"{d}.sub") for d in by.index) else ""
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(v, nd=2, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.{nd}f}"


METHOD_COLS = [
    ("syn_hard.success@5px", "syn_hard ≤5px", True, 0), ("syn_hard.center_err_median", "syn_hard err med, px", False, 2),
    ("syn_hard.reg", "syn_hard reg", True, 0), ("syn_easy.center_err_median", "syn_easy err med, px", False, 2),
    ("crops.reg", "crops reg", True, 0), ("video0061.reg", "v61 reg", True, 0), ("video0062.reg", "v62 reg", True, 0),
    ("real.cons_median_px", "real cons, px", False, 3), ("video0062.ref_dev_median_px", "v62 dev vs SIFT, px", False, 2),
    ("fallback_max", "SIFT fallback (max)", True, 0), ("fps_mean", "fps (mean)", False, 1),
    ("vram_max_mb", "VRAM, MB", False, 0),
]
NEW_COLS = [(f"{NEW_VIDEO}.reg", "reg", True, 0), (f"{NEW_VIDEO}.cons_median_px", "cons med, px", False, 3),
            (f"{NEW_VIDEO}.cons_p95_px", "cons p95, px", False, 2),
            (f"{NEW_VIDEO}.ref_dev_median_px", "dev vs SIFT med, px", False, 2),
            (f"{NEW_VIDEO}.ref_dev_p95_px", "dev p95, px", False, 2), (f"{NEW_VIDEO}.fallback", "SIFT fallback", True, 0),
            (f"{NEW_VIDEO}.mean_inliers", "inliers", False, 0), (f"{NEW_VIDEO}.fps", "fps", False, 2)]


def table(p: pd.DataFrame, cols=METHOD_COLS) -> str:
    head = "| method | " + " | ".join(c[1] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in p.iterrows():
        name = r["label"] + (" ¹" if r.get("videos_sub") else "")
        body += f"| {name} | " + " | ".join(fmt(r.get(k, np.nan), nd, pct) for k, _, pct, nd in cols) + " |\n"
    return head + body


def rank(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["_s5"] = -p.get("syn_hard.success@5px", pd.Series(np.nan, index=p.index)).fillna(-1)
    p["_err"] = p.get("syn_hard.center_err_median", pd.Series(np.nan, index=p.index)).fillna(1e9)
    p["_cons"] = p["real.cons_median_px"].fillna(1e9)
    return p.sort_values(["_s5", "_err", "_cons"]).drop(columns=["_s5", "_err", "_cons"])


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#b9b8b3")
    ax.spines["bottom"].set_color("#b9b8b3")
    ax.tick_params(colors=INK2)
    ax.grid(color=GRID, linewidth=1, zorder=0)


GREY = "#a3a29d"


def _short(label: str) -> str:
    return (label.replace("_track", "").replace("_global", " (global)").replace("superpoint", "SP")
            .replace("lightglue", "LG").replace("superglue", "SG").replace("map_baseline_sift", "SIFT baseline"))


def _full_reg(q: pd.DataFrame) -> pd.Series:
    """True where the method registered at least as many syn_hard frames as the typical method (comparable median)."""
    return q["syn_hard.reg"] >= q["syn_hard.reg"].median() - 0.01


def fig_accuracy(p: pd.DataFrame, path: Path):
    q = p.dropna(subset=["syn_hard.center_err_median"]).copy()
    q = q.sort_values("syn_hard.center_err_median", ascending=False)
    full = _full_reg(q).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(q) + 2.6), dpi=150, facecolor=SURF)
    style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(q))
    ax.barh(y, q["syn_hard.center_err_median"], color=np.where(full, S1, GREY), height=0.6, zorder=3,
            edgecolor=SURF, linewidth=2)
    for yi, f, (_, r) in zip(y, full, q.iterrows()):
        note = "" if f else "  (fewer frames registered: hardest frames left out)"
        ax.text(r["syn_hard.center_err_median"], yi, f"  {r['syn_hard.center_err_median']:.2f} px · "
                f"reg {100 * r['syn_hard.reg']:.0f}%{note}", va="center", color=INK if f else INK2, fontsize=10)
    ax.set_yticks(y, [_short(l) for l in q["label"]], color=INK, fontsize=11)
    ax.set_xlabel("median centre error on the hard synthetic flight (map px, lower is better)", color=INK2)
    ax.set_xlim(0, q["syn_hard.center_err_median"].max() * 2.1)
    fig.suptitle("Accuracy with ground truth: synthetic flight filmed 'from a screen'", x=0.02, ha="left",
                 color=INK, fontsize=15)
    ax.set_title("80 frames: moire, motion blur, glare, washed-out colours, JPEG, bezel, strong tilt, full rotation. "
                 "Grey = median over fewer frames, not comparable.", loc="left", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


KEY_METHODS = {"map_baseline_sift", "superpoint_lightglue_track", "disk_lightglue_track", "loftr_track",
               "roma_track", "xfeat_track", "efficient_loftr_official_track", "superpoint_superglue_official_track"}


def fig_pareto(p: pd.DataFrame, path: Path):
    q = p.dropna(subset=["syn_hard.center_err_median", "fps_mean"]).copy()
    q = q[_full_reg(q)]  # comparable medians only
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150, facecolor=SURF)
    style(ax)
    for mach, col, name in (("local", S2, "classical, dev PC (CPU)"), ("gpu", S1, "learned / GPU server (RTX 5060 Ti)")):
        g = q[q["machine"] == mach]
        ax.scatter(g["fps_mean"], g["syn_hard.center_err_median"], s=80, color=col, edgecolor=SURF, linewidth=2,
                   zorder=3, label=name)
    pts = q.sort_values("fps_mean", ascending=False)
    front, best = [], np.inf
    for _, r in pts.iterrows():  # Pareto front: no other method is both faster and more accurate
        if r["syn_hard.center_err_median"] < best:
            front.append(r)
            best = r["syn_hard.center_err_median"]
    if len(front) > 1:
        fr = pd.DataFrame(front)
        ax.plot(fr["fps_mean"], fr["syn_hard.center_err_median"], color=INK2, lw=1.5, ls="--", zorder=2,
                label="Pareto front")
    show = set(KEY_METHODS) | {r["label"] for r in front}
    lab = q[q["label"].isin(show)]
    # dense fast cluster: labels in a column to the right with leader lines; the rest next to their point
    right = lab[lab["fps_mean"] > 5].sort_values("syn_hard.center_err_median", ascending=False)
    ys = np.linspace(right["syn_hard.center_err_median"].max() + 0.12, right["syn_hard.center_err_median"].min()
                     - 0.08, len(right)) if len(right) else []
    x_col = q["fps_mean"].max() * 2.2
    for yl, (_, r) in zip(ys, right.iterrows()):
        ax.annotate(_short(r["label"]), (r["fps_mean"], r["syn_hard.center_err_median"]), xytext=(x_col, yl),
                    textcoords="data", fontsize=10, color=INK, va="center",
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8})
    for _, r in lab[lab["fps_mean"] <= 5].iterrows():
        ax.annotate(_short(r["label"]), (r["fps_mean"], r["syn_hard.center_err_median"]), xytext=(8, -4),
                    textcoords="offset points", fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(q["fps_mean"].min() / 1.5, x_col * 4)
    ax.set_xlabel("speed, frames per second (mean over datasets, log scale)", color=INK2)
    ax.set_ylabel("median error on the hard synthetic flight, map px", color=INK2)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.suptitle("Accuracy vs speed", x=0.02, ha="left", color=INK, fontsize=15)
    ax.set_title("Speeds come from two machines and are comparable only within one colour.", loc="left",
                 color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def _variant(label: str) -> str:
    """'name[a.b=1,c.d=2]' -> 'b=1, d=2' (short override names for the slides); no override -> 'base'."""
    if "[" not in label:
        return "base"
    parts = label.split("[", 1)[1].rstrip("]").split(",")
    return ", ".join(x.split(".")[-1] for x in parts)


def fig_ablations(groups: dict, path: Path):
    """groups: title -> DataFrame(label, value) with the base first."""
    n = len(groups)
    if not n:
        return
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.0 + sum(0.42 * len(g) + 0.6 for g in groups.values())),
                             dpi=150, facecolor=SURF, squeeze=False,
                             gridspec_kw={"height_ratios": [len(g) + 1 for g in groups.values()]})
    for ax, (title, g) in zip(axes[:, 0], groups.items()):
        style(ax)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(g))[::-1]
        cols = [S2 if i == 0 else S1 for i in range(len(g))]
        ax.barh(y, g["value"], color=cols, height=0.6, zorder=3, edgecolor=SURF, linewidth=2)
        for yi, v, t in zip(y, g["value"], g["text"]):
            ax.text(v, yi, f"  {t}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, g["label"], fontsize=10, color=INK)
        ax.set_xlim(0, max(g["value"].max(), 1e-6) * 1.6)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.suptitle(f"Ablation: {path.stem.replace('3_ablation_', '')} (orange = base configuration; median error on syn_hard)",
                 x=0.02, ha="left", color=INK, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "map_report.md"))
    a = ap.parse_args()
    df = load(a.csv)
    p = pivot(df)
    FIG.mkdir(parents=True, exist_ok=True)
    md = ["# Map-localisation benchmark\n",
          "Datasets: `crops` (50 task frames), `syn_easy` / `syn_hard` (80-frame synthetic flights over the map with "
          "exact GT; hard = screen degradations + strong tilt), `video0061` / `video0062` (own videos, every 5th frame).",
          "Metrics: reg = share of frames registered on the map; err = centre error vs GT (map px); "
          "cons = frame-to-frame vs frame-to-map disagreement (GT-free, map px); dev = distance to the SIFT whole-map "
          "reference trajectory; fps = localisation speed.\n"]
    ref = p[p["tag"] == "reference"]
    methods = rank(pd.concat([ref, p[p["tag"] == "methods"]]))
    excluded = methods[methods["label"].isin(EXCLUDED)]
    methods = methods[~methods["label"].isin(EXCLUDED)]
    md += ["## Methods\n", table(methods),
           "\n¹ videos evaluated on every 2nd keyframe (`…s2`, every 10th video frame) to fit the time budget. "
           "SIFT fallback = share of frames located by the global SIFT re-initialisation instead of the method "
           "itself; a high value means the method mostly failed and its numbers are SIFT's.\n",
           "\nExcluded from the ranking:\n" + "\n".join(f"- `{k}`: {v}" for k, v in EXCLUDED.items()) + "\n\n"
           + table(excluded)]
    cross = ROOT / "experiments" / "cross_consistency.csv"
    if cross.exists():  # written by scripts/cross_consistency.py
        cx = pd.read_csv(cross)
        cx = cx[~cx["label"].isin(EXCLUDED)]
        piv = cx.pivot_table(index="label", columns="set", values="cons_other")
        piv = piv.loc[piv.mean(axis=1).sort_values().index]
        md += ["\n## Cross-matcher consistency (GT-free, family-neutral)\n",
               "Median disagreement (map px) between consecutive map registrations and frame-to-frame homographies "
               "estimated by matchers of OTHER families than the method (SIFT / SuperPoint+LightGlue / LoFTR). "
               "Videos on every 2nd keyframe. Lower = smoother; it does not measure absolute accuracy (Spearman "
               "correlation with the true error over methods on the synthetic sets is only "
               + ", ".join(f"{s}: {g['cons_other'].corr(g['gt_err'], method='spearman'):.2f}"
                           for s, g in cx.groupby("set") if "gt_err" in g and g["gt_err"].notna().sum() > 3)
               + ").\n",
               "| method | " + " | ".join(piv.columns) + " |\n|---|" + "---|" * len(piv.columns) + "\n"
               + "".join(f"| {l} | " + " | ".join(fmt(v, 3) for v in r) + " |\n" for l, r in piv.iterrows())]
    nv = p[p["tag"].isin(["reference", "methods"])].dropna(subset=[f"{NEW_VIDEO}.reg"])
    nv = nv[~nv["label"].isin(EXCLUDED)]
    if len(nv):
        nv = nv.sort_values(f"{NEW_VIDEO}.ref_dev_median_px")
        md += [f"\n## New video `{NEW_VIDEO}` (strong roll + perspective; every 2nd keyframe)\n", table(nv, NEW_COLS)]
    sub = p[p["tag"] == "methods_sub4"]
    if len(sub):
        md += ["\n## Videos, every 4th keyframe (incl. full RoMa)\n",
               table(sub, [("video0061.reg", "v61 reg", True, 0), ("video0062.reg", "v62 reg", True, 0),
                           ("real.cons_median_px", "cons, px", False, 3),
                           ("video0062.ref_dev_median_px", "v62 dev, px", False, 2), ("fps_mean", "fps", False, 2)])]
    abl_groups = {}
    for tag in [t for t in p["tag"].unique() if t not in ("reference", "methods", "methods_sub4", "smoke")]:
        g = p[p["tag"] == tag]
        bases = []
        for base_cfg in g["config"].unique():  # one base per method of the group
            if base_cfg == "baseline_sift":
                bases.append(p[p["tag"] == "reference"])
            else:
                bases.append(p[(p["tag"] == "methods") & (p["config"] == base_cfg)])
        base = pd.concat(bases)
        both = pd.concat([base.assign(label=base["label"] + " (base)"), g])
        md += [f"\n## {tag}\n", table(both)]
        for cfg_name, gg in both.groupby("config", sort=False):
            vals = gg.dropna(subset=["syn_hard.center_err_median"])
            if len(vals) < 2:
                continue
            abl_groups[f"{tag}: {cfg_name}"] = pd.DataFrame({
                "label": [_variant(l) for l in vals["label"]],
                "value": vals["syn_hard.center_err_median"].to_numpy(),
                "text": [f"{e:.2f} px · reg {100 * r:.0f}% · {f:.1f} fps" for e, r, f in
                         zip(vals["syn_hard.center_err_median"], vals["syn_hard.reg"], vals["fps_mean"])]})
    Path(a.out).write_text("\n".join(md), encoding="utf-8")
    p.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    fig_accuracy(methods, FIG / "3_methods_accuracy.png")
    fig_pareto(methods, FIG / "3_methods_pareto.png")
    for old in FIG.glob("3_ablation*.png"):
        old.unlink()
    for tag in dict.fromkeys(k.split(":")[0] for k in abl_groups):
        fig_ablations({k: v for k, v in abl_groups.items() if k.startswith(tag + ":")}, FIG / f"3_ablation_{tag}.png")
    print(f"report: {a.out}; {len(p)} configurations, {len(df)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
