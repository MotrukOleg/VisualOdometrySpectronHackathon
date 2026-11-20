"""Cross-matcher consistency: a GT-free localisation check that does not favour one feature family.

The plain consistency check compares the map registrations of consecutive frames with a frame-to-frame homography
estimated by SIFT. A SIFT-based localisation shares the errors of that reference, so it looks better than it is.
Here the frame-to-frame reference is computed by three matchers of different families (SIFT, SuperPoint+LightGlue,
LoFTR), and every trajectory is checked against each of them:

    cons_<m>      median disagreement (map px) against the f2f of matcher m
    cons_all      mean of the three medians
    cons_other    mean over the matchers of a DIFFERENT family than the method (the fair one; used for ranking)

On the synthetic sets the script also reports how well each metric ranks the methods like the true centre error
(Spearman correlation over methods), which shows whether the cross check removes the bias.

Trajectories are read from experiments/results/bench/{methods,reference}/<label>/<dataset>/trajectory.csv under each
--roots directory; videos are evaluated on every 2nd keyframe (runs on the full video are sub-sampled by frame name).

    python scripts/cross_consistency.py --roots experiments/results/bench experiments/results/bench_remote
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vo.consistency import F2F_MATCHERS, consistency, frame_to_frame  # noqa: E402
from vo.io import FrameSource  # noqa: E402
from vo.schema import read_trajectory  # noqa: E402

EVAL = {  # evaluation set -> (dataset key in configs/datasets.yaml, step)
    "crops": ("crops", 1), "syn_easy": ("syn_easy", 1), "syn_hard": ("syn_hard", 1),
    "video0061": ("video0061", 2), "video0062": ("video0062", 2), "video214816": ("video214816", 2),
}


def family(label: str) -> str | None:
    if label.startswith(("sift", "rootsift", "map_baseline_sift")):
        return "sift"
    if label.startswith("superpoint"):
        return "splg"
    if label.startswith(("loftr", "efficient_loftr")):
        return "loftr"
    return None


def find_runs(roots, ev: str) -> dict[str, Path]:
    """label -> trajectory.csv; the sub-sampled ('…s2') run wins over the full one, then the newest file."""
    found = {}
    for root in roots:
        for tag in ("methods", "reference"):
            for ds in (ev + "s2", ev):
                for p in Path(root).glob(f"{tag}/*/{ds}/trajectory.csv"):
                    label = p.parent.parent.name
                    rank = (ds.endswith("s2"), p.stat().st_mtime)
                    if label not in found or rank > found[label][0]:
                        found[label] = (rank, p)
    return {k: v[1] for k, v in found.items()}


def align(traj: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    t = traj.copy()
    t["name"] = t["name"].astype(str).map(lambda s: Path(s).stem)
    t = t.drop_duplicates("name").set_index("name").reindex(names).reset_index()
    t["status"] = t["status"].fillna("lost")
    return t


def gt_error(t: pd.DataFrame, gt_path: str) -> float:
    gt = pd.read_csv(ROOT / gt_path)
    key = "name" if "name" in gt.columns else "frame"
    if key == "name":
        gt["name"] = gt["name"].astype(str).map(lambda s: Path(s).stem)
        g = gt.set_index("name").reindex(t["name"])
    else:
        g = gt.set_index("frame").reindex(t["frame"])
    ok = t["status"].isin(["map", "interp"]).to_numpy()
    e = np.hypot(t["x"].to_numpy(float) - g["x"].to_numpy(float), t["y"].to_numpy(float) - g["y"].to_numpy(float))
    return float(np.nanmedian(e[ok])) if ok.any() else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[str(ROOT / "experiments" / "results" / "bench")])
    ap.add_argument("--sets", nargs="+", default=list(EVAL))
    ap.add_argument("--matchers", nargs="+", default=list(F2F_MATCHERS))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "cross_consistency.csv"))
    a = ap.parse_args()
    with open(ROOT / "configs" / "datasets.yaml", encoding="utf-8") as f:
        datasets = yaml.safe_load(f)
    cache = ROOT / "experiments" / "cache"
    rows = []
    for ev in a.sets:
        key, step = EVAL[ev]
        d = datasets[key]
        src = FrameSource(ROOT / d["frames"], step=step)
        names = [src.name(i) for i in range(len(src))]
        size = FrameSource(ROOT / d["frames"]).size(0)
        f2f = {}
        for m in a.matchers:
            tag = "" if m == "sift" else f"_{m}"
            path = cache / (f"f2f_{key}{'s2' if step == 2 else ''}{tag}.npz")
            frames = str(ROOT / d["frames"]) + (f"#step{step}" if step > 1 else "")
            f2f[m] = frame_to_frame(frames, path, step=step, matcher=m)
            print(f"{ev:12s} f2f {m:5s}: {len(f2f[m])}/{len(names) - 1} pairs", flush=True)
        for label, p in sorted(find_runs(a.roots, ev).items()):
            t = align(read_trajectory(p), names)
            r = {"set": ev, "label": label, "family": family(label), "reg": float((t["status"] == "map").mean()),
                 "run": str(p.parent)}
            for m in a.matchers:
                c = consistency(t, f2f[m], size)
                r[f"cons_{m}"] = c.get("cons_median_px", np.nan)
                r[f"cons_{m}_p95"] = c.get("cons_p95_px", np.nan)
            meds = [r[f"cons_{m}"] for m in a.matchers]
            r["cons_all"] = float(np.nanmean(meds))
            other = [r[f"cons_{m}"] for m in a.matchers if m != r["family"]]
            r["cons_other"] = float(np.nanmean(other))
            if d.get("gt"):
                r["gt_err"] = gt_error(t, d["gt"])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = ["label", "reg"] + [f"cons_{m}" for m in a.matchers] + ["cons_all", "cons_other"]
    for ev, g in df.groupby("set", sort=False):
        cols = show + (["gt_err"] if "gt_err" in g and g["gt_err"].notna().any() else [])
        print(f"\n== {ev}\n" + g.sort_values("cons_other")[cols].round(3).to_string(index=False))
        if "gt_err" in g and g["gt_err"].notna().sum() > 3:
            corr = {c: g[c].corr(g["gt_err"], method="spearman") for c in cols[2:-1]}
            print("Spearman vs GT error: " + ", ".join(f"{k}={v:.2f}" for k, v in corr.items()))
    print(f"\nsaved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
