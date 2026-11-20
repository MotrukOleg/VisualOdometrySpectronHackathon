"""Report of the exact-GT pair benchmark (scripts/run_pair_benchmark.py): tables + a figure for the slides.

    python scripts/make_pair_report.py experiments/pairs/finetune_pos_crops/results_*.csv
writes experiments/pairs/<name>/report.md and docs/presentation/figures/4_pairs_success.png
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

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "presentation" / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
VAR_COL = {"plain": "#2a78d6", "geo": "#e07a1f", "hard": "#7a4fc9"}
VAR_DESC = {"plain": "as is (ortho vs ortho, shift)", "geo": "+ rotation ±15°, scale 0.75–1.3",
            "hard": "+ screen degradations"}
SEQ_DESC = {"01": "suburb", "02": "fields", "03": "town + river", "04": "industrial / wetland", "05": "fields + forest",
            "08": "water + fields"}
EXCLUDED = {"efficient_loftr_track": "transformers port (coarse-only matches)"}


def summary(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        e = g["err_px"].to_numpy(float)
        return pd.Series({"pairs": len(g), "ok": np.isfinite(e).mean(), "@1px": (e <= 1).mean(), "@3px": (e <= 3).mean(),
                          "@10px": (e <= 10).mean(), "median_px": float(np.median(e)),
                          "inliers": g["inliers"].mean(), "ms": g["ms"].median()})
    return df.groupby(["method", "variant"])[["err_px", "inliers", "ms"]].apply(agg).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    a = ap.parse_args()
    files = [p for pat in a.csv for p in (glob.glob(pat) or ([pat] if Path(pat).exists() else []))]

    def read(fs):
        if not fs:
            return pd.DataFrame(columns=["method", "variant", "img0", "seq", "err_px", "inliers", "ms", "machine"])
        d = pd.concat([pd.read_csv(p, dtype={"seq": str}).assign(machine="local" if "local" in Path(p).name else "gpu")
                       for p in fs], ignore_index=True)
        d["seq"] = d["seq"].str.zfill(2)
        return d.drop_duplicates(["method", "variant", "img0"], keep="last")

    # the slow methods ran on a separate, smaller selection ("results_C*", "small*" files: --max-pairs 120) whose
    # geo/hard views are seeded differently, so it is reported on its own, with reference methods on the same views
    is_small = [("results_C" in Path(p).name or "small" in Path(p).name) for p in files]
    df = read([p for p, sm in zip(files, is_small) if not sm])
    small = read([p for p, sm in zip(files, is_small) if sm])
    out = Path(files[0]).with_name("report.md")
    machine = pd.concat([df, small]).groupby("method")["machine"].agg(lambda x: x.mode().iloc[0])
    common = []  # fair comparison: every method on the pairs all methods ran (per variant)
    for v, g in df.groupby("variant"):
        keep = set.intersection(*[set(gg["img0"]) for _, gg in g.groupby("method")])
        common.append(g[g["img0"].isin(keep)])
    df = pd.concat(common)
    s = summary(df)
    variants = [v for v in ("plain", "geo", "hard") if v in s["variant"].unique()]
    wide = s.pivot(index="method", columns="variant")
    order = wide[("@3px", "hard")].sort_values(ascending=False).index if "hard" in variants else wide.index

    md = ["# Pair benchmark with exact ground truth (`data/_finetune_pos_crops`)\n",
          "Consecutive crops of one north-up orthophoto; the true shift is found by an exhaustive pixel-exact search "
          "(no feature matcher involved). Sequence 11 has no overlapping consecutive crops and is excluded. "
          "Error = mean corner error of img1 (512 px), failed estimate = ∞. One estimator for all methods "
          "(homography, MAGSAC++, 3 px). Variants: "
          + "; ".join(f"`{v}` {VAR_DESC[v]}" for v in variants) + ".\n"]
    def table(w, ords, mach):
        head = "| method | " + " | ".join(f"{v} @1px | {v} @3px | {v} med px" for v in variants) \
               + " | pairs | ms (plain) | machine |\n"
        head += "|---|" + "---|" * (3 * len(variants) + 3) + "\n"
        body = ""
        for m in ords:
            cells = []
            for v in variants:
                if ("@3px", v) not in w.columns or pd.isna(w.loc[m, ("@3px", v)]):
                    cells += ["–", "–", "–"]
                    continue
                cells += [f"{100 * w.loc[m, ('@1px', v)]:.0f}%", f"{100 * w.loc[m, ('@3px', v)]:.0f}%",
                          f"{w.loc[m, ('median_px', v)]:.2f}"]
            note = " ²" if m in EXCLUDED else ""
            body += f"| {m}{note} | " + " | ".join(cells) + f" | {int(w.loc[m, ('pairs', variants[0])])} | " \
                    f"{w.loc[m, ('ms', variants[0])]:.0f} | {mach[m]} |\n"
        return head + body

    md += ["Pairs per variant (common to all methods): "
           + ", ".join(f"{v} {df[df['variant'] == v]['img0'].nunique()}" for v in variants)
           + ". ms = median time per pair; `gpu` = RTX 5060 Ti server (shared), `local` = GTX 1650 dev PC — "
             "times are comparable only within one machine.\n",
           table(wide, order, machine), "\n² " + "; ".join(f"{k}: {v}" for k, v in EXCLUDED.items()) + "\n"]
    if len(small):
        ws = summary(small).pivot(index="method", columns="variant")
        last = [v for v in variants if ("@3px", v) in ws.columns][-1]
        md += ["\n## Slow methods on a 120-pair selection (references on exactly the same views)\n",
               table(ws, ws[("@3px", last)].sort_values(ascending=False).index, machine)]

    # per terrain (sequence), success @3px
    for v in variants:
        t = df[df["variant"] == v].groupby(["method", "seq"])["err_px"].apply(lambda e: (e <= 3).mean()).unstack()
        t = t.loc[[m for m in order if m in t.index]]
        md += [f"\n## Success @3px by terrain, variant `{v}`\n",
               "| method | " + " | ".join(f"{c} {SEQ_DESC.get(c, '')}" for c in t.columns) + " |\n|---|"
               + "---|" * len(t.columns) + "\n"
               + "".join(f"| {m} | " + " | ".join(f"{100 * x:.0f}%" for x in r) + " |\n" for m, r in t.iterrows())]
    out.write_text("\n".join(md), encoding="utf-8")
    s.to_csv(out.with_name("summary.csv"), index=False)

    # figure: success @3px per variant, methods sorted by the hard variant
    keep = [m for m in order if m not in EXCLUDED]
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(keep) + 2.4), dpi=150, facecolor=SURF)
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, zorder=0)
    y = np.arange(len(keep))[::-1]
    hgt = 0.8 / len(variants)
    for k, v in enumerate(variants):
        vals = [wide.loc[m, ("@3px", v)] * 100 for m in keep]
        ax.barh(y + (len(variants) / 2 - k - 0.5) * hgt, vals, height=hgt, color=VAR_COL[v], edgecolor=SURF,
                linewidth=1.5, zorder=3, label=f"{v}: {VAR_DESC[v]}")
    ax.set_yticks(y, [m.replace("_track", "") for m in keep], fontsize=10, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("pairs with corner error ≤ 3 px, %", color=INK2)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.suptitle(f"Exact-GT pair benchmark: orthophoto crops ({df['seq'].nunique()} terrains, "
                 f"{df[df['variant'] == variants[0]]['img0'].nunique()} pairs per variant)", x=0.02, ha="left",
                 fontsize=15, color=INK)
    ax.set_title("sorted by the hardest variant (rotation + scale + screen degradations)", loc="left", fontsize=10,
                 color=INK2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "4_pairs_success.png", facecolor=SURF)
    print(f"report: {out}")
    print(s.sort_values(["variant", "@3px"], ascending=[True, False]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
