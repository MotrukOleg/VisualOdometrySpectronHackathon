"""Comparison on the real flight UAV-VisLoc 24: every precise matcher in three regimes + retrieval embedders.

    tracking  : scripts/localize.py, track mode (experiments/results/visloc24_<m>/metrics.json)
    GT hint   : scripts/oracle_roi.py, ROI centred on the true GPS position (experiments/oracle_roi/visloc_24/)
    islands   : scripts/localize_islands.py, fine-tuned DINOv2 retrieval + the matcher as stage-3b check
                (experiments/islands/visloc_24_dinoft_<m>/)
Retrieval: experiments/retrieval/summary.csv (scripts/eval_retrieval.py) + island runs without a check.

Writes experiments/islands/report.md, experiments/islands/methods_visloc24.csv and the slide figure
docs/presentation/figures/7_methods_visloc24.png.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
LABELS = {"track_sift": "SIFT+FLANN", "track_sift_lg": "SIFT+LightGlue", "track_superpoint_lg": "SuperPoint+LG",
          "track_aliked_lg": "ALIKED+LG", "track_disk_lg": "DISK+LG", "track_xfeat": "XFeat",
          "track_eloftr_official": "Efficient LoFTR", "track_loftr": "LoFTR", "track_superglue_official": "SuperGlue"}


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def methods_table():
    oracle = pd.read_csv(EXP / "oracle_roi" / "visloc_24" / "summary.csv").set_index("method")
    rows = []
    for m, lab in LABELS.items():
        tr = load_json(EXP / "results" / f"visloc24_{m}" / "metrics.json")
        isl = load_json(EXP / "islands" / f"visloc_24_dinoft_{m}" / "metrics.json")
        o = oracle.loc[m] if m in oracle.index else None
        rows.append({
            "method": lab,
            "track_registered_%": round(100 * tr["registered_rate"], 1) if tr else np.nan,
            "hint_le5m_%": o["le_5m_%"] if o is not None else np.nan,
            "hint_le15m_%": o["le_15m_%"] if o is not None else np.nan,
            "islands_correct": isl["islands_correct"] if isl else np.nan,
            "islands_le5m_%": isl["le_5m_%"] if isl else np.nan,
            "islands_le15m_%": isl["le_15m_%"] if isl else np.nan,
            "islands_le50m_%": isl["le_50m_%"] if isl else np.nan,
            "islands_median_m": isl["err_m_median_placed"] if isl else np.nan,
            "check_s": isl.get("verify_s") if isl else np.nan,
        })
    return pd.DataFrame(rows)


def retrieval_table():
    r = pd.read_csv(EXP / "retrieval" / "summary.csv")
    r = r[r.dataset == "visloc_24"]
    piv = r.pivot_table(index="embedder", columns="frames", values=["recall@1_%", "recall@10_%"])
    piv.columns = [f"{a.replace('_%', '')}_{b}" for a, b in piv.columns]
    names = {"thumb": "мініатюра 16×16 (класика)"}
    out = []
    for e, row in piv.iterrows():
        lab = names.get(e, e.replace("dinov2[dinov2-small,224,", "DINOv2-S ").replace("]", "")
                        .replace("head,ft=dinov2s_ft", "донавчений"))
        runs = {"thumb": "visloc_24_thumb_noverify", "dinov2[dinov2-small,224,cls]": "visloc_24_dinov2_pooling_cls_noverify",
                "dinov2[dinov2-small,224,head,ft=dinov2s_ft]": "visloc_24_dinoft_noverify"}
        isl = load_json(EXP / "islands" / runs[e] / "metrics.json") if e in runs else None
        out.append({"embedder": lab, "R@1_all_%": row.get("recall@1_all"), "R@10_all_%": row.get("recall@10_all"),
                    "R@1_structured_%": row.get("recall@1_matchable"), "R@10_forest_%": row.get("recall@10_unmatchable"),
                    "islands_correct": isl["islands_correct"] if isl else np.nan,
                    "islands_le15m_%": isl["le_15m_%"] if isl else np.nan,
                    "islands_le50m_%": isl["le_50m_%"] if isl else np.nan})
    return pd.DataFrame(out).sort_values("R@10_all_%")


def to_md(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    fmt = lambda v: "–" if isinstance(v, float) and np.isnan(v) else (f"{v:g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def figure(df, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.sort_values("islands_le5m_%")
    y = np.arange(len(df))
    h = 0.38
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    series = [("hint_le5m_%", "кадр окремо, навіть якщо підказати правильне місце", "#9cc3ee"),
              ("islands_le5m_%", "острови + DINOv2 (донавчений) + метод", "#2a78d6")]
    for i, (col, lab, c) in enumerate(series):
        v = df[col].fillna(0).to_numpy()
        off = h / 2 if i == 0 else -h / 2
        ax.barh(y + off, v, h * 0.92, color=c, label=lab)
        if col == "islands_le5m_%":
            for yy, vv in zip(y, v):
                ax.text(vv + 0.8, yy + off, f"{vv:.0f}%", va="center", fontsize=13, color="#0b0b0b")
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"], fontsize=15)
    ax.set_xlabel("кадрів з похибкою ≤ 5 м, %", fontsize=15)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", color="#e4e3df")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=14, frameon=False)
    ax.set_title("Реальний політ UAV-VisLoc 24 (Перу, 235 кадрів): частка кадрів з похибкою ≤ 5 м\n"
                 "звичайний трекінг: 0 % у всіх методів (старт через глобальний SIFT не знаходить місця)",
                 fontsize=16, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    m = methods_table()
    r = retrieval_table()
    m.to_csv(EXP / "islands" / "methods_visloc24.csv", index=False)
    md = ["# Реальний політ UAV-VisLoc 24: порівняння методів\n",
          "## Точні матчери в трьох режимах\n", to_md(m), "",
          "## Ембединги для пошуку тайлів (етап 2)\n", to_md(r), ""]
    (EXP / "islands" / "report.md").write_text("\n".join(md), encoding="utf-8")
    figure(m, ROOT / "docs" / "presentation" / "figures" / "7_methods_visloc24.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
