"""Numbers for the reveal.js slides, generated from the benchmark outputs (never typed by hand).

Reads experiments/map_report.csv, the any-area CSVs, the pair-benchmark summary and the final runs' metrics.json,
and writes docs/presentation/data.js (`window.SLIDES = {...}`), which index.html uses to fill [data-k] elements and
[data-table] tables. A .js file (not .json) so the slides also open from disk without a web server.

    python scripts/make_slides_data.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "presentation" / "data.js"

NAMES = {"map_baseline_sift": "SIFT + FLANN (бейзлайн, вся мапа)", "sift_flann_global": "SIFT + FLANN (вся мапа)",
         "sift_flann_track": "SIFT + FLANN (трекінг)", "sift_lightglue_track": "SIFT + LightGlue",
         "superpoint_lightglue_track": "SuperPoint + LightGlue", "disk_lightglue_track": "DISK + LightGlue",
         "aliked_lightglue_track": "ALIKED + LightGlue", "loftr_track": "LoFTR",
         "efficient_loftr_official_track": "Efficient LoFTR", "superpoint_superglue_official_track": "SuperPoint + SuperGlue",
         "xfeat_track": "XFeat", "tiny_roma_track": "tiny-RoMa", "orb_bf_track": "ORB", "akaze_bf_track": "AKAZE"}
KEY = ["sift_lightglue_track", "disk_lightglue_track", "loftr_track", "superpoint_lightglue_track",
       "efficient_loftr_official_track", "map_baseline_sift"]
AREAS = ["area_lviv", "area_kyiv", "area_carpathians", "area_oleshky", "area_gorgany", "area_askania", "area_tylihul",
         "area_kyiv_sea", "area_poltava"]


def num(v, nd=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return int(round(float(v))) if nd == 0 else round(float(v), nd)


def pct(v):
    return None if v is None or not math.isfinite(float(v)) else int(round(100 * float(v)))


def metrics(run):
    p = EXP / "results" / run / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


REAL_NAMES = {"track_sift": "SIFT", "track_sift_lg": "SIFT + LightGlue", "track_superpoint_lg": "SuperPoint + LG",
              "track_aliked_lg": "ALIKED + LG", "track_disk_lg": "DISK + LG", "track_superglue_official": "SuperGlue",
              "track_eloftr_official": "Efficient LoFTR", "track_loftr": "LoFTR", "track_xfeat": "XFeat"}


def real_flight() -> dict:
    """Real UAV-VisLoc flight #24: registration with the oracle ROI and the GPS spoofing demo."""
    D = {}
    r = pd.read_csv(EXP / "oracle_roi" / "visloc_24" / "summary.csv").sort_values("registered_%")
    D["real_table"] = {"head": ["Метод", "прив'язано", "≤ 5 м", "медіана, м", "с/кадр"],
                       "rows": [[REAL_NAMES.get(m, m), num(p, 0), num(f, 0), num(e, 1), num(s, 2)]
                                for m, p, f, e, s in r[["method", "registered_%", "le_5m_%", "err_m_median",
                                                        "s_per_frame"]].itertuples(index=False)]}
    D["real"] = {"frames": int(r["frames"].iloc[0]), "best": num(r["registered_%"].max(), 0),
                 "sift": num(r.set_index("method").loc["track_sift", "registered_%"], 0),
                 "err": num(r.loc[r["registered"] > 0, "err_m_median"].median(), 1)}
    s = pd.read_csv(EXP / "spoofing" / "visloc_24" / "summary.csv").sort_values("detected_%", ascending=False)
    D["spoof_table"] = {"head": ["Метод", "атак виявлено", "хибні тривоги", "зсув до тривоги", "похибка: GPS",
                                 "похибка: виправлено"],
                        "rows": [[REAL_NAMES.get(m, m), num(d, 0), int(fa), num(dr, 0), num(g, 0), num(c, 1)]
                                 for m, d, fa, dr, g, c in s[["method", "detected_%", "false_alarm_honest_gps",
                                                              "drag_at_alarm_m_median", "err_after_trusting_gps_m",
                                                              "err_after_correction_m"]].itertuples(index=False)]}
    best = s.iloc[0]
    D["spoof"] = {"attacks": int(best["attacks"]), "drag": num(best["drag_at_alarm_m_median"], 0),
                  "gps": num(best["err_after_trusting_gps_m"], 0), "fixed": num(best["err_after_correction_m"], 1)}
    return D


def main():
    if "--real-only" in sys.argv:  # refresh only the real-flight keys (no local benchmark results needed)
        text = OUT.read_text(encoding="utf-8")
        head, body = text.split("window.SLIDES = ", 1)
        D = json.loads(body.rstrip().rstrip(";"))
        D.update(real_flight())
        OUT.write_text(head + "window.SLIDES = " + json.dumps(D, ensure_ascii=False, indent=1) + ";\n",
                       encoding="utf-8")
        print(f"updated real-flight keys in {OUT}")
        return 0
    D = {}
    rep = pd.read_csv(EXP / "map_report.csv")
    meth = rep[rep["tag"].isin(["methods", "reference"])].drop_duplicates("label").set_index("label")
    uniq = rep.drop_duplicates("label").set_index("label")

    # --- final method on the task data
    fin = {}
    for k in ("crops", "video0061", "video0062", "video214816"):
        m = metrics(f"final_{k}")
        fin[k] = {"reg": m.get("n_registered"), "n": m.get("n_frames"), "inl": num(m.get("mean_inliers"), 0),
                  "fps": num(m.get("fps"), 1)}
    D["final"] = fin
    base = metrics("bench/reference/map_baseline_sift/crops")
    D["baseline_crops"] = {"reg": base.get("n_registered"), "n": base.get("n_frames"),
                           "inl": num(base.get("mean_inliers"), 0), "fps": num(base.get("fps"), 1)}

    # --- map benchmark (synthetic GT + own videos)
    rows = []
    for lab in KEY:
        if lab not in meth.index:
            continue
        r = meth.loc[lab]
        rows.append([NAMES.get(lab, lab), num(r.get("syn_easy.center_err_median")),
                     num(r.get("syn_hard.center_err_median")), pct(r.get("syn_hard.reg")),
                     num(r.get("real.cons_median_px"), 3), num(r.get("fps_mean"), 1)])
    D["map_table"] = {"head": ["Метод", "syn_easy, px", "syn_hard, px", "syn_hard прив'язано",
                               "узгодженість на реальних, px", "кадр/с"], "rows": rows}

    # --- ablations (syn_hard median error)
    def abl(label):
        src = meth if label in meth.index else uniq  # a method's own row first (tags share labels)
        return num(src["syn_hard.center_err_median"].get(label)) if label in src.index else None
    D["abl"] = {
        "rot_sift": [abl("sift_flann_track"), abl("sift_flann_track[tracking.rotation=none]"),
                     abl("sift_flann_track[tracking.rotation=test]")],
        "rot_splg": [abl("superpoint_lightglue_track"), abl("superpoint_lightglue_track[tracking.rotation=none]"),
                     abl("superpoint_lightglue_track[tracking.rotation=test]")],
        "scale": {"s" + s.replace(".", ""): abl(f"map_baseline_sift[localize.frame_scale={s}]")
                  for s in ("0.25", "0.35", "1.0")},
        "scale_fps": {"s" + s.replace(".", ""): num(uniq["fps_mean"].get(f"map_baseline_sift[localize.frame_scale={s}]"), 0)
                      for s in ("0.25", "0.35", "1.0")},
        "savgol": [abl("map_baseline_sift"), abl("map_baseline_sift[smoothing.savgol_window=7]"),
                   abl("map_baseline_sift[smoothing.savgol_window=15]")],
    }

    # --- exact-GT pairs
    ps = pd.read_csv(EXP / "pairs" / "finetune_pos_crops" / "summary.csv")
    pw = ps.pivot(index="method", columns="variant", values="@3px")
    pm = ps.pivot(index="method", columns="variant", values="median_px")
    D["pairs_n"] = int(ps[ps["variant"] == "plain"]["pairs"].max())
    D["pairs_table"] = {"head": ["Метод", "як є", "+ поворот і масштаб", "+ спотворення з екрана", "похибка (поворот), px"],
                        "rows": [[NAMES.get(m, m), pct(pw.loc[m, "plain"]), pct(pw.loc[m, "geo"]),
                                  pct(pw.loc[m, "hard"]), num(pm.loc[m, "geo"])]
                                 for m in ["sift_lightglue_track", "efficient_loftr_official_track", "loftr_track",
                                           "disk_lightglue_track", "sift_flann_track", "superpoint_lightglue_track"]
                                 if m in pw.index]}

    # --- any-area benchmark
    files = [EXP / "map_benchmark.csv"] + sorted(EXP.glob("map_benchmark_areas_*.csv"))
    a = pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
    a = a[a["dataset"].astype(str).isin(AREAS) & a["tag"].isin(["areas", "reference"])]
    a["time"] = pd.to_datetime(a["time"])
    a = a.sort_values("time").drop_duplicates(["label", "dataset"], keep="last")
    a = a[a["status"] == "ok"]
    s5 = a.pivot(index="label", columns="dataset", values="success@5px").reindex(columns=AREAS)
    D["areas_n"] = len(AREAS)
    D["areas_runs"] = int(len(a))
    D["areas_table"] = {"head": ["Метод", "середнє", "ліс", "степ", "лиман", "вода", "поля"],
                        "rows": [[NAMES.get(m, m), pct(s5.loc[m].mean()), pct(s5.loc[m, "area_gorgany"]),
                                  pct(s5.loc[m, "area_askania"]), pct(s5.loc[m, "area_tylihul"]),
                                  pct(s5.loc[m, "area_kyiv_sea"]), pct(s5.loc[m, "area_poltava"])]
                                 for m in ["loftr_track", "disk_lightglue_track", "sift_lightglue_track",
                                           "map_baseline_sift"] if m in s5.index]}

    # --- decision table: one line per candidate over the three tests
    def dec(lab):
        r = meth.loc[lab] if lab in meth.index else {}
        pl = lab if lab != "map_baseline_sift" else "sift_flann_track"
        return [NAMES.get(lab, lab), num(r.get("syn_hard.center_err_median")) if len(r) else None,
                pct(pw.loc[pl, "hard"]) if pl in pw.index else None,
                pct(s5.loc[lab].mean()) if lab in s5.index else None,
                pct(s5.loc[lab, "area_poltava"]) if lab in s5.index else None]
    D["decision"] = {"head": ["Метод", "syn_hard, px", "пари: найважчий варіант ≤ 3 px", "райони: у межах 5 px",
                              "поля (Полтава)"],
                     "rows": [dec(m) for m in ["sift_lightglue_track", "disk_lightglue_track", "loftr_track",
                                               "map_baseline_sift"]]}

    # --- concrete metrics -----------------------------------------------------------------------------------------
    # test 1: synthetic flight "from a screen" (exact GT, map px) + speed / memory
    t1 = []
    for lab in KEY:
        if lab not in meth.index:
            continue
        r = meth.loc[lab]
        t1.append([NAMES.get(lab, lab), num(r.get("syn_hard.center_err_median")), num(r.get("syn_hard.center_err_mean")),
                   num(r.get("syn_hard.center_err_max")), pct(r.get("syn_hard.success@5px")),
                   pct(r.get("syn_hard.success@10px")), num(r.get("syn_easy.center_err_median")),
                   num(r.get("fps_mean"), 1), num(r.get("vram_max_mb"), 0) or "CPU"])
    D["t1_full"] = {"head": ["Метод", "медіана", "середнє", "максимум", "≤ 5 px", "≤ 10 px", "syn_easy медіана",
                             "кадр/с", "VRAM, МБ"], "rows": t1}

    # test 3: frame-level errors over all 9 areas, in map px and metres (each map has its own m/px)
    import yaml
    dsets = yaml.safe_load(open(ROOT / "configs" / "datasets.yaml", encoding="utf-8"))

    def traj_path(label, ds):
        tag = "reference" if label == "map_baseline_sift" else "areas"
        c = [q for root in (EXP / "results" / "bench", EXP / "remote_snapshot" / "results" / "bench")
             for q in [root / tag / label / ds / "trajectory.csv"] if q.exists()]
        return max(c, key=lambda q: q.stat().st_mtime) if c else None

    t3 = []
    for lab in ["sift_lightglue_track", "disk_lightglue_track", "loftr_track", "map_baseline_sift"]:
        e_px, e_m, n_all, n_pos, n_map = [], [], 0, 0, 0
        for ds in AREAS:
            tp = traj_path(lab, ds)
            if tp is None:
                continue
            g = pd.read_csv(ROOT / dsets[ds]["gt"])
            mpp = json.loads((ROOT / Path(dsets[ds]["map"]).parent / "geo.json").read_text(encoding="utf-8"))["meters_per_px"]
            t = pd.read_csv(tp).merge(g[["name", "x", "y"]], on="name", suffixes=("", "_gt"))
            pos = t["status"].isin(["map", "interp"]) & t["x"].notna()
            e = np.hypot(t["x"] - t["x_gt"], t["y"] - t["y_gt"]).where(pos, np.inf).to_numpy(float)
            e_px.append(e)
            e_m.append(e * mpp)
            n_all += len(t)
            n_pos += int(pos.sum())
            n_map += int((t["status"] == "map").sum())
        e, em = np.concatenate(e_px), np.concatenate(e_m)
        okf = np.isfinite(e)
        t3.append([NAMES.get(lab, lab), pct((e <= 5).mean()), pct((e <= 10).mean()), pct((e <= 25).mean()),
                   num(np.median(em[okf]), 2), num(np.percentile(em[okf], 95), 1), pct(n_map / n_all),
                   pct(1 - n_pos / n_all)])
    D["t3_full"] = {"head": ["Метод", "≤ 5 px (≈4 м)", "≤ 10 px", "≤ 25 px", "медіана, м", "95-й перц., м",
                             "прив'язано", "без позиції"], "rows": t3, "frames": int(n_all)}

    # pairs: precision and robustness
    p1 = ps.pivot(index="method", columns="variant", values="@1px")
    pms = ps.pivot(index="method", columns="variant", values="ms")
    D["pairs_full"] = {"head": ["Метод", "геометрія: ≤ 1 px", "≤ 3 px", "медіана, px", "спотворення: ≤ 3 px",
                                "медіана, px", "мс на пару"],
                       "rows": [[NAMES.get(m, m), pct(p1.loc[m, "geo"]), pct(pw.loc[m, "geo"]), num(pm.loc[m, "geo"]),
                                 pct(pw.loc[m, "hard"]), num(pm.loc[m, "hard"], 1) if np.isfinite(pm.loc[m, "hard"])
                                 else "∞", num(pms.loc[m, "plain"], 0)]
                                for m in ["sift_lightglue_track", "efficient_loftr_official_track", "loftr_track",
                                          "disk_lightglue_track", "sift_flann_track", "superpoint_lightglue_track"]
                                if m in pw.index]}

    # final method, per dataset: registration, quality and agreement with the baseline
    sys.path.insert(0, str(ROOT / "src"))
    from vo.consistency import agreement, consistency, frame_to_frame
    from vo.io import FrameSource
    from vo.schema import read_trajectory
    fr_dirs = {"crops": "data/train/crops", "video0061": "data/video/IMG_0061_every5",
               "video0062": "data/video/IMG_0062_every5", "video214816": "data/video/VID_214816_every5"}
    fnames = {"crops": "50 кадрів завдання", "video0061": "IMG_0061", "video0062": "IMG_0062 (вертикальне)",
              "video214816": "третє відео (крен)"}
    t4 = []
    for k, fd in fr_dirs.items():
        m = metrics(f"final_{k}")
        tr = read_trajectory(EXP / "results" / f"final_{k}" / "trajectory.csv")
        ref = read_trajectory(EXP / "results" / "bench" / "reference" / "map_baseline_sift" / k / "trajectory.csv")
        f2f = frame_to_frame(ROOT / fd, EXP / "cache" / f"f2f_{k}.npz")
        c = consistency(tr, f2f, FrameSource(ROOT / fd).size(0))
        a = agreement(tr, ref)
        t4.append([fnames[k], f"{m['n_registered']}/{m['n_frames']}", num(m.get("mean_inliers"), 0),
                   num(m.get("mean_reproj_err"), 2),
                   f"{num(c.get('cons_median_px'), 2)} / {num(c.get('cons_p95_px'), 2)}",
                   f"{num(a.get('ref_dev_median_px'), 2)} / {num(a.get('ref_dev_max_px'), 2)}", num(m.get("fps"), 1),
                   num(m.get("vram_peak_mb"), 0)])
    D["final_full"] = {"head": ["Набір", "прив'язано", "inliers", "репроєкція, px", "узгодженість мед. / p95, px",
                                "Δ від бейзлайну мед. / макс., px", "кадр/с", "VRAM, МБ"], "rows": t4}

    D.update(real_flight())
    D["counts"] = {"crops": 50, "videos": 3, "methods": int(len(meth)), "pairs": D["pairs_n"],
                   "areas": D["areas_n"], "keyframes": sum(v["n"] or 0 for k, v in fin.items() if k != "crops")}
    OUT.write_text("// generated by scripts/make_slides_data.py — do not edit by hand\nwindow.SLIDES = "
                   + json.dumps(D, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps({k: D[k] for k in ("final", "baseline_crops", "abl", "decision", "counts")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
