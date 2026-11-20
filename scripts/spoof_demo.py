"""GPS spoofing demo on a real flight: a fake GPS drag is caught by cross-checking against visual map fixes.

Inputs: the flight GT (name, lon, lat; the real GPS) and per-frame visual fixes of a method (name, x, y, ok in map
px, e.g. experiments/oracle_roi/visloc_24/track_xfeat.csv). The real GPS is dragged away from frame --start; the
detector (vo.spoofing) compares it with the visual fixes. The demo is repeated for every spoofing start (--sweep)
and every method to measure detection delay, missed attacks and false alarms on the honest GPS.

Assumption: the visual fixes come from a search window around the expected position (Oracle ROI = the true one);
in flight this prior comes from inertial dead reckoning, not from GPS, so the spoofer cannot steer it.

Writes to --out: summary.csv (one row per method), sweep.csv (one row per method x start) and, with --figure,
the demo plot for the first method.

Example:
    python scripts/spoof_demo.py --gt data/real/visloc_24/flight/gt.csv --geo data/real/visloc_24/geo.json \
        --map data/real/visloc_24/map.jpg --visual experiments/oracle_roi/visloc_24/track_xfeat.csv \
        experiments/oracle_roi/visloc_24/track_loftr.csv --figure docs/presentation/figures/4_spoofing_visloc_24.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.spoofing import detect, spoof, to_enu  # noqa: E402
from vo.tiles import GeoRef  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a"


def load_visual(path: Path, names: pd.Series, geo: GeoRef, origin) -> np.ndarray:
    v = pd.read_csv(path).set_index("name").reindex(names.astype(str))
    ok = v["ok"].fillna(False).astype(bool).to_numpy()
    lon, lat = geo.to_lonlat(v["x"].to_numpy(float), v["y"].to_numpy(float))
    enu = to_enu(lon, lat, *origin)
    enu[~ok] = np.nan
    return enu


def plot(path, a, geo, gt, truth, gps, visual, det, origin, method):
    import matplotlib.pyplot as plt

    from vo.io import imread

    def px(enu):  # local metres -> map px
        lat = origin[1] + np.degrees(enu[:, 1] / 6_371_008.8)
        lon = origin[0] + np.degrees(enu[:, 0] / (6_371_008.8 * np.cos(np.radians(origin[1]))))
        return np.stack(geo.to_px(lon, lat), 1)

    plt.rcParams.update({"font.size": 15})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={"width_ratios": [1.45, 1]})
    m = imread(a.map)[:, :, ::-1]
    ax.imshow(m)
    t, g, p = px(truth), px(gps), px(det.position)
    ax.plot(*t.T, color="white", lw=2.5, label="true path")
    ax.plot(*g.T, color=ORANGE, lw=2, ls="--", label="spoofed GPS")
    ax.plot(*p[det.alarm:].T, color=GREEN, lw=2.5, label="navigation after alarm (GPS - offset)")
    v = px(visual)
    ax.scatter(*v.T, s=28, color=BLUE, edgecolor="white", lw=0.8, zorder=3, label=f"visual fixes ({method})")
    s = a.start
    ax.scatter(*g[s], s=220, marker="o", facecolor="none", edgecolor=ORANGE, lw=3, zorder=4)
    ax.annotate("spoofing starts", g[s], xytext=(12, -16), textcoords="offset points", color="white",
                fontweight="bold")
    ax.scatter(*g[det.alarm], s=320, marker="X", color="#d62828", edgecolor="white", lw=1.5, zorder=5,
               label="alarm")
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(m.shape[0], 0)
    ax.set_axis_off()
    ax.legend(loc="lower left", framealpha=0.9, fontsize=12)
    ax.set_title("Real UAV flight (UAV-VisLoc #24): GPS dragged away, caught by vision", loc="left")

    k = np.arange(len(truth))
    err_gps = np.linalg.norm(gps - truth, axis=1)
    err_nav = np.linalg.norm(det.position - truth, axis=1)
    bx.plot(k, err_gps, color=ORANGE, lw=2.5, label="trusting GPS")
    bx.plot(k[det.alarm:], err_nav[det.alarm:], color=GREEN, lw=2.5, label="after alarm: GPS - offset from vision")
    has = np.isfinite(det.residual_m)
    bx.scatter(k[has], det.residual_m[has], s=22, color=BLUE, zorder=3, label="|GPS - visual fix|")
    bx.axhline(a.threshold, color=GREY, lw=1.2, ls=":")
    bx.text(2, a.threshold + 4, f"threshold {a.threshold:.0f} m", color=GREY, fontsize=12)
    bx.axvline(s, color=ORANGE, lw=1, alpha=0.5)
    bx.axvline(det.alarm, color="#d62828", lw=1.5)
    bx.text(det.alarm + 3, bx.get_ylim()[1] * 0.72,
            f"alarm after {det.alarm - s + 1} frames\n({min((det.alarm - s + 1) * a.rate, a.max):.0f} m drag)",
            color="#d62828", fontsize=13)
    bx.set_xlabel("frame")
    bx.set_ylabel("position error, m")
    bx.spines[["top", "right"]].set_visible(False)
    bx.grid(axis="y", color="#e5e5e5")
    bx.legend(loc="upper left", frameon=False, fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    print(f"figure: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="CSV with name, lon, lat (the real GPS)")
    ap.add_argument("--geo", required=True)
    ap.add_argument("--map", help="map image (for --figure)")
    ap.add_argument("--visual", nargs="+", required=True, help="per-frame visual fixes: name, x, y, ok")
    ap.add_argument("--start", type=int, default=80, help="frame where spoofing starts (demo figure)")
    ap.add_argument("--rate", type=float, default=3.0, help="drag, metres per frame")
    ap.add_argument("--max", type=float, default=300.0, help="maximum drag, metres")
    ap.add_argument("--bearing", type=float, default=135.0, help="drag direction, degrees from north")
    ap.add_argument("--threshold", type=float, default=15.0, help="alarm threshold on |GPS - visual|, metres")
    ap.add_argument("--confirm", type=int, default=3, help="consecutive disagreeing fixes needed for an alarm")
    ap.add_argument("--sweep", type=int, default=5, help="spoofing start every N frames for the statistics")
    ap.add_argument("--out", default="experiments/spoofing/visloc_24")
    ap.add_argument("--figure", help="PNG of the demo (first --visual method)")
    a = ap.parse_args()

    geo = GeoRef.load(a.geo)
    gt = pd.read_csv(a.gt)
    origin = (float(gt["lon"].iloc[0]), float(gt["lat"].iloc[0]))
    truth = to_enu(gt["lon"], gt["lat"], *origin)
    n = len(truth)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sweep, summary = [], []
    for path in map(Path, a.visual):
        method = path.stem
        visual = load_visual(path, gt["name"], geo, origin)
        honest = detect(truth, visual, a.threshold, a.confirm)
        for s in range(a.sweep, n - 20, a.sweep):
            gps = spoof(truth, s, a.rate, a.max, a.bearing)
            det = detect(gps, visual, a.threshold, a.confirm)
            row = {"method": method, "start": s, "alarm": det.alarm}
            if det.alarm is not None:
                after = slice(det.alarm, None)
                row.update(delay_frames=det.alarm - s + 1, drag_at_alarm_m=min((det.alarm - s + 1) * a.rate, a.max),
                           err_gps_after_m=float(np.median(np.linalg.norm(gps[after] - truth[after], axis=1))),
                           err_nav_after_m=float(np.median(np.linalg.norm(det.position[after] - truth[after],
                                                                          axis=1))))
            sweep.append(row)
        rows = pd.DataFrame([r for r in sweep if r["method"] == method]).reindex(columns=[
            "start", "alarm", "delay_frames", "drag_at_alarm_m", "err_gps_after_m", "err_nav_after_m"])
        caught = rows["alarm"].notna()
        summary.append({
            "method": method, "visual_fixes_%": round(100 * np.isfinite(visual[:, 0]).mean(), 1),
            "attacks": len(rows), "detected_%": round(100 * caught.mean(), 1),
            "false_alarm_honest_gps": honest.alarm is not None,
            "delay_frames_median": float(rows.loc[caught, "delay_frames"].median()),
            "drag_at_alarm_m_median": float(rows.loc[caught, "drag_at_alarm_m"].median()),
            "drag_at_alarm_m_p90": float(rows.loc[caught, "drag_at_alarm_m"].quantile(0.9)),
            "err_after_trusting_gps_m": float(rows.loc[caught, "err_gps_after_m"].median()),
            "err_after_correction_m": float(rows.loc[caught, "err_nav_after_m"].median()),
        })
        if a.figure and path == Path(a.visual[0]):
            gps = spoof(truth, a.start, a.rate, a.max, a.bearing)
            det = detect(gps, visual, a.threshold, a.confirm)
            if det.alarm is None:
                print("no alarm for the demo start, no figure")
            else:
                plot(a.figure, a, geo, gt, truth, gps, visual, det, origin, method.replace("track_", ""))

    pd.DataFrame(sweep).to_csv(out / "sweep.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(out / "summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
