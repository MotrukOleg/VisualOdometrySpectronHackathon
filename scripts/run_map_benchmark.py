"""Map-localisation benchmark: every config x every dataset -> one row in experiments/map_benchmark.csv.

Per run: registration rate, inliers, speed, VRAM; GT accuracy (synthetic); GT-free frame-to-frame
consistency and agreement with the reference run (SIFT whole-map baseline, tag 'reference').

Examples:
    python scripts/run_map_benchmark.py --tag reference --configs configs/map/baseline_sift.yaml
    python scripts/run_map_benchmark.py --tag methods --configs "configs/map/methods/*.yaml"
    python scripts/run_map_benchmark.py --tag est --configs configs/map/baseline_sift.yaml \
        --variants "geometry.estimator=ransac" "geometry.estimator=usac_accurate"
"""
import argparse
import glob
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vo.config import load_config  # noqa: E402
from vo.consistency import agreement, consistency, frame_to_frame  # noqa: E402
from vo.io import FrameSource  # noqa: E402
from vo.localize import run_localization  # noqa: E402
from vo.schema import read_trajectory  # noqa: E402

BENCH = ROOT / "experiments" / "results" / "bench"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["crops", "syn_easy", "syn_hard", "video0061", "video0062"])
    ap.add_argument("--tag", required=True, help="experiment group (e.g. methods, estimators, ablation)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for every run")
    ap.add_argument("--variants", nargs="*", default=[None],
                    help="each variant = one extra override set ('a=1,b=2'); runs config x variant")
    ap.add_argument("--csv", default=str(ROOT / "experiments" / "map_benchmark.csv"))
    ap.add_argument("--no-viz", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with open(ROOT / "configs" / "datasets.yaml", encoding="utf-8") as f:
        datasets = yaml.safe_load(f)
    paths = sorted({p for pat in a.configs for p in (glob.glob(pat) or [pat])})
    for cpath in paths:
        for var in a.variants:
            vset = [s for s in (var or "").split(",") if s]
            for ds in a.datasets:
                d = datasets[ds]
                over = [f"data.frames={d['frames']}", f"data.gt={d['gt']}" if d["gt"] else "data.gt=null"]
                cfg = load_config(cpath, over + a.set + vset)
                label = cfg["name"] + ("" if not vset else "[" + ",".join(vset) + "]")
                out = BENCH / a.tag / label.replace("/", "_").replace("=", "-").replace(",", "+") / ds
                row = {"tag": a.tag, "config": Path(cpath).stem, "variant": ",".join(vset), "label": label,
                       "dataset": ds, "time": time.strftime("%Y-%m-%d %H:%M")}
                t0 = time.time()
                try:
                    if a.no_viz:
                        cfg["viz"]["footprints"] = False
                    m = run_localization(cfg, out)
                    row.update({k: v for k, v in m.items() if not isinstance(v, (list, dict))})
                    traj = read_trajectory(out / "trajectory.csv")
                    src = FrameSource(d["frames"])
                    f2f = frame_to_frame(d["frames"], ROOT / "experiments" / "cache" / f"f2f_{ds}.npz")
                    row.update(consistency(traj, f2f, src.size(0)))
                    ref = BENCH / "reference" / "map_baseline_sift" / ds / "trajectory.csv"
                    if ref.exists() and a.tag != "reference":
                        row.update(agreement(traj, read_trajectory(ref)))
                    row["status"] = "ok"
                except Exception as e:  # keep the benchmark going; the error is recorded
                    row.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
                    logging.error("%s on %s failed:\n%s", label, ds, traceback.format_exc())
                row["wall_s"] = time.time() - t0
                df = pd.DataFrame([row])
                csv = Path(a.csv)
                if csv.exists():
                    df = pd.concat([pd.read_csv(csv), df], ignore_index=True)
                df.to_csv(csv, index=False)
                print(f"{label:55s} {ds:10s} {row['status']:5s} reg={row.get('registered_rate', float('nan')):.3f} "
                      f"err_med={row.get('center_err_median', float('nan')):.2f} "
                      f"cons={row.get('cons_median_px', float('nan')):.3f} dev={row.get('ref_dev_median_px', float('nan')):.2f} "
                      f"fps={row.get('fps', float('nan')):.1f} {row['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
