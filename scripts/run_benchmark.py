"""Run several configs on the same data and collect a comparison table.

Example:
    python scripts/run_benchmark.py --configs configs/baseline.yaml configs/orb_bf.yaml --set data.max_frames=100
    python scripts/run_benchmark.py --configs "configs/*.yaml"
"""
import argparse
import glob
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.benchmark import COLUMNS, append_summary  # noqa: E402
from vo.config import load_config  # noqa: E402
from vo.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True, help="paths or glob patterns")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides applied to every config")
    ap.add_argument("--summary", default="experiments/summary.csv")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths = sorted({p for pat in a.configs for p in (glob.glob(pat) or [pat])})
    rows = []
    for p in paths:
        try:
            m = run(load_config(p, a.set))
            append_summary(m, a.summary)
            rows.append(m)
        except Exception:  # keep going: one broken method must not stop the benchmark
            logging.error("config %s failed:\n%s", p, traceback.format_exc())
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        cols = [c for c in COLUMNS if c in df.columns]
        print(df[cols].sort_values("ate_rmse" if "ate_rmse" in df else "seq_fail_rate").to_string(index=False))


if __name__ == "__main__":
    main()
