"""Run one experiment from a YAML config.

Example:
    python scripts/run_experiment.py --config configs/baseline.yaml --set data.max_frames=50 mosaic.blend=multiband
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.benchmark import append_summary  # noqa: E402
from vo.config import load_config  # noqa: E402
from vo.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--summary", default="experiments/summary.csv")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.set)
    metrics = run(cfg)
    append_summary(metrics, a.summary)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
