"""Collecting experiment metrics into one comparison table (experiments/summary.csv)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns shown first in the comparison table (others are kept after them).
COLUMNS = [
    "name", "matcher", "n_frames", "n_lost", "seq_fail_rate", "mean_inliers", "mean_inlier_ratio",
    "mean_reproj_err", "ate_rmse", "ate_pct_of_path", "rpe_rmse_d10", "edge_res_before_loop",
    "edge_res_after_loop", "photometric_err", "t_extract_ms", "t_match_ms", "fps_register", "vram_peak_mb",
    "run_id",
]


def append_summary(metrics: dict, path: str | Path = "experiments/summary.csv") -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([metrics])
    df = pd.concat([pd.read_csv(path), row], ignore_index=True) if path.exists() else row
    cols = [c for c in COLUMNS if c in df.columns] + [c for c in df.columns if c not in COLUMNS]
    df[cols].to_csv(path, index=False)
    return df
