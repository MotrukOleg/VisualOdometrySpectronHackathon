"""Shared file formats (team contract): localisation trajectory and ground truth.

All coordinates are pixels of the global map image. Corners c0..c3 are the frame corners
(0,0), (w,0), (w,h), (0,h) projected onto the map. heading_deg is the direction of the frame's
"up" axis on the map: 0 = map up (north), 90 = map right (east), clockwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATUSES = ("map", "interp", "lost")  # map: registered to the map; interp: filled; lost: no estimate
CORNER_COLS = [f"c{k}{a}" for k in range(4) for a in "xy"]
TRAJ_COLUMNS = ["frame", "name", "x", "y", "heading_deg", "scale", *CORNER_COLS,
                "n_matches", "n_inliers", "reproj_err", "status"]
GT_COLUMNS = ["name", "x", "y"]  # optional: heading_deg, c0x..c3y, source


def corners_of(df: pd.DataFrame) -> np.ndarray:
    """(N, 4, 2) corners from a trajectory/GT table."""
    return df[CORNER_COLS].to_numpy(float).reshape(-1, 4, 2)


def write_trajectory(df: pd.DataFrame, path: str | Path) -> None:
    missing = [c for c in TRAJ_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"trajectory is missing columns {missing}")
    bad = set(df["status"]) - set(STATUSES)
    if bad:
        raise ValueError(f"unknown statuses {bad}")
    cols = TRAJ_COLUMNS + [c for c in df.columns if c not in TRAJ_COLUMNS]
    df[cols].to_csv(path, index=False, float_format="%.3f")


def read_trajectory(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in TRAJ_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df
