"""Post-processing of a localisation trajectory: outlier rejection, interpolation, smoothing."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .schema import CORNER_COLS

log = logging.getLogger(__name__)


def reject_outliers(xy: np.ndarray, valid: np.ndarray, factor: float = 4.0, min_px: float = 50.0) -> np.ndarray:
    """Mark fixes that jump away from both neighbours (isolated wrong registrations).

    A valid fix k is rejected when its distance to the midpoint of its valid neighbours is larger than
    max(min_px, factor * median step). Returns the new validity mask.
    """
    valid = valid.copy()
    idx = np.flatnonzero(valid)
    if len(idx) < 3:
        return valid
    steps = np.linalg.norm(np.diff(xy[idx], axis=0), axis=1)
    thr = max(min_px, factor * float(np.median(steps)))
    for _ in range(3):  # a few passes: removing one outlier can expose another
        idx = np.flatnonzero(valid)
        if len(idx) < 3:
            break
        removed = []
        for a, k, b in zip(idx[:-2], idx[1:-1], idx[2:]):
            # expected position: linear interpolation between neighbours (handles unequal gaps)
            t = (k - a) / (b - a)
            expected = xy[a] + t * (xy[b] - xy[a])
            if np.linalg.norm(xy[k] - expected) > thr and np.linalg.norm(xy[a] - xy[b]) < 2 * thr * (b - a):
                removed.append(k)
        if not removed:
            break
        valid[removed] = False
        log.info("rejected %d outlier fixes (thr %.1f px): %s", len(removed), thr, removed)
    return valid


def _interp(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    k = np.arange(len(values))
    out = values.astype(float).copy()
    if valid.sum() >= 2:
        out[~valid] = np.interp(k[~valid], k[valid], values[valid])
    return out


def postprocess(traj: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Applies outlier rejection, interpolation and optional Savitzky-Golay smoothing.

    Input rows have status 'map' (registered) or 'lost'. Raw estimates are kept in raw_x / raw_y.
    """
    df = traj.copy()
    df["raw_x"], df["raw_y"] = df["x"], df["y"]
    xy = df[["x", "y"]].to_numpy(float)
    valid = (df["status"] == "map").to_numpy() & np.all(np.isfinite(xy), axis=1)
    keep = reject_outliers(xy, valid, cfg["outlier_factor"], cfg["outlier_min_px"])
    df.loc[valid & ~keep, "status"] = "lost"
    df["rejected"] = valid & ~keep

    if cfg["interpolate"] and keep.sum() >= 2:
        inside = (np.arange(len(df)) > np.flatnonzero(keep)[0]) & (np.arange(len(df)) < np.flatnonzero(keep)[-1])
        fill = ~keep & inside  # no extrapolation before the first / after the last fix
        heading = np.degrees(np.unwrap(np.radians(df["heading_deg"].to_numpy(float))))
        for col in ["x", "y", "scale", *CORNER_COLS]:
            df[col] = np.where(fill, _interp(df[col].to_numpy(float), keep), df[col])
        df["heading_deg"] = np.where(fill, _interp(heading, keep) % 360, df["heading_deg"])
        df.loc[fill, "status"] = "interp"

    w = int(cfg.get("savgol_window") or 0)
    ok = df["status"].isin(["map", "interp"]).to_numpy()
    if w >= 3 and ok.sum() > w:
        w += (w + 1) % 2  # must be odd
        span = np.flatnonzero(ok)
        sl = slice(span[0], span[-1] + 1)  # contiguous after interpolation
        for col in ("x", "y"):
            vals = df[col].to_numpy(float).copy()  # pandas >= 3 (copy-on-write) returns read-only views
            smoothed = savgol_filter(vals[sl], w, cfg["savgol_order"])
            shift = smoothed - vals[sl]
            vals[sl] = smoothed
            df[col] = vals
            # move the footprint together with its centre
            for c in CORNER_COLS:
                if c.endswith(col[-1]):
                    cv = df[c].to_numpy(float).copy()
                    cv[sl] += shift
                    df[c] = cv
    return df
