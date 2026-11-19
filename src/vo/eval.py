"""Trajectory metrics (ATE / RPE after similarity alignment) and pairwise statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares s, R, t such that dst ~ s * R @ src + t (Umeyama 1991)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if with_scale and var > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def align(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    """Similarity-align pred to gt using frames where gt is finite. Returns (aligned pred, scale)."""
    ok = np.all(np.isfinite(gt), axis=1)
    s, R, t = umeyama(pred[ok], gt[ok])
    return (s * (R @ pred.T)).T + t, s


def path_length(p: np.ndarray) -> float:
    p = p[np.all(np.isfinite(p), axis=1)]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def trajectory_metrics(pred: np.ndarray, gt: np.ndarray, rpe_delta: int = 10) -> tuple[dict, np.ndarray]:
    ok = np.all(np.isfinite(gt), axis=1)
    if ok.sum() < 3:
        return {}, pred
    aligned, s = align(pred, gt)
    e = np.linalg.norm(aligned[ok] - gt[ok], axis=1)
    L = path_length(gt)
    out = {
        "ate_rmse": float(np.sqrt((e**2).mean())),
        "ate_mean": float(e.mean()),
        "ate_median": float(np.median(e)),
        "ate_max": float(e.max()),
        "ate_pct_of_path": float(100 * np.sqrt((e**2).mean()) / L) if L > 0 else np.nan,
        "gt_path_length": L,
        "align_scale": s,
    }
    d = rpe_delta
    if len(pred) > d:
        idx = np.arange(len(pred) - d)
        v = ok[idx] & ok[idx + d]
        if v.any():
            r = (aligned[idx + d] - aligned[idx]) - (gt[idx + d] - gt[idx])
            r = np.linalg.norm(r[v], axis=1)
            out[f"rpe_rmse_d{d}"] = float(np.sqrt((r**2).mean()))
    return out, aligned


def pair_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    seq = df[df["kind"] == "seq"]
    return {
        "n_pairs": int(len(df)),
        "seq_fail_rate": float(1 - seq["ok"].mean()) if len(seq) else np.nan,
        "mean_kpts": float(df[["n_kpts0", "n_kpts1"]].mean().mean()),
        "mean_matches": float(df["n_matches"].mean()),
        "mean_inliers": float(df["n_inliers"].mean()),
        "median_inliers": float(df["n_inliers"].median()),
        "mean_inlier_ratio": float(df["inlier_ratio"].mean()),
        "mean_reproj_err": float(df.loc[df["ok"], "reproj_err"].mean()),
        "t_match_ms": float(1000 * df["t_match"].mean()),
        "t_geom_ms": float(1000 * df["t_geom"].mean()),
    }


def localization_metrics(traj: pd.DataFrame, gt: np.ndarray, thresholds=(5, 10, 25)) -> dict:
    """Absolute map-localisation accuracy (no alignment: both are in map pixels).

    Frames without an estimate (status 'lost') count as failures in success rates.
    """
    has_gt = np.all(np.isfinite(gt), axis=1)
    est = traj[["x", "y"]].to_numpy(float)
    has_est = traj["status"].isin(["map", "interp"]).to_numpy() & np.all(np.isfinite(est), axis=1)
    err = np.full(len(traj), np.inf)
    both = has_gt & has_est
    err[both] = np.linalg.norm(est[both] - gt[both], axis=1)
    out: dict = {"n_gt": int(has_gt.sum())}
    if both.any():
        e = err[both]
        out.update(center_err_mean=float(e.mean()), center_err_median=float(np.median(e)),
                   center_err_rmse=float(np.sqrt((e**2).mean())), center_err_max=float(e.max()))
        reg = both & (traj["status"] == "map").to_numpy()
        if reg.any():
            out["center_err_registered_mean"] = float(err[reg].mean())
    for t in thresholds:
        out[f"success@{t}px"] = float((err[has_gt] < t).mean()) if has_gt.any() else float("nan")
    return out
