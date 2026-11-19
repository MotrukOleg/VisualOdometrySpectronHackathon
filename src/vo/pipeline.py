"""End-to-end run: frames -> pairwise registration -> pose graph -> mosaic + trajectory -> metrics."""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from . import graph as G
from .config import resolve_device
from .eval import pair_summary, trajectory_metrics, umeyama
from .geometry import decompose, estimate_transform
from .io import FrameSource, imwrite, load_gt
from .matching import FeatureMatcher, build_pair_matcher
from .mosaic import build_mosaic, overlap_photometric_error
from .trajectory import draw_on_mosaic, plot_xy, to_canvas

log = logging.getLogger(__name__)


def _cuda_peak_reset():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_mb() -> float:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    return 0.0


class Registrar:
    """Matches frame pairs, estimates relative transforms and records per-pair statistics."""

    def __init__(self, source: FrameSource, matcher, geom_cfg: dict, pts_per_edge: int, seed: int):
        self.source = source
        self.matcher = matcher
        self.geom = geom_cfg
        self.k = pts_per_edge
        self.rng = np.random.default_rng(seed)
        self.records: list[dict] = []
        self.tried: set[tuple[int, int]] = set()

    def __call__(self, i: int, j: int, kind: str, min_inliers: int | None = None) -> G.Edge | None:
        self.tried.add((i, j))
        src = self.source
        corr = self.matcher.match(i, src.load(i), j, src.load(j))
        geom = dict(self.geom)
        if min_inliers is not None:
            geom["min_inliers"] = min_inliers
        t0 = time.perf_counter()
        est = estimate_transform(corr.pts0, corr.pts1, geom, src.size(j))
        t_geom = time.perf_counter() - t0
        rec = {
            "i": i, "j": j, "kind": kind,
            "n_kpts0": corr.n_kpts0, "n_kpts1": corr.n_kpts1, "n_matches": len(corr),
            "n_inliers": est.n_inliers, "inlier_ratio": est.n_inliers / max(len(corr), 1),
            "reproj_err": est.reproj_err, "ok": est.ok, "reason": est.reason,
            "t_match": corr.timings.get("match", np.nan), "t_geom": t_geom,
        }
        if est.T is not None:
            rec.update(decompose(est.T))
        self.records.append(rec)
        if not est.ok:
            return None
        inl = np.flatnonzero(est.inliers)
        sel = inl[G.subsample(len(inl), self.k, self.rng)]
        return G.Edge(i, j, est.T, corr.pts0[sel], corr.pts1[sel], est.n_inliers, kind)


def run(cfg: dict) -> dict:
    t_start = time.perf_counter()
    np.random.seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    d, gc = cfg["data"], cfg["graph"]
    run_id = f"{cfg['name']}_{datetime.now():%Y%m%d-%H%M%S}"
    out = Path(cfg["output"]["dir"]) / run_id
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    log.info("run %s on %s -> %s", cfg["name"], device, out)

    source = FrameSource(d["frames"], d["start"], d["step"], d["max_frames"], d["resize_long"], d["scale"])
    n = len(source)
    matcher = build_pair_matcher(cfg, device)
    _cuda_peak_reset()
    reg = Registrar(source, matcher, cfg["geometry"], gc["opt_points_per_edge"], cfg["seed"])

    # 1. sequential chaining with fallbacks
    poses = [np.eye(3)]
    lost = [False]
    edges: list[G.Edge] = []
    last_rel = np.eye(3)
    t_reg = time.perf_counter()
    for j in tqdm(range(1, n), desc="register", unit="frame"):
        e = reg(j - 1, j, "seq")
        if e is None:
            for k in gc["fallback_skips"]:
                if j - k >= 0 and not lost[j - k]:
                    e = reg(j - k, j, "skip")
                    if e is not None:
                        break
        if e is not None:
            edges.append(e)
            Gj = poses[e.i] @ e.T
            last_rel = np.linalg.inv(poses[j - 1]) @ Gj
            lost.append(False)
        else:  # constant-velocity guess, flagged as lost
            Gj = poses[j - 1] @ last_rel
            lost.append(True)
        poses.append(Gj / Gj[2, 2])
        for k in gc["extra_skips"]:
            if j - k >= 0 and (j - k, j) not in reg.tried:
                e2 = reg(j - k, j, "skip")
                if e2 is not None:
                    edges.append(e2)
    sizes = [source.size(k) for k in range(n)]

    # 2. loop closure
    if gc["loop_closure"]:
        cands = G.loop_candidates(poses, sizes, gc["loop_min_gap"], gc["loop_radius"], gc["loop_stride"],
                                  gc["loop_max_candidates"], reg.tried)
        found = 0
        for i, j in tqdm(cands, desc="loops", unit="pair"):
            e = reg(i, j, "loop", min_inliers=gc["loop_min_inliers"])
            if e is not None:
                edges.append(e)
                found += 1
        log.info("loop closure: %d/%d candidates accepted", found, len(cands))
    t_reg = time.perf_counter() - t_reg

    # 3. global optimisation
    res_before = G.edge_residuals(poses, edges)
    if gc["optimize"]:
        poses = G.optimize(poses, edges, cfg["geometry"]["model"], gc["opt_loss"], gc["opt_f_scale"])
    res_after = G.edge_residuals(poses, edges)

    # 4. metrics
    df = pd.DataFrame(reg.records)
    df.to_csv(out / "pairs.csv", index=False)
    np.save(out / "poses.npy", np.stack(poses))
    centers = G.frame_centers(poses, sizes)
    metrics = {"name": cfg["name"], "run_id": run_id, "matcher": matcher.name, "device": device,
               "n_frames": n, "n_lost": int(np.sum(lost)), "n_edges": len(edges),
               "n_loop_edges": sum(e.kind == "loop" for e in edges)}
    metrics.update(pair_summary(df))
    if isinstance(matcher, FeatureMatcher) and matcher.extract_times:
        metrics["t_extract_ms"] = 1000 * float(np.mean(matcher.extract_times))
    metrics["t_register_s"] = t_reg
    metrics["fps_register"] = n / t_reg if t_reg > 0 else np.nan
    metrics["vram_peak_mb"] = _cuda_peak_mb()
    metrics.update({f"edge_res_before_{k}": v for k, v in res_before.items()})
    metrics.update({f"edge_res_after_{k}": v for k, v in res_after.items()})
    every = max(1, cfg["eval"]["photometric_every"])
    photo_pairs = [(k, k + 1) for k in range(0, n - 1, every)] + [(e.i, e.j) for e in edges if e.kind == "loop"]
    metrics["photometric_err"] = overlap_photometric_error(source.load, poses, photo_pairs)

    gt = load_gt(d.get("gt"), source)
    aligned = None
    if gt is not None:
        tm, aligned = trajectory_metrics(centers, gt, cfg["eval"]["rpe_delta"])
        metrics.update(tm)

    traj = pd.DataFrame({"frame": source.indices, "name": [source.name(k) for k in range(n)],
                         "x": centers[:, 0], "y": centers[:, 1], "lost": lost})
    if gt is not None:
        traj["gt_x"], traj["gt_y"] = gt[:, 0], gt[:, 1]
        traj["aligned_x"], traj["aligned_y"] = aligned[:, 0], aligned[:, 1]
    traj.to_csv(out / "trajectory.csv", index=False)
    plot_xy(aligned if gt is not None else centers, gt, out / "trajectory.png",
            f"{cfg['name']}  ATE={metrics.get('ate_rmse', float('nan')):.2f}")

    # 5. mosaic + trajectory overlay
    mc = cfg["mosaic"]
    if mc["enabled"]:
        t0 = time.perf_counter()
        mosaic, C = build_mosaic(source.load, poses, sizes, mc["scale"], mc["max_canvas_px"], mc["blend"],
                                 mc["every"])
        gt_canvas = None
        if gt is not None:
            ok = np.all(np.isfinite(gt), axis=1)
            s, R, t = umeyama(centers[ok], gt[ok])
            gt_world = (R.T @ (gt - t).T).T / s
            gt_canvas = to_canvas(C, gt_world)
        overlay = draw_on_mosaic(mosaic, to_canvas(C, centers), np.array(lost), gt_canvas)
        if cfg["output"]["save_mosaic"]:
            imwrite(out / "mosaic.jpg", mosaic, [1, 92])  # cv2.IMWRITE_JPEG_QUALITY = 1
        imwrite(out / "mosaic_trajectory.jpg", overlay, [1, 92])
        metrics["t_mosaic_s"] = time.perf_counter() - t0
        metrics["mosaic_size"] = f"{mosaic.shape[1]}x{mosaic.shape[0]}"

    metrics["t_total_s"] = time.perf_counter() - t_start
    metrics["out_dir"] = str(out)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)
    log.info("done: %s", {k: metrics[k] for k in ("n_lost", "seq_fail_rate", "ate_rmse") if k in metrics})
    return metrics
