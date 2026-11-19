"""Drawing the drone trajectory on the mosaic and plotting it against ground truth."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .geometry import apply  # noqa: E402


def draw_on_mosaic(mosaic: np.ndarray, pts: np.ndarray, lost: np.ndarray | None = None,
                   gt_pts: np.ndarray | None = None) -> np.ndarray:
    """pts: (N, 2) canvas coords of frame centres, coloured by time (turbo colormap)."""
    out = mosaic.copy()
    H, W = out.shape[:2]
    th = max(2, int(round(max(H, W) / 800)))
    if gt_pts is not None:
        g = gt_pts[np.all(np.isfinite(gt_pts), axis=1)]
        cv2.polylines(out, [np.round(g).astype(np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), th + 2,
                      cv2.LINE_AA)
    n = len(pts)
    colors = cv2.applyColorMap(np.linspace(0, 255, max(n, 2)).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    p = np.round(pts).astype(int)
    for k in range(n - 1):
        cv2.line(out, tuple(p[k]), tuple(p[k + 1]), tuple(int(c) for c in colors[k]), th, cv2.LINE_AA)
    if lost is not None:
        for k in np.flatnonzero(lost):
            cv2.drawMarker(out, tuple(p[k]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 6 * th, th)
    if n:
        cv2.circle(out, tuple(p[0]), 4 * th, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, tuple(p[-1]), 4 * th, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def to_canvas(C: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
    return apply(C, world_pts)


def plot_xy(pred: np.ndarray, gt: np.ndarray | None, path: str | Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    if gt is not None:
        ax.plot(gt[:, 0], gt[:, 1], "k-", lw=2, label="ground truth")
    ax.plot(pred[:, 0], pred[:, 1], "-", color="tab:orange", lw=1.5, label="estimated (aligned)" if gt is not None
            else "estimated")
    ax.plot(*pred[0], "go", ms=8, label="start")
    ax.plot(*pred[-1], "ro", ms=8, label="end")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image convention: y down
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
