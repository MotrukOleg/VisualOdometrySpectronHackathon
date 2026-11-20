"""Route visualisation on the map: order of points, START / FINISH, frame footprints, legend."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .io import imwrite
from .schema import corners_of

FONT = cv2.FONT_HERSHEY_SIMPLEX
START_COLOR = (60, 200, 60)  # BGR
FINISH_COLOR = (40, 40, 230)
GT_COLOR = (255, 255, 255)


def _text(img, text, org, scale, color, thick, bg=(0, 0, 0)):
    """Text with a dark outline so it stays readable on any background."""
    org = (int(org[0]), int(org[1]))
    cv2.putText(img, text, org, FONT, scale, bg, thick + max(2, thick), cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _label_box(img, text, anchor, scale, thick, color):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(anchor[0]), int(anchor[1])
    pad = int(6 * scale)
    x = int(np.clip(x, pad, img.shape[1] - tw - 2 * pad))
    y = int(np.clip(y, th + 2 * pad, img.shape[0] - pad))
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad // 2), color, -1)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad // 2), (255, 255, 255), 1)
    cv2.putText(img, text, (x, y), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def time_colors(n: int) -> np.ndarray:
    """(n, 3) BGR colours from early (blue) to late (red), turbo colormap."""
    ramp = np.linspace(30, 230, max(n, 2)).astype(np.uint8)[:, None]
    return cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)[:, 0][:n].astype(int)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def render_route(map_bgr: np.ndarray, traj: pd.DataFrame, out_path: str | Path | None = None,
                 zoom_path: str | Path | None = None, cfg: dict | None = None, title: str = "",
                 gt: np.ndarray | None = None) -> np.ndarray:
    """Draws the trajectory on the map and returns the image (also saved if paths are given).

    - frame footprints (outline of each photo with its true size and rotation), coloured by time;
    - route polyline through frame centres with direction arrows; interpolated parts are dashed-grey;
    - numbered points (order of the route), START / FINISH labels, rejected/lost frames marked;
    - legend with the colour ramp; optional ground truth in white.
    """
    cfg = cfg or {}
    img = (map_bgr.astype(np.float32) * 0.72).astype(np.uint8)  # darken for contrast
    H, W = img.shape[:2]
    u = max(1, int(round(max(H, W) / 1500)))  # line-width unit
    fscale = 0.45 * u
    n = len(traj)
    col = time_colors(n)
    st = traj["status"].to_numpy()
    xy = traj[["x", "y"]].to_numpy(float)
    has = np.isin(st, ["map", "interp"]) & np.all(np.isfinite(xy), axis=1)

    # ground truth (synthetic data)
    if gt is not None:
        g = gt[np.all(np.isfinite(gt), axis=1)]
        if len(g) > 1:
            cv2.polylines(img, [np.round(g).astype(np.int32).reshape(-1, 1, 2)], False, GT_COLOR, u, cv2.LINE_AA)

    # footprints
    fp_every = 1
    if cfg.get("footprints", True):
        # long sequences (video): outline only a subset of frames, otherwise the map is buried
        valid = np.flatnonzero(has)
        fp_every = int(cfg.get("footprint_every") or 0) or max(1, int(np.ceil(len(valid) / 40)))
        fp = valid[::fp_every]
        if len(valid) and fp[-1] != valid[-1]:
            fp = np.append(fp, valid[-1])
        overlay = img.copy()
        cs = corners_of(traj)
        for k in fp[st[fp] == "map"]:
            q = np.round(cs[k]).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [q], tuple(int(c) for c in col[k]))
        img = cv2.addWeighted(overlay, 0.18 if fp_every == 1 else 0.10, img, 0.82 if fp_every == 1 else 0.90, 0)
        for k in fp:
            q = np.round(cs[k]).astype(np.int32).reshape(-1, 1, 2)
            c = tuple(int(v) for v in col[k]) if st[k] == "map" else (150, 150, 150)
            cv2.polylines(img, [q], True, c, u, cv2.LINE_AA)
            # mark the top edge of the photo (its 'up' direction)
            cv2.line(img, tuple(q[0, 0]), tuple(q[1, 0]), c, 2 * u, cv2.LINE_AA)

    # route polyline + arrows
    idx = np.flatnonzero(has)
    p = np.round(xy).astype(int)
    for a, b in zip(idx[:-1], idx[1:]):
        interp = st[a] == "interp" or st[b] == "interp" or b - a > 1
        c = (170, 170, 170) if interp else tuple(int(v) for v in col[a])
        cv2.line(img, tuple(p[a]), tuple(p[b]), (0, 0, 0), 5 * u, cv2.LINE_AA)
        cv2.line(img, tuple(p[a]), tuple(p[b]), c, 3 * u, cv2.LINE_AA)
        seg = np.linalg.norm(p[b] - p[a])
        if seg > 25 * u:
            tip = p[a] + 0.62 * (p[b] - p[a])
            tail = p[a] + 0.38 * (p[b] - p[a])
            cv2.arrowedLine(img, tuple(np.round(tail).astype(int)), tuple(np.round(tip).astype(int)), c, 2 * u,
                            cv2.LINE_AA, tipLength=min(0.9, 12 * u / max(seg * 0.24, 1)))

    # points and order labels
    steps = np.linalg.norm(np.diff(xy[idx], axis=0), axis=1) if len(idx) > 1 else np.array([100.0])
    every = int(cfg.get("label_every") or 0) or (1 if np.median(steps) > 30 * u else 5)
    for k in idx:
        c = tuple(int(v) for v in col[k])
        if st[k] == "map":
            cv2.circle(img, tuple(p[k]), 4 * u, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p[k]), 3 * u, c, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, tuple(p[k]), 4 * u, (170, 170, 170), u, cv2.LINE_AA)
        if k % every == 0 or k in (idx[0], idx[-1]):
            _text(img, str(k), p[k] + np.array([5 * u, -5 * u]), fscale, (255, 255, 255), max(1, u // 2 + 1))
    # rejected registrations (outliers) where they were wrongly placed
    if "rejected" in traj:
        raw = traj[["raw_x", "raw_y"]].to_numpy(float) if "raw_x" in traj else xy
        for k in np.flatnonzero(traj["rejected"].to_numpy(bool)):
            if np.all(np.isfinite(raw[k])):
                cv2.drawMarker(img, tuple(np.round(raw[k]).astype(int)), (0, 0, 255), cv2.MARKER_TILTED_CROSS,
                               10 * u, 2 * u, cv2.LINE_AA)

    # START / FINISH
    if len(idx):
        s, f = idx[0], idx[-1]
        cv2.circle(img, tuple(p[s]), 9 * u, START_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p[s]), 9 * u, (255, 255, 255), u, cv2.LINE_AA)
        cv2.circle(img, tuple(p[f]), 9 * u, FINISH_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p[f]), 9 * u, (255, 255, 255), u, cv2.LINE_AA)
        ts, tf = f"START (frame {s})", f"FINISH (frame {f})"
        if np.linalg.norm(p[s] - p[f]) < 150 * u:  # closed loop: put the labels on opposite sides
            ws = cv2.getTextSize(ts, FONT, 0.6 * u, u)[0][0]
            left = p[s][0] <= p[f][0]
            a_s = p[s] + np.array([-(ws + 20 * u) if left else 16 * u, 6 * u])
            wf = cv2.getTextSize(tf, FONT, 0.6 * u, u)[0][0]
            a_f = p[f] + np.array([16 * u if left else -(wf + 20 * u), 6 * u])
        else:
            a_s, a_f = p[s] + np.array([-40 * u, -14 * u]), p[f] + np.array([-40 * u, 26 * u])
        _label_box(img, ts, a_s, 0.6 * u, u, START_COLOR)
        _label_box(img, tf, a_f, 0.6 * u, u, FINISH_COLOR)

    # the zoomed view is cropped BEFORE the legend is drawn, so it gets its own complete legend
    zoom = None
    if zoom_path and len(idx):
        m = int(cfg.get("zoom_margin", 150))
        pts = np.concatenate([xy[idx], corners_of(traj)[idx].reshape(-1, 2)])
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - m, 0)
        x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int) + m, [W, H])
        zoom = img[y0:y1, x0:x1].copy()
    route_pts = np.concatenate([xy[idx], corners_of(traj)[idx].reshape(-1, 2)]) if len(idx) else np.zeros((0, 2))
    _legend(img, n, u, st, gt is not None, title, route_pts, fp_every)
    if out_path:
        imwrite(out_path, img)
    if zoom is not None:
        _legend(zoom, n, u, st, gt is not None, title, route_pts - [x0, y0], fp_every)
        imwrite(zoom_path, zoom)
    return img


def _legend(img, n, u, st, has_gt, title, route_pts=None, fp_every=1):
    """Legend box, placed in the image corner that covers the fewest route points."""
    H, W = img.shape[:2]
    lines = [
        (f"Route: frames 0 .. {n - 1}  (colour = time)", None),
        (f"registered: {int((st == 'map').sum())}   interpolated: {int((st == 'interp').sum())}   "
         f"lost: {int((st == 'lost').sum())}", None),
        ("outline = frame footprint (thick side = top of photo)" + (f", every {_ordinal(fp_every)} frame" if fp_every > 1 else ""), None),
    ]
    if has_gt:
        lines.append(("white line = ground truth", None))
    fs = 0.5 * u
    th = max(1, u)
    lh = int(26 * u)
    bw = int(max(cv2.getTextSize(t, FONT, fs, th)[0][0] for t, _ in lines + [(title, None)]) + 30 * u)
    bh = int(lh * (len(lines) + 2) + 20 * u)
    g = int(12 * u)
    corners = [(g, H - bh - g), (W - bw - g, H - bh - g), (g, g), (W - bw - g, g)]  # BL, BR, TL, TR
    corners = [(max(0, x), max(0, y)) for x, y in corners]
    if route_pts is not None and len(route_pts):
        p = route_pts[np.all(np.isfinite(route_pts), axis=1)]

        def covered(c):
            x, y = c
            return int(((p[:, 0] >= x) & (p[:, 0] <= x + bw) & (p[:, 1] >= y) & (p[:, 1] <= y + bh)).sum())

        x0, y0 = min(corners, key=covered)  # ties keep the bottom-left default
    else:
        x0, y0 = corners[0]
    roi = img[y0:y0 + bh, x0:x0 + bw]
    roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), 1)
    y = y0 + lh
    if title:
        _text(img, title, (x0 + 12 * u, y), fs * 1.1, (255, 255, 255), th)
        y += lh
    # colour ramp
    ramp_w = bw - int(24 * u)
    cols = time_colors(ramp_w)
    for i in range(ramp_w):
        cv2.line(img, (x0 + 12 * u + i, y - int(14 * u)), (x0 + 12 * u + i, y - int(4 * u)),
                 tuple(int(c) for c in cols[i]), 1)
    _text(img, "start", (x0 + 12 * u, y + int(14 * u)), fs * 0.8, (255, 255, 255), th)
    _text(img, "finish", (x0 + bw - int(70 * u), y + int(14 * u)), fs * 0.8, (255, 255, 255), th)
    y += lh
    for t, _ in lines:
        _text(img, t, (x0 + 12 * u, y + int(6 * u)), fs, (235, 235, 235), th)
        y += lh
