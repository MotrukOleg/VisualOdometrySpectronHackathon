"""Render a video with the UAV's current position on the map (Advanced-level result).

Layout (1920x1080):
    +-------------------+--------------------------------------+
    | camera frame      |                                      |
    +-------------------+   overview: map with the route so    |
    | zoom: map around  |   far, current frame footprint,      |
    | the current frame |   START marker, HUD                  |
    +-------------------+--------------------------------------+

Poses are known for the localised keyframes (e.g. every 5th video frame, from trajectory.csv);
for the other video frames they are linearly interpolated, so the output keeps the source frame rate.

Example:
    python scripts/render_video.py --video data/video/IMG_0061.MOV \
        --run experiments/results/video0061_sift_fs0.5 --out experiments/results/video0061_sift_fs0.5/route_video.mp4
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.config import load_config  # noqa: E402
from vo.io import imread, imwrite  # noqa: E402
from vo.schema import corners_of, read_trajectory  # noqa: E402
from vo.viz import FINISH_COLOR, START_COLOR, _text, time_colors  # noqa: E402

W, H = 1920, 1080
LEFT_W = 768
CAM_H = 432
FONT = cv2.FONT_HERSHEY_SIMPLEX


def keyframe_index(names: list[str]) -> np.ndarray:
    """Video frame number of each keyframe from names like frame_000125."""
    out = []
    for n in names:
        m = re.search(r"(\d+)$", str(n))
        if not m:
            raise ValueError(f"cannot parse a frame number from {n!r}")
        out.append(int(m.group(1)))
    return np.array(out)


def interpolate_poses(key_t: np.ndarray, key_xy: np.ndarray, key_c: np.ndarray, ok: np.ndarray, n_frames: int):
    """Per video frame centre (n,2) and corners (n,4,2); NaN before the first / after the last valid keyframe."""
    t = np.arange(n_frames)
    kt, kxy, kc = key_t[ok], key_xy[ok], key_c[ok].reshape(-1, 8)
    xy = np.full((n_frames, 2), np.nan)
    cs = np.full((n_frames, 8), np.nan)
    inside = (t >= kt[0]) & (t <= kt[-1])
    for d in range(2):
        xy[inside, d] = np.interp(t[inside], kt, kxy[:, d])
    for d in range(8):
        cs[inside, d] = np.interp(t[inside], kt, kc[:, d])
    return xy, cs.reshape(-1, 4, 2)


def fit_view(map_bgr, pts, out_w, out_h, margin=120):
    """Crop of the map around `pts` with the aspect of the output panel. Returns (scaled image, 3x3 map->panel)."""
    lo, hi = np.nanmin(pts, 0) - margin, np.nanmax(pts, 0) + margin
    cx, cy = (lo + hi) / 2
    w, h = hi - lo
    if w / h < out_w / out_h:
        w = h * out_w / out_h
    else:
        h = w * out_h / out_w
    s = out_w / w
    M = np.array([[s, 0, -(cx - w / 2) * s], [0, s, -(cy - h / 2) * s], [0, 0, 1]])
    img = cv2.warpAffine(map_bgr, M[:2], (out_w, out_h), flags=cv2.INTER_AREA, borderValue=(25, 25, 25))
    return img, M


def tr(M, p):
    p = np.asarray(p, float).reshape(-1, 2)
    return (p @ M[:2, :2].T + M[:2, 2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--run", required=True, help="localisation run directory (trajectory.csv, config.yaml)")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--map", help="map image (default: map.path from the run config)")
    ap.add_argument("--gif", help="also write a short GIF preview (for slides)")
    ap.add_argument("--gif-seconds", type=float, default=12.0)
    a = ap.parse_args()

    run = Path(a.run)
    cfg = load_config(run / "config.yaml")
    map_bgr = imread(a.map or cfg["map"]["path"])
    traj = read_trajectory(run / "trajectory.csv")
    key_t = keyframe_index(traj["name"].tolist())
    key_xy = traj[["x", "y"]].to_numpy(float)
    key_c = corners_of(traj)
    ok = traj["status"].isin(["map", "interp"]).to_numpy() & np.all(np.isfinite(key_xy), axis=1)

    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    xy, cs = interpolate_poses(key_t, key_xy, key_c, ok, n)
    col = time_colors(n)

    # static backgrounds
    route_pts = np.concatenate([key_xy[ok], key_c[ok].reshape(-1, 2)])
    over_w, over_h = W - LEFT_W, H
    overview, M_over = fit_view(map_bgr, route_pts, over_w, over_h)
    overview = (overview * 0.75).astype(np.uint8)
    p_over = tr(M_over, np.nan_to_num(xy))
    diam = np.nanmedian(np.linalg.norm(cs[:, 0] - cs[:, 2], axis=1))  # footprint diagonal, map px
    zoom_h = H - CAM_H
    zoom_span = 1.6 * diam  # map px across the zoom panel width

    tmp = Path(a.out).with_suffix(".tmp.mp4")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gif_frames = []
    gif_every = max(1, int(round(fps / 6)))
    # route drawn incrementally in two layers, so that the dark outline never covers the coloured line
    outline = np.zeros((over_h, over_w), np.uint8)
    colour = np.zeros((over_h, over_w, 3), np.uint8)
    colour_mask = np.zeros((over_h, over_w), np.uint8)
    last = None
    for t in tqdm(range(n), desc="render", unit="frame"):
        okf, frame = cap.read()
        if not okf:
            break
        canvas = np.zeros((H, W, 3), np.uint8)
        # camera panel
        fh, fw = frame.shape[:2]
        s = min(LEFT_W / fw, CAM_H / fh)
        cam = cv2.resize(frame, (int(fw * s), int(fh * s)), interpolation=cv2.INTER_AREA)
        ox = (LEFT_W - cam.shape[1]) // 2  # centre portrait videos in the camera panel
        canvas[:cam.shape[0], ox:ox + cam.shape[1]] = cam
        _text(canvas, "camera", (12, 30), 0.8, (255, 255, 255), 2)

        have = np.all(np.isfinite(xy[t]))
        # incremental trail on the overview
        if have:
            p = tuple(np.round(p_over[t]).astype(int))
            if last is not None:
                cv2.line(outline, last, p, 255, 8, cv2.LINE_AA)
                cv2.line(colour, last, p, tuple(int(c) for c in col[t]), 4, cv2.LINE_AA)
                cv2.line(colour_mask, last, p, 255, 4, cv2.LINE_AA)
            last = p
        view = overview.copy()
        a_out = (outline.astype(np.float32) / 255.0)[..., None]
        view = (view * (1 - a_out)).astype(np.uint8)
        m = colour_mask > 127
        view[m] = colour[m]
        first = np.flatnonzero(np.all(np.isfinite(xy), axis=1))
        if len(first):
            ps = tuple(np.round(p_over[first[0]]).astype(int))
            cv2.circle(view, ps, 11, START_COLOR, -1, cv2.LINE_AA)
            cv2.circle(view, ps, 11, (255, 255, 255), 2, cv2.LINE_AA)
            _text(view, "START", (ps[0] + 14, ps[1] + 6), 0.7, START_COLOR, 2)
        if have:
            q = np.round(tr(M_over, cs[t])).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [q], True, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(view, tuple(q[0, 0]), tuple(q[1, 0]), (0, 215, 255), 5, cv2.LINE_AA)  # top edge of the photo
            cv2.circle(view, p, 9, FINISH_COLOR, -1, cv2.LINE_AA)
            cv2.circle(view, p, 9, (255, 255, 255), 2, cv2.LINE_AA)
        # HUD
        k = int(np.searchsorted(key_t, t, side="right")) - 1
        k = max(0, min(k, len(traj) - 1))
        hud = [f"t = {t / fps:5.1f} s   video frame {t}/{n - 1}",
               (f"position (map px): {xy[t, 0]:7.1f}, {xy[t, 1]:7.1f}" if have else "position: not localised"),
               f"last keyframe {traj['name'].iloc[k]}: {traj['status'].iloc[k]}, {int(traj['n_inliers'].iloc[k])} inliers"]
        box_w = 620
        roi = view[16:16 + 26 * len(hud) + 22, 16:16 + box_w]
        roi[:] = (roi * 0.35).astype(np.uint8)
        for i, line in enumerate(hud):
            _text(view, line, (30, 46 + 26 * i), 0.62, (240, 240, 240), 1)
        canvas[:, LEFT_W:] = view

        # zoom panel: north-up crop around the current position
        if have:
            sz = LEFT_W / zoom_span
            cx, cy = xy[t]
            Mz = np.array([[sz, 0, LEFT_W / 2 - cx * sz], [0, sz, zoom_h / 2 - cy * sz]])
            zoom = cv2.warpAffine(map_bgr, Mz, (LEFT_W, zoom_h), flags=cv2.INTER_LINEAR, borderValue=(25, 25, 25))
            q = np.round(cs[t] @ Mz[:, :2].T + Mz[:, 2]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(zoom, [q], True, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(zoom, tuple(q[0, 0]), tuple(q[1, 0]), (0, 215, 255), 4, cv2.LINE_AA)
            cv2.drawMarker(zoom, (LEFT_W // 2, zoom_h // 2), FINISH_COLOR, cv2.MARKER_CROSS, 26, 3, cv2.LINE_AA)
        else:
            zoom = np.full((zoom_h, LEFT_W, 3), 25, np.uint8)
            _text(zoom, "not localised", (LEFT_W // 2 - 110, zoom_h // 2), 0.9, (200, 200, 200), 2)
        canvas[CAM_H:, :LEFT_W] = zoom
        _text(canvas, "map around the drone (north up)", (12, CAM_H + 30), 0.75, (255, 255, 255), 2)
        cv2.line(canvas, (LEFT_W, 0), (LEFT_W, H), (255, 255, 255), 2)
        cv2.line(canvas, (0, CAM_H), (LEFT_W, CAM_H), (255, 255, 255), 2)
        writer.write(canvas)
        if a.gif and t % gif_every == 0 and t / fps <= a.gif_seconds:
            gif_frames.append(cv2.resize(canvas, (720, 405), interpolation=cv2.INTER_AREA))
    writer.release()
    cap.release()
    imwrite(Path(a.out).with_name(Path(a.out).stem + "_last.png"), canvas)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:  # H.264 so that browsers / reveal.js can play it
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-preset", "medium",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], check=True)
        tmp.unlink()
    else:
        tmp.replace(a.out)
        print("ffmpeg not found: wrote MPEG-4 Part 2 (mp4v); browsers may not play it")
    if a.gif and gif_frames and ffmpeg:
        frames_dir = Path(a.out).with_suffix("")
        frames_dir = frames_dir.parent / (frames_dir.name + "_gif_frames")
        frames_dir.mkdir(exist_ok=True)
        for i, f in enumerate(gif_frames):
            imwrite(frames_dir / f"{i:05d}.png", f)
        pattern = str(frames_dir / "%05d.png")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps / gif_every), "-i", pattern,
                        "-vf", "split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        str(a.gif)], check=True)
        shutil.rmtree(frames_dir)
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
