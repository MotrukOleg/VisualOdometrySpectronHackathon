"""Localisation jobs for the demo: one background worker, per-job event log streamed to the browser.

A job = map + frames (a directory, uploaded photos or frames extracted from an uploaded video) + method.
It runs `vo.localize.run_localization` with a progress callback; every callback becomes a JSON event
appended to `job.events`. Clients read the log from any offset, so a finished job can be replayed later.
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vo.config import load_config
from vo.io import FrameSource, imread, imwrite, resize_long

from .catalog import ROOT

log = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "experiments" / "results" / "app"
UPLOADS_DIR = ROOT / "data" / "app_uploads"
RESULT_FILES = ("trajectory.csv", "route.png", "route_zoom.png", "metrics.json", "config.yaml",
                "trajectory_gps.csv", "route.geojson", "route_map.html")
MAP_PREVIEW_LONG = 2400  # px, map image sent to the browser
FRAME_PREVIEW_LONG = 720


class Cancelled(Exception):
    pass


def clean(obj):
    """JSON-safe copy: NaN/inf -> None, numpy scalars/arrays -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


@dataclass
class Job:
    id: str
    title: str
    method: str
    map_path: Path
    frames_dir: Path | None  # None until the video is extracted
    out_dir: Path
    overrides: list[str]
    config: str
    video_path: Path | None = None
    video_every: int = 5
    max_frames: int | None = None
    step: int = 1
    gt_path: Path | None = None
    area: dict | None = None  # {lat, lon, w_km, h_km, frames}: the map is downloaded before localisation
    gps: dict | None = None  # {"csv": path, "zoom": z}: own flight with GPS, map from tiles around the GPS track
    data_kind: str = "real_nogt"  # synthetic | real (real frames + GPS / GT) | real_nogt (real frames, no truth)
    status: str = "queued"  # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    events: list = field(default_factory=list)
    error: str = ""
    cancel_requested: bool = False
    source: FrameSource | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, kind: str, data: dict | None = None) -> None:
        if self.cancel_requested:
            raise Cancelled()
        with self._lock:
            self.events.append({"i": len(self.events), "type": kind, "data": clean(data or {})})

    def events_from(self, i: int) -> list:
        with self._lock:
            return self.events[i:]

    @property
    def finished(self) -> bool:
        return self.status in ("done", "error", "cancelled")

    def summary(self) -> dict:
        n_frames = next((e["data"].get("n_frames") for e in self.events if e["type"] == "start"), None)
        n_done = sum(e["type"] == "frame" for e in self.events)
        return {"id": self.id, "title": self.title, "method": self.method, "status": self.status,
                "created": self.created, "n_frames": n_frames, "n_done": n_done, "error": self.error}


def extract_video(video: Path, out: Path, every: int, max_frames: int | None, on_progress=None) -> int:
    """Every `every`-th frame of the video -> out/frame_000000.jpg ... Returns the number of frames written."""
    cap = cv2.VideoCapture(str(video))
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        cap.release()
        # e.g. HEVC .MOV from an iPhone: OpenCV's FFmpeg build often cannot decode it -> re-encode to H.264
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("OpenCV не може прочитати це відео (ймовірно HEVC/.MOV), а ffmpeg не знайдено в PATH. "
                               "Перекодуйте у H.264 MP4 або витягніть кадри: scripts/extract_frames.py")
        h264 = video.with_name(video.stem + "_h264.mp4")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "18", "-an", str(h264)], check=True)
        cap = cv2.VideoCapture(str(h264))
        if not cap.isOpened():
            raise RuntimeError("Не вдалося відкрити відео навіть після перекодування")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out.mkdir(parents=True, exist_ok=True)
    k = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % every == 0:
            imwrite(out / f"frame_{k:06d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
            if on_progress and written % 10 == 0:
                on_progress(k, total, written)
            if max_frames and written >= max_frames:
                break
        k += 1
    cap.release()
    if not written:
        raise RuntimeError("У відео не знайдено жодного кадру")
    return written


def build_area(job: Job) -> None:
    """Map of any place from satellite tiles + a synthetic flight with exact GT (scripts/make_area.py)."""
    a = job.area
    out = UPLOADS_DIR / job.id / "area"
    job.push("stage", {"text": f"Завантажую супутникову мапу {a['lat']:.4f}, {a['lon']:.4f}…"})
    cmd = [sys.executable, str(ROOT / "scripts" / "make_area.py"), "--name", f"app_{job.id}",
           "--center", f"{a['lat']},{a['lon']}", "--size-km", f"{a['w_km']}x{a['h_km']}",
           "--frames", str(a["frames"]), "--out", str(out), "--cache", str(ROOT / "data" / "cache" / "tiles")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not (out / "map.jpg").exists():
        raise RuntimeError("Не вдалося побудувати мапу з тайлів (немає інтернету або знімків для цих координат):\n"
                           + (r.stderr or r.stdout)[-600:])
    job.map_path, job.frames_dir, job.gt_path = out / "map.jpg", out / "flight" / "frames", out / "flight" / "gt.csv"
    job.overrides += [f"map.path={job.map_path.as_posix()}", f"data.gt={job.gt_path.as_posix()}"]
    (job.out_dir / "map_preview.jpg").unlink(missing_ok=True)
    job.push("stage", {"text": "Мапу завантажено"})


GPS_NAME_COLS, GPS_LAT_COLS, GPS_LON_COLS = ("name", "filename", "file", "image", "frame"), ("lat", "latitude"), \
    ("lon", "lng", "long", "longitude")


def read_gps_csv(path: Path) -> pd.DataFrame:
    """name, lat, lon from a GPS CSV with any common column names (name = frame file stem)."""
    d = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    cols = {c.lower().strip(): c for c in d.columns}
    pick = lambda opts: next((cols[o] for o in opts if o in cols), None)  # noqa: E731
    n, la, lo = pick(GPS_NAME_COLS), pick(GPS_LAT_COLS), pick(GPS_LON_COLS)
    if not (n and la and lo):
        raise ValueError(f"CSV з GPS має містити колонки назви кадру ({'/'.join(GPS_NAME_COLS)}), "
                         f"широти ({'/'.join(GPS_LAT_COLS)}) і довготи ({'/'.join(GPS_LON_COLS)})")
    out = pd.DataFrame({"name": d[n].astype(str).map(lambda s: Path(s).stem),
                        "lat": pd.to_numeric(d[la], errors="coerce"), "lon": pd.to_numeric(d[lo], errors="coerce")})
    out = out.dropna()
    if len(out) < 2:
        raise ValueError("у CSV з GPS менше двох валідних рядків")
    return out


def build_gps_map(job: Job) -> None:
    """Own flight with GPS: the satellite map around the GPS track (margin 300 m) + GT in map px."""
    from vo.tiles import fetch_box, lonlat_to_px, meters_per_px

    gps = read_gps_csv(job.gps["csv"])
    names = {p.stem for p in job.frames_dir.iterdir()} if job.frames_dir else set()
    if names and not set(gps["name"]) & names:
        raise RuntimeError("назви кадрів у CSV з GPS не збігаються з назвами файлів кадрів")
    z = job.gps["zoom"]
    x, y = lonlat_to_px(gps["lon"].to_numpy(), gps["lat"].to_numpy(), z)
    pad = 300 / meters_per_px(float(gps["lat"].mean()), z)
    x0, y0 = int(x.min() - pad), int(y.min() - pad)
    w, h = int(x.max() + pad) - x0, int(y.max() + pad) - y0
    if w * h > 12000 * 12000:
        raise RuntimeError("політ завеликий для мапи з тайлів на цьому zoom — оберіть менший zoom")
    job.push("stage", {"text": f"Завантажую супутникову мапу навколо GPS-треку ({w}x{h} px, zoom {z})…"})
    img, geo = fetch_box(x0, y0, w, h, z, "esri", ROOT / "data" / "cache" / "tiles")
    out = UPLOADS_DIR / job.id / "gpsmap"
    out.mkdir(parents=True, exist_ok=True)
    imwrite(out / "map.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    geo.save(out / "geo.json")
    gps["x"], gps["y"] = x - x0, y - y0
    gps[["name", "x", "y", "lon", "lat"]].to_csv(out / "gt.csv", index=False, float_format="%.7f")
    job.map_path, job.gt_path = out / "map.jpg", out / "gt.csv"
    job.overrides += [f"map.path={job.map_path.as_posix()}", f"data.gt={job.gt_path.as_posix()}"]
    (job.out_dir / "map_preview.jpg").unlink(missing_ok=True)
    job.push("stage", {"text": "Мапу з тайлів завантажено"})


def export_gps(job: Job) -> None:
    """lat/lon trajectory, GeoJSON and the interactive web map, if the map is georeferenced (geo.json next to it)."""
    import json

    from vo.gps import add_lonlat, gps_errors, gps_metrics, to_geojson, write_web_map
    from vo.schema import read_trajectory
    from vo.tiles import GeoRef

    geo_path = job.map_path.parent / "geo.json"
    if not geo_path.exists():
        return
    geo = GeoRef.load(geo_path)
    traj = add_lonlat(read_trajectory(job.out_dir / "trajectory.csv"), geo)
    gt = pd.read_csv(job.gt_path) if job.gt_path and job.gt_path.exists() else None
    if gt is not None and not {"lon", "lat"} <= set(gt.columns):
        gt = None
    summary = {"frames": len(traj), "registered": int((traj["status"] == "map").sum()),
               "lost": int((traj["status"] == "lost").sum()), "map resolution": f"{geo.meters_per_px():.2f} m/px"}
    if gt is not None:
        traj["err_m"] = gps_errors(traj, gt)
        m = gps_metrics(traj["err_m"])
        summary["median error"] = f"{m['gps_err_median_m']:.2f} m"
    traj.to_csv(job.out_dir / "trajectory_gps.csv", index=False, float_format="%.7f")
    gj = to_geojson(traj, gt)
    (job.out_dir / "route.geojson").write_text(json.dumps(gj, separators=(",", ":")), encoding="utf-8")
    write_web_map(gj, job.out_dir / "route_map.html", f"UAV route: {job.title}", summary)
    lon, lat = traj.loc[traj["status"] == "map", ["lon", "lat"]].median()
    job.push("gps", {"files": ["trajectory_gps.csv", "route.geojson", "route_map.html"],
                     "center": [float(lat), float(lon)] if np.isfinite(lat) else None})


class JobManager:
    """In-memory registry; jobs run one at a time (a single GPU with 4 GB is the bottleneck)."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="localize")

    def new_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def submit(self, job: Job) -> Job:
        job.out_dir.mkdir(parents=True, exist_ok=True)
        self.jobs[job.id] = job
        job.push("queued", {"title": job.title, "method": job.method})
        self.pool.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.summary() for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def _run(self, job: Job) -> None:
        from vo.localize import run_localization

        job.status = "running"
        try:
            if job.cancel_requested:
                raise Cancelled()
            if job.area is not None:
                build_area(job)
            if job.gps is not None:
                build_gps_map(job)
            if job.video_path is not None:
                job.push("stage", {"text": "Витягую кадри з відео…"})
                frames = UPLOADS_DIR / job.id / "frames"
                n = extract_video(job.video_path, frames, job.video_every, job.max_frames,
                                  lambda k, total, w: job.push("stage", {"text": f"Витягую кадри з відео… {w} кадрів"
                                                                                + (f" ({100 * k // total}%)" if total else "")}))
                job.frames_dir = frames
                job.push("stage", {"text": f"Витягнуто {n} кадрів"})
            job.push("meta", {"kind": job.data_kind, "geo": (job.map_path.parent / "geo.json").exists()})
            job.source = FrameSource(job.frames_dir, 0, job.step, job.max_frames)
            overrides = job.overrides + [f"data.frames={job.frames_dir.as_posix()}", f"data.step={job.step}",
                                         f"data.max_frames={job.max_frames or 'null'}"]
            cfg = load_config(ROOT / job.config, overrides)
            job.push("stage", {"text": "Рахую ознаки мапи…"})
            run_localization(cfg, job.out_dir, progress=job.push)
            export_gps(job)
            job.status = "done"
            job.events.append({"i": len(job.events), "type": "finished", "data": {"status": "done"}})
        except Cancelled:
            job.status = "cancelled"
            job.events.append({"i": len(job.events), "type": "finished", "data": {"status": "cancelled"}})
        except Exception as e:  # noqa: BLE001 - reported to the UI
            log.exception("job %s failed", job.id)
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.events.append({"i": len(job.events), "type": "finished",
                               "data": {"status": "error", "error": job.error, "trace": traceback.format_exc()[-2000:]}})

    # ---- media for the browser -------------------------------------------------------------------------------------
    def map_preview(self, job: Job) -> Path:
        out = job.out_dir / "map_preview.jpg"
        if not out.exists():
            img, _ = resize_long(imread(job.map_path), MAP_PREVIEW_LONG)
            imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return out

    def frame_preview(self, job: Job, k: int) -> bytes:
        if job.source is None:
            raise KeyError("frames are not ready yet")
        img, _ = resize_long(imread(job.source.files[k]), FRAME_PREVIEW_LONG)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

