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
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vo.config import load_config
from vo.io import FrameSource, imread, imwrite, resize_long

from .catalog import ROOT

log = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "experiments" / "results" / "app"
UPLOADS_DIR = ROOT / "data" / "app_uploads"
RESULT_FILES = ("trajectory.csv", "route.png", "route_zoom.png", "metrics.json", "config.yaml")
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
            if job.video_path is not None:
                job.push("stage", {"text": "Витягую кадри з відео…"})
                frames = UPLOADS_DIR / job.id / "frames"
                n = extract_video(job.video_path, frames, job.video_every, job.max_frames,
                                  lambda k, total, w: job.push("stage", {"text": f"Витягую кадри з відео… {w} кадрів"
                                                                                + (f" ({100 * k // total}%)" if total else "")}))
                job.frames_dir = frames
                job.push("stage", {"text": f"Витягнуто {n} кадрів"})
            job.source = FrameSource(job.frames_dir, 0, job.step, job.max_frames)
            overrides = job.overrides + [f"data.frames={job.frames_dir.as_posix()}", f"data.step={job.step}",
                                         f"data.max_frames={job.max_frames or 'null'}"]
            cfg = load_config(ROOT / job.config, overrides)
            job.push("stage", {"text": "Рахую ознаки мапи…"})
            run_localization(cfg, job.out_dir, progress=job.push)
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

