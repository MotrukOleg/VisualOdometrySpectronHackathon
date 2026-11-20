"""FastAPI backend of the Mission Replay demo.

    python -m app                     # http://127.0.0.1:8080
    python -m app --host 0.0.0.0      # reachable from other devices in the LAN

Endpoints (all JSON unless noted):
    GET  /api/info                    methods, presets, whether the task map exists
    GET  /api/jobs                    recent jobs
    POST /api/jobs                    multipart form -> {"id"}; start a localisation job
    GET  /api/jobs/{id}/events?start=N Server-Sent Events: the job log from offset N, then live
    GET  /api/jobs/{id}/map           map preview (JPEG)
    GET  /api/jobs/{id}/frame/{k}     frame k preview (JPEG)
    GET  /api/jobs/{id}/files/{name}  trajectory.csv | route.png | route_zoom.png | metrics.json | config.yaml
    POST /api/jobs/{id}/cancel
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from vo.io import IMG_EXTS

from .catalog import DEFAULT_MAP, METHODS, METHODS_BY_ID, presets
from .jobs import RESULT_FILES, RESULTS_DIR, UPLOADS_DIR, Job, JobManager

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Drone Mission Replay")
manager = JobManager()


def _safe_name(name: str) -> str:
    stem = Path(name or "file").name
    return re.sub(r"[^\w.\-]+", "_", stem)[:120] or "file"


async def _save(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        while chunk := await upload.read(1 << 20):
            f.write(chunk)
    return dest


def _job(job_id: str) -> Job:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/info")
def info():
    return {"methods": [m.to_dict() for m in METHODS],
            "presets": [{"id": p["id"], "label": p["label"], "gt": bool(p["gt"])} for p in presets()],
            "default_map": DEFAULT_MAP.exists()}


@app.get("/api/jobs")
def list_jobs():
    return manager.list()


@app.post("/api/jobs")
async def create_job(
    method: str = Form("baseline"),
    preset: str | None = Form(None),
    every: int = Form(5),
    max_frames: int | None = Form(None),
    frame_scale: float | None = Form(None),
    map_file: UploadFile | None = File(None),
    frames: list[UploadFile] | None = File(None),
):
    m = METHODS_BY_ID.get(method)
    if m is None:
        raise HTTPException(400, f"unknown method {method!r}")
    if not m.available():
        raise HTTPException(400, f"метод «{m.label}» потребує пакетів: {', '.join(m.needs)}")
    job_id = manager.new_id()
    up = UPLOADS_DIR / job_id
    overrides = [f"name=app_{m.id}"]
    video, frames_dir, step = None, None, 1
    max_frames = max_frames or None

    if preset:
        p = next((p for p in presets() if p["id"] == preset), None)
        if p is None:
            raise HTTPException(400, f"набір {preset!r} не знайдено локально")
        map_path, frames_dir, step = Path(p["map"]), Path(p["frames"]), int(p.get("step", 1))
        if p["gt"]:
            overrides.append(f"data.gt={Path(p['gt']).as_posix()}")
        fs = frame_scale or p["frame_scale"]
        title = p["label"]
    else:
        files = [f for f in (frames or []) if f.filename]
        if not files:
            raise HTTPException(400, "завантажте фото кадрів або одне відео")
        if map_file is not None and map_file.filename:
            if Path(map_file.filename).suffix.lower() not in IMG_EXTS:
                raise HTTPException(400, "мапа має бути зображенням")
            map_path = await _save(map_file, up / ("map" + Path(map_file.filename).suffix.lower()))
        elif DEFAULT_MAP.exists():
            map_path = DEFAULT_MAP
        else:
            raise HTTPException(400, "завантажте мапу (мапи завдання локально немає)")
        exts = {Path(f.filename).suffix.lower() for f in files}
        if exts & VIDEO_EXTS:
            if len(files) != 1:
                raise HTTPException(400, "відео завантажується одним файлом, без фото")
            video = await _save(files[0], up / ("video" + Path(files[0].filename).suffix.lower()))
            title = f"Відео {files[0].filename}"
        else:
            bad = exts - IMG_EXTS
            if bad:
                raise HTTPException(400, f"непідтримувані файли: {', '.join(sorted(bad))}")
            frames_dir = up / "frames"
            for f in files:
                await _save(f, frames_dir / _safe_name(f.filename))
            title = f"{len(files)} фото"
        fs = frame_scale or 0.5
    overrides += [f"map.path={Path(map_path).as_posix()}", f"localize.frame_scale={fs}"]

    job = Job(id=job_id, title=title, method=m.label, map_path=Path(map_path), frames_dir=frames_dir,
              out_dir=RESULTS_DIR / job_id, overrides=overrides, config=m.config, video_path=video,
              video_every=max(1, every), max_frames=max_frames, step=step)
    manager.submit(job)
    return {"id": job.id}


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str, start: int = 0):
    job = _job(job_id)

    async def stream():
        i = start
        while True:
            batch = job.events_from(i)
            for e in batch:
                yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
            i += len(batch)
            if job.finished and not job.events_from(i):
                break
            if not batch:
                await asyncio.sleep(0.1)
                yield ": ping\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/map")
def job_map(job_id: str):
    return FileResponse(manager.map_preview(_job(job_id)), media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/frame/{k}")
def job_frame(job_id: str, k: int):
    job = _job(job_id)
    if job.source is None or not 0 <= k < len(job.source):
        raise HTTPException(404, "frame not available")
    return Response(manager.frame_preview(job, k), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    job = _job(job_id)
    if name not in RESULT_FILES or not (job.out_dir / name).exists():
        raise HTTPException(404, "file not available")
    return FileResponse(job.out_dir / name, filename=f"{job.id}_{name}")


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    job = _job(job_id)
    job.cancel_requested = True  # checked by the worker at the next frame (or before start if still queued)
    return {"status": job.status}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

