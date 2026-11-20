"""Mission Replay demo (app/): upload a synthetic flight (photos or a video), stream the job, fetch results."""
import json
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")  # python-multipart, needed for form uploads
from fastapi.testclient import TestClient  # noqa: E402

from vo.io import imread, imwrite  # noqa: E402
from vo.synthetic import make_flight, make_world  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app import jobs, server

    monkeypatch.setattr(jobs, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(jobs, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(server, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(server, "manager", jobs.JobManager())
    return TestClient(server.app)


@pytest.fixture()
def flight(tmp_path):
    world = make_world(1600, 1200, seed=5)
    imwrite(tmp_path / "map.png", world)
    data = make_flight(tmp_path / "syn", n_frames=12, frame_size=(400, 300), turns=0.5, tilt_std=2e-5, seed=5,
                       world_path=tmp_path / "map.png", frame_scale=0.5)
    return tmp_path / "map.png", sorted((data / "frames").iterdir())


def _events(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:  # the stream ends when the job is finished
        r = client.get(f"/api/jobs/{job_id}/events")
        events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
        if events and events[-1]["type"] == "finished":
            return events
    raise TimeoutError("job did not finish")


def _check_done(client, job_id, events, n):
    kinds = [e["type"] for e in events]
    assert events[-1]["data"]["status"] == "done", events[-1]
    assert kinds.count("frame") == n and "start" in kinds and "done" in kinds
    assert [e["i"] for e in events] == list(range(len(events)))
    start = next(e["data"] for e in events if e["type"] == "start")
    assert start["n_frames"] == n and start["map_size"] == [1600, 1200]
    frames = [e["data"] for e in events if e["type"] == "frame"]
    assert sum(f["status"] == "map" for f in frames) >= n - 1
    assert all(f["x"] is None or isinstance(f["x"], float) for f in frames)  # NaN sent as null
    for name in ("trajectory.csv", "route.png", "metrics.json"):
        assert client.get(f"/api/jobs/{job_id}/files/{name}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/files/..%2Fsecret").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/frame/0").headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/jobs/{job_id}/map").status_code == 200
    assert client.get("/api/jobs").json()[0]["status"] == "done"


def test_upload_photos(client, flight):
    map_path, frames = flight
    files = [("map_file", ("map.png", map_path.read_bytes(), "image/png"))]
    files += [("frames", (p.name, p.read_bytes(), "image/png")) for p in frames]
    r = client.post("/api/jobs", data={"method": "baseline", "frame_scale": "0.5"}, files=files)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    _check_done(client, job_id, _events(client, job_id), len(frames))


def test_upload_video(client, flight, tmp_path):
    map_path, frames = flight
    video = tmp_path / "flight.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (400, 300))
    for p in frames:
        for _ in range(2):  # every frame twice; the job keeps every 2nd
            w.write(imread(p))
    w.release()
    files = [("map_file", ("map.png", map_path.read_bytes(), "image/png")),
             ("frames", ("flight.avi", video.read_bytes(), "video/x-msvideo"))]
    r = client.post("/api/jobs", data={"method": "track_sift", "every": "2", "frame_scale": "0.5"}, files=files)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    events = _events(client, job_id)
    assert any(e["type"] == "stage" and "Витягнуто" in e["data"]["text"] for e in events)
    _check_done(client, job_id, events, len(frames))


def test_rejects_bad_requests(client):
    assert client.post("/api/jobs", data={"method": "nope"}).status_code == 400
    r = client.post("/api/jobs", data={"method": "baseline"},
                    files=[("frames", ("notes.txt", b"x", "text/plain"))])
    assert r.status_code == 400
    info = client.get("/api/info").json()
    assert {m["id"] for m in info["methods"]} >= {"baseline", "final"}
