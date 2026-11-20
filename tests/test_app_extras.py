"""Demo tabs besides the missions: live camera, coordinates validation, spoofing / stress when their data is local."""
import time

import cv2
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from vo.io import imread, imwrite  # noqa: E402
from vo.synthetic import make_flight, make_world  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app import extras, jobs, server

    monkeypatch.setattr(jobs, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(jobs, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(extras, "UPLOADS", tmp_path / "uploads" / "stress")
    monkeypatch.setattr(server, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(extras, "CACHE", tmp_path / "cache")
    manager = jobs.JobManager()
    monkeypatch.setattr(server, "manager", manager)
    extras.bind_manager(manager)
    return TestClient(server.app)


def test_live_camera_tracks_frames(client, tmp_path, monkeypatch):
    from app import extras

    imwrite(tmp_path / "map.png", make_world(1600, 1200, seed=3))
    data = make_flight(tmp_path / "syn", n_frames=6, frame_size=(400, 300), turns=0.3, seed=3,
                       world_path=tmp_path / "map.png", frame_scale=0.5)
    monkeypatch.setattr(extras, "_live_maps", lambda: [{"id": "t", "label": "t", "map": str(tmp_path / "map.png")}])
    s = client.post("/api/live/start", data={"map_id": "t", "method": "track_sift"}).json()
    assert s["map_size"] == [1600, 1200]
    assert client.get("/api/live/preview/t").status_code == 200
    fixes = []
    for p in sorted((data / "frames").iterdir()):
        ok, buf = cv2.imencode(".jpg", imread(p))
        r = client.post(f"/api/live/{s['id']}/frame", content=buf.tobytes(), headers={"Content-Type": "image/jpeg"})
        fixes.append(r.json())
    assert sum(f["ok"] for f in fixes) >= 4  # the procedural world is harder than real maps for the track ROI
    assert any(f["source"] in ("roi", "roi_retry") for f in fixes)  # frames after the first are tracked
    assert client.post(f"/api/live/{s['id']}/frame", content=b"junk").status_code == 400
    assert client.post("/api/live/nope/frame", content=buf.tobytes()).status_code == 404


def test_coordinates_are_validated(client):
    assert client.post("/api/jobs", data={"lat": "95", "lon": "24"}).status_code == 400
    assert client.post("/api/jobs", data={"lat": "49.8", "lon": "24", "size_km": "20x1"}).status_code == 400
    assert client.post("/api/jobs", data={"lat": "49.8", "lon": "24", "size_km": "abc"}).status_code == 400


def test_spoofing_on_own_flight_with_gps(client, tmp_path, monkeypatch):
    """A mission on a georeferenced map with a GPS ground truth becomes a spoofing source."""
    import pandas as pd

    from app import jobs, server
    from vo.tiles import GeoRef

    imwrite(tmp_path / "map.png", make_world(1600, 1200, seed=4))
    data = make_flight(tmp_path / "syn", n_frames=16, frame_size=(400, 300), turns=0.4, seed=4,
                       world_path=tmp_path / "map.png", frame_scale=0.5)
    geo = GeoRef(17, 74_000 * 256 // 256 * 256, 44_000 * 256 // 256 * 256, 1600, 1200)
    geo.save(tmp_path / "geo.json")
    gt = pd.read_csv(data / "gt.csv")
    gt["name"] = [p.stem for p in sorted((data / "frames").iterdir())][: len(gt)]
    gt["lon"], gt["lat"] = geo.to_lonlat(gt["x"], gt["y"])
    gt[["name", "lat", "lon"]].to_csv(tmp_path / "gps.csv", index=False)
    files = [("frames", (p.name, open(p, "rb"), "image/png")) for p in sorted((data / "frames").iterdir())]
    files += [("map_file", ("map.png", open(tmp_path / "map.png", "rb"), "image/png")),
              ("geo_file", ("geo.json", open(tmp_path / "geo.json", "rb"), "application/json")),
              ("gps_file", ("gps.csv", open(tmp_path / "gps.csv", "rb"), "text/csv"))]
    job_id = client.post("/api/jobs", data={"method": "baseline", "frame_scale": "0.5"}, files=files).json()["id"]
    job = server.manager.get(job_id)
    for _ in range(600):
        if job.finished:
            break
        time.sleep(0.1)
    assert job.status == "done", job.error
    assert job.data_kind == "real" and (job.out_dir / "route_map.html").exists()
    src = [s for s in client.get("/api/spoof/sources").json()["sources"] if s["id"] == f"job:{job_id}"]
    assert src and src[0]["fixes_pct"] >= 40  # procedural world: the baseline registers part of the frames
    r = client.get("/api/spoof/run", params={"source": src[0]["id"], "start": 4, "rate": 20, "threshold": 15}).json()
    assert r["stats"]["alarm"] is not None and r["stats"]["false_alarm"] is None


def test_gps_csv_is_validated(client, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    files = [("frames", ("f.jpg", cv2.imencode(".jpg", make_world(64, 64, seed=1))[1].tobytes(), "image/jpeg")),
             ("gps_file", ("bad.csv", open(bad, "rb"), "text/csv"))]
    r = client.post("/api/jobs", data={"method": "baseline"}, files=files)
    assert r.status_code == 400 and "колонки" in r.json()["detail"]


def test_stress_own_frame(client, tmp_path):
    world = make_world(1600, 1200, seed=6)
    imwrite(tmp_path / "map.png", world)
    data = make_flight(tmp_path / "syn", n_frames=2, frame_size=(400, 300), seed=6, world_path=tmp_path / "map.png",
                       frame_scale=0.5)
    frame = sorted((data / "frames").iterdir())[0]
    up = client.post("/api/stress/upload", files={"frame": open(frame, "rb"), "map_file": open(tmp_path / "map.png", "rb")},
                     data={"frame_scale": "0.5"}).json()
    r = client.get("/api/stress/try", params={"source": up["id"], "axis": "noise", "level_idx": 0}).json()
    assert r["success"] and r["units"] == "px"
    assert client.get("/api/stress/try", params={"source": up["id"], "axis": "tilt"}).status_code == 400
    cv = client.get("/api/stress/curve", params={"source": up["id"], "axis": "noise", "method": "sift_flann"}).json()
    assert cv["frames"] == 1 and cv["success"][0] == 1.0
