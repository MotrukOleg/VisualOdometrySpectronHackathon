"""Demo tabs besides the missions: live camera, coordinates validation, spoofing / stress when their data is local."""
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
    monkeypatch.setattr(server, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(extras, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(server, "manager", jobs.JobManager())
    return TestClient(server.app)


def test_live_camera_tracks_frames(client, tmp_path, monkeypatch):
    from app import extras

    imwrite(tmp_path / "map.png", make_world(1600, 1200, seed=3))
    data = make_flight(tmp_path / "syn", n_frames=6, frame_size=(400, 300), turns=0.3, seed=3,
                       world_path=tmp_path / "map.png", frame_scale=0.5)
    monkeypatch.setattr(extras, "_live_maps", lambda: [{"id": "t", "label": "t", "map": str(tmp_path / "map.png")}])
    s = client.post("/api/live/start", data={"map_id": "t", "method": "track_sift"}).json()
    assert s["map_size"] == [1600, 1200]
    assert client.get(f"/api/live/{s['id']}/map").status_code == 200
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


def test_spoofing_tab(client):
    info = client.get("/api/spoof/info").json()
    if not info["available"]:
        pytest.skip("real flight data/real/visloc_24 is not local")
    m = info["methods"][0]["id"]
    r = client.get("/api/spoof/run", params={"method": m, "start": 80, "rate": 3}).json()
    assert len(r["truth"]) == len(r["gps"]) == info["n_frames"]
    assert r["stats"]["alarm"] is not None and not r["stats"]["false_alarm"]
    assert r["stats"]["err_fixed_m"] < 20 < r["stats"]["err_gps_m"]


def test_stress_tab(client):
    info = client.get("/api/stress/info").json()
    assert {a["name"] for a in info["axes"]} >= {"blur", "clouds", "darkness"}
    if not info["available"]:
        pytest.skip("stress area is not local")
    r = client.get("/api/stress/try", params={"axis": "clouds", "level_idx": 0, "method": "sift_flann"}).json()
    assert r["image"].startswith("data:image/jpeg") and r["success"]
    assert client.get("/api/stress/try", params={"axis": "nope"}).status_code == 400
