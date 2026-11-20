"""Interactive tabs of the demo besides the missions: GPS spoofing, degradations (stress test) and the live camera.

    GET  /api/spoof/info              real flight + methods with visual fixes (experiments/oracle_roi/visloc_24)
    GET  /api/spoof/map               map of the real flight (JPEG)
    GET  /api/spoof/run?...           spoofing attack + detector for the given sliders (map px, metres)
    GET  /api/stress/info             axes, levels, methods, curves from experiments/stress/<area>/summary.csv
    GET  /api/stress/map              map of the stress-test area (JPEG)
    GET  /api/stress/try?...          one degraded frame localised by one method
    POST /api/live/start              map + method for the live camera -> {"id", "map_size"}
    GET  /api/live/{id}/map           map preview (JPEG)
    POST /api/live/{id}/frame         JPEG body -> pose of the frame on the map
"""
from __future__ import annotations

import base64
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse

from vo.config import load_config, resolve_device
from vo.io import imread, imwrite, resize_long
from vo.tiles import GeoRef

from .catalog import DEFAULT_MAP, ROOT, presets
from .jobs import MAP_PREVIEW_LONG, RESULTS_DIR, clean

router = APIRouter()
CACHE = RESULTS_DIR / "_cache"
_lock = threading.Lock()


def _preview(src: Path, name: str) -> Path:
    out = CACHE / name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        img, _ = resize_long(imread(src), MAP_PREVIEW_LONG)
        imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def _jpeg_b64(img: np.ndarray, long: int = 720) -> str:
    img, _ = resize_long(img, long)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _pose(fix) -> dict:
    return clean({"ok": bool(fix.ok), "x": fix.center[0], "y": fix.center[1], "corners": fix.corners,
                  "heading": fix.heading_deg, "n_inliers": fix.n_inliers, "n_matches": fix.n_matches,
                  "reason": fix.reason, "source": getattr(fix, "source", "global")})


# ---------------------------------------------------------------- GPS spoofing on the real flight
REAL = ROOT / "data" / "real" / "visloc_24"
FIXES = ROOT / "experiments" / "oracle_roi" / "visloc_24"
FIX_LABELS = {"track_xfeat": "XFeat", "track_loftr": "LoFTR", "track_eloftr_official": "Efficient LoFTR",
              "track_superpoint_lg": "SuperPoint + LightGlue", "track_aliked_lg": "ALIKED + LightGlue",
              "track_disk_lg": "DISK + LightGlue", "track_superglue_official": "SuperGlue",
              "track_sift_lg": "SIFT + LightGlue", "track_sift": "SIFT"}


def _spoof_available() -> bool:
    return all((REAL / p).exists() for p in ("map.jpg", "geo.json", "flight/gt.csv")) and FIXES.is_dir()


@lru_cache(maxsize=1)
def _flight():
    from vo.spoofing import to_enu

    geo = GeoRef.load(REAL / "geo.json")
    gt = pd.read_csv(REAL / "flight" / "gt.csv")
    origin = (float(gt["lon"].iloc[0]), float(gt["lat"].iloc[0]))
    return geo, gt, origin, to_enu(gt["lon"], gt["lat"], *origin)


def _enu_to_px(enu: np.ndarray, geo: GeoRef, origin) -> np.ndarray:
    from vo.spoofing import EARTH_RADIUS_M

    lat = origin[1] + np.degrees(enu[:, 1] / EARTH_RADIUS_M)
    lon = origin[0] + np.degrees(enu[:, 0] / (EARTH_RADIUS_M * np.cos(np.radians(origin[1]))))
    return np.stack(geo.to_px(lon, lat), 1)


@router.get("/api/spoof/info")
def spoof_info():
    if not _spoof_available():
        return {"available": False, "hint": "потрібні data/real/visloc_24 (scripts/import_uavvisloc.py) і "
                                            "experiments/oracle_roi/visloc_24"}
    geo, gt, _, _ = _flight()
    methods = []
    for p in sorted(FIXES.glob("track_*.csv")):
        ok = pd.read_csv(p)["ok"].astype(bool).mean()
        methods.append({"id": p.stem, "label": FIX_LABELS.get(p.stem, p.stem), "fixes": round(100 * float(ok))})
    methods.sort(key=lambda m: -m["fixes"])
    return {"available": True, "n_frames": len(gt), "map_size": [geo.width, geo.height], "methods": methods,
            "step_m": float(np.median(np.hypot(np.diff(gt["x"]), np.diff(gt["y"])))) * geo.meters_per_px()}


@router.get("/api/spoof/map")
def spoof_map():
    if not _spoof_available():
        raise HTTPException(404, "real flight not available")
    return FileResponse(_preview(REAL / "map.jpg", "spoof_map.jpg"), media_type="image/jpeg")


@router.get("/api/spoof/run")
def spoof_run(method: str = "track_xfeat", start: int = 80, rate: float = 3.0, max_m: float = 300.0,
              bearing: float = 135.0, threshold: float = 15.0, confirm: int = 3):
    from vo.spoofing import detect, spoof, to_enu

    path = FIXES / f"{Path(method).name}.csv"
    if not _spoof_available() or not path.exists():
        raise HTTPException(404, "no visual fixes for this method")
    geo, gt, origin, truth = _flight()
    v = pd.read_csv(path).set_index("name").reindex(gt["name"].astype(str))
    ok = v["ok"].fillna(False).astype(bool).to_numpy()
    lon, lat = geo.to_lonlat(v["x"].to_numpy(float), v["y"].to_numpy(float))
    visual = to_enu(lon, lat, *origin)
    visual[~ok] = np.nan

    start = int(np.clip(start, 0, len(truth) - 1))
    gps = spoof(truth, start, rate, max_m, bearing)
    det = detect(gps, visual, threshold, max(1, confirm))
    honest = detect(truth, visual, threshold, max(1, confirm))
    err_gps = np.linalg.norm(gps - truth, axis=1)
    err_nav = np.linalg.norm(det.position - truth, axis=1)
    stats = {"alarm": det.alarm, "false_alarm": honest.alarm is not None, "fixes": int(ok.sum())}
    if det.alarm is not None:
        after = slice(det.alarm, None)
        stats.update(delay_frames=det.alarm - start + 1, drag_m=min((det.alarm - start + 1) * rate, max_m),
                     err_gps_m=float(np.median(err_gps[after])), err_fixed_m=float(np.median(err_nav[after])))
    else:
        stats.update(err_gps_m=float(np.median(err_gps[start:])))
    return clean({"truth": _enu_to_px(truth, geo, origin), "gps": _enu_to_px(gps, geo, origin),
                  "visual": _enu_to_px(visual, geo, origin), "position": _enu_to_px(det.position, geo, origin),
                  "residual_m": det.residual_m, "err_gps_m": err_gps, "err_nav_m": err_nav, "start": start,
                  "threshold": threshold, "stats": stats})


# ---------------------------------------------------------------- degradations (stress test)
STRESS_AREA = "lviv_old_town"
STRESS_METHODS = {"sift_flann": "SIFT", "orb_bf": "ORB", "akaze_bf": "AKAZE"}
STRESS_FRAMES = 30


def _area_dir() -> Path:
    return ROOT / "data" / "areas" / STRESS_AREA


def _stress_available() -> bool:
    a = _area_dir()
    return all((a / p).exists() for p in ("map.jpg", "detail.jpg", "geo.json", "flight/gt.csv"))


@lru_cache(maxsize=1)
def _area():
    a = _area_dir()
    map_bgr, detail = imread(a / "map.jpg"), imread(a / "detail.jpg")
    gt = pd.read_csv(a / "flight" / "gt.csv")
    picks = np.unique(np.linspace(0, len(gt) - 1, STRESS_FRAMES).round().astype(int))
    return map_bgr, detail, detail.shape[1] / map_bgr.shape[1], gt, picks, GeoRef.load(a / "geo.json")


_localizers: dict = {}


def _map_localizer(key, map_bgr, config: str, overrides: list[str]):
    """One MapLocalizer per (map, method): the map features take seconds, every frame after that is fast."""
    from vo.localize import MapLocalizer

    with _lock:
        if key not in _localizers:
            cfg = load_config(ROOT / config, overrides)
            _localizers[key] = MapLocalizer(map_bgr, cfg, resolve_device(cfg["device"]))
        return _localizers[key]


@router.get("/api/stress/info")
def stress_info():
    from vo.stress import AXES

    curves = []
    summ = ROOT / "experiments" / "stress" / STRESS_AREA / "summary.csv"
    if summ.exists():
        s = pd.read_csv(summ)
        curves = [{"method": m, "axis": ax, "levels": g["level"].tolist(), "success": g["success"].tolist()}
                  for (m, ax), g in s.sort_values("level_idx").groupby(["method", "axis"], sort=False)]
    out = {"available": _stress_available(), "area": STRESS_AREA, "curves": curves,
           "axes": [{"name": a.name, "label": a.label, "levels": a.levels} for a in AXES.values()],
           "methods": [{"id": k, "label": v} for k, v in STRESS_METHODS.items()], "n_frames": STRESS_FRAMES}
    if out["available"]:
        map_bgr, *_, geo = _area()
        out.update(map_size=[map_bgr.shape[1], map_bgr.shape[0]], m_per_px=geo.meters_per_px())
    else:
        out["hint"] = f"потрібен район data/areas/{STRESS_AREA} (scripts/make_area.py --area {STRESS_AREA})"
    return out


@router.get("/api/stress/map")
def stress_map():
    if not _stress_available():
        raise HTTPException(404, "area not available")
    return FileResponse(_preview(_area_dir() / "map.jpg", f"stress_{STRESS_AREA}.jpg"), media_type="image/jpeg")


@router.get("/api/stress/try")
def stress_try(axis: str = "clouds", level_idx: int = 0, method: str = "sift_flann", frame: int = 0):
    from vo.stress import AXES, degraded

    if not _stress_available():
        raise HTTPException(404, "area not available")
    if axis not in AXES or method not in STRESS_METHODS:
        raise HTTPException(400, "unknown axis or method")
    map_bgr, detail, k, gt, picks, geo = _area()
    ax = AXES[axis]
    li = int(np.clip(level_idx, 0, len(ax.levels) - 1))
    i = int(picks[int(np.clip(frame, 0, len(picks) - 1))])
    ai = list(AXES).index(axis)
    clean_img = imread(_area_dir() / "flight" / "frames" / f"{gt.name[i]}.jpg")
    img = degraded(ax, ax.levels[li], clean_img, detail, k, gt.iloc[i], seed=hash((ai, li, i)) % 2**32)
    loc = _map_localizer(("stress", method), map_bgr, f"configs/map/methods/{method}.yaml",
                         [f"map.path={(_area_dir() / 'map.jpg').as_posix()}"])
    t = time.perf_counter()
    fix = loc.localize(img)
    ms = 1000 * (time.perf_counter() - t)
    g = gt.loc[i, ["x", "y"]].to_numpy(float)
    err_px = float(np.hypot(*(fix.center - g))) if fix.ok else None
    return {"image": _jpeg_b64(img), "level": ax.levels[li], "gt": g.tolist(), "ms": ms,
            "err_m": err_px * geo.meters_per_px() if err_px is not None else None,
            "success": err_px is not None and err_px < 5, **_pose(fix)}


# ---------------------------------------------------------------- live camera
LIVE_METHODS = {"track_sift": ("SIFT + FLANN, трекінг", "configs/map/methods/track_sift.yaml"),
                "baseline": ("SIFT + FLANN, вся мапа", "configs/map/baseline_sift.yaml")}
_live: dict = {}


def _live_maps() -> list[dict]:
    out = [{"id": "task", "label": "Мапа завдання", "map": str(DEFAULT_MAP)}] if DEFAULT_MAP.exists() else []
    seen = {str(DEFAULT_MAP)}
    for p in presets():
        if p["map"] not in seen:
            seen.add(p["map"])
            out.append({"id": p["id"], "label": p["label"].replace(" (є GT)", ""), "map": p["map"]})
    return out


@router.get("/api/live/info")
def live_info():
    return {"maps": [{"id": m["id"], "label": m["label"]} for m in _live_maps()],
            "methods": [{"id": k, "label": v[0]} for k, v in LIVE_METHODS.items()]}


@router.get("/api/live/preview/{map_id}")
def live_preview(map_id: str):
    m = next((m for m in _live_maps() if m["id"] == map_id), None)
    if m is None:
        raise HTTPException(404, "unknown map")
    return FileResponse(_preview(Path(m["map"]), f"live_{map_id}.jpg"), media_type="image/jpeg")


@router.post("/api/live/start")
def live_start(map_id: str = Form("task"), method: str = Form("track_sift"), frame_scale: float = Form(0.5)):
    m = next((m for m in _live_maps() if m["id"] == map_id), None)
    if m is None or method not in LIVE_METHODS:
        raise HTTPException(400, "unknown map or method")
    cfg = load_config(ROOT / LIVE_METHODS[method][1], [f"map.path={Path(m['map']).as_posix()}",
                                                        f"localize.frame_scale={frame_scale}"])
    map_bgr = imread(m["map"])
    device = resolve_device(cfg["device"])
    if cfg["localize"].get("mode") == "track":
        from vo.track import TrackLocalizer

        loc = TrackLocalizer(map_bgr, cfg, device)
    else:
        from vo.localize import MapLocalizer

        loc = MapLocalizer(map_bgr, cfg, device)
    sid = uuid.uuid4().hex[:8]
    with _lock:
        _live.clear()  # one camera at a time
        _live[sid] = {"loc": loc, "map": Path(m["map"]), "map_id": map_id, "n": 0}
    return {"id": sid, "map_size": [map_bgr.shape[1], map_bgr.shape[0]]}


def _session(sid: str) -> dict:
    s = _live.get(sid)
    if s is None:
        raise HTTPException(404, "live session not found: press Start again")
    return s


@router.get("/api/live/{sid}/map")
def live_map(sid: str):
    s = _session(sid)
    return FileResponse(_preview(s["map"], f"live_{s['map_id']}.jpg"), media_type="image/jpeg")


@router.post("/api/live/{sid}/frame")
async def live_frame(sid: str, request: Request):
    s = _session(sid)
    data = np.frombuffer(await request.body(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "not an image")
    t = time.perf_counter()
    with _lock:
        fix = s["loc"].localize(frame)
        s["n"] += 1
    return {"ms": 1000 * (time.perf_counter() - t), "k": s["n"], **_pose(fix)}
