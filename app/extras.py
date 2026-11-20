"""Interactive tabs of the demo besides the missions: GPS spoofing, degradations and the live camera, plus the
street / satellite basemaps for georeferenced maps.

    GET  /api/basemap/{kind}?map=<path>     street or satellite map aligned with a georeferenced map (JPEG)
    GET  /api/spoof/sources                 flights with GPS and visual fixes (real flight + finished missions)
    GET  /api/spoof/run?source=..&...       spoofing attack + detector for the given sliders
    GET  /api/stress/info                   sources, axes, methods
    POST /api/stress/upload                 own frame (+ map) -> new source
    GET  /api/stress/try?...                one degraded frame localised by one method
    GET  /api/stress/curve?...              success rate over all levels of an axis (computed on the fly)
    GET  /api/live/info                     maps and methods for the live camera
    POST /api/live/start                    map + method -> session
    POST /api/live/{id}/frame               JPEG body -> pose of the frame on the map
"""
from __future__ import annotations

import base64
import hashlib
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from vo.config import load_config, resolve_device
from vo.io import imread, imwrite, resize_long
from vo.tiles import TILE, GeoRef, fetch_box

from .catalog import DEFAULT_MAP, ROOT, presets
from .jobs import MAP_PREVIEW_LONG, RESULTS_DIR, clean

router = APIRouter()
CACHE = RESULTS_DIR / "_cache"
UPLOADS = ROOT / "data" / "app_uploads" / "stress"
_lock = threading.Lock()
_manager = None  # set by server.py: finished missions are spoofing sources


def bind_manager(manager) -> None:
    global _manager
    _manager = manager


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


def _key(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _geo_for(map_path: Path) -> Path | None:
    p = Path(map_path).parent / "geo.json"
    return p if p.exists() else None


# ---------------------------------------------------------------- basemaps
BASEMAPS = {"streets": "esri_streets", "satellite": "esri"}
MAX_BASEMAP_TILES = 600


def basemap(geo_path: Path, kind: str) -> Path:
    """Street / satellite map of exactly the box of a georeferenced map, at the preview resolution."""
    if kind not in BASEMAPS:
        raise HTTPException(400, "kind: streets | satellite")
    geo = GeoRef.load(geo_path)
    out = CACHE / f"basemap_{kind}_{_key(geo_path, geo.zoom, geo.x0, geo.y0, geo.width)}.jpg"
    if out.exists():
        return out
    z, k = geo.zoom, 1.0
    while z > 1 and (geo.width * k / TILE + 1) * (geo.height * k / TILE + 1) > MAX_BASEMAP_TILES:
        z, k = z - 1, k / 2  # a large map: coarser tiles, upsampled (it is a backdrop only)
    img, _ = fetch_box(int(geo.x0 * k), int(geo.y0 * k), max(1, int(geo.width * k)), max(1, int(geo.height * k)), z,
                       BASEMAPS[kind], ROOT / "data" / "cache" / "tiles")
    img = cv2.resize(img, (geo.width, geo.height), interpolation=cv2.INTER_LINEAR)
    img, _ = resize_long(img, MAP_PREVIEW_LONG)
    out.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


@router.get("/api/basemap/{kind}")
def basemap_for_map(kind: str, map: str):
    """`map` is a map path the server already knows (preset, mission or real flight) with geo.json next to it."""
    known = {str(Path(p["map"])) for p in presets()} | {str(REAL / "map.jpg")}
    if _manager is not None:
        known |= {str(j.map_path) for j in _manager.jobs.values()}
    if map not in known or _geo_for(Path(map)) is None:
        raise HTTPException(404, "map has no georeference")
    try:
        return FileResponse(basemap(_geo_for(Path(map)), kind), media_type="image/jpeg")
    except OSError as e:
        raise HTTPException(503, f"немає доступу до тайлів Esri: {e}")


# ---------------------------------------------------------------- GPS spoofing
REAL = ROOT / "data" / "real" / "visloc_24"
ORACLE = ROOT / "experiments" / "oracle_roi" / "visloc_24"
ISLANDS = ROOT / "experiments" / "islands"
FIX_LABELS = {"track_xfeat": "XFeat", "track_loftr": "LoFTR", "track_eloftr_official": "Efficient LoFTR",
              "track_superpoint_lg": "SuperPoint + LightGlue", "track_aliked_lg": "ALIKED + LightGlue",
              "track_disk_lg": "DISK + LightGlue", "track_superglue_official": "SuperGlue",
              "track_sift_lg": "SIFT + LightGlue", "track_sift": "SIFT"}


def _real_available() -> bool:
    return all((REAL / p).exists() for p in ("map.jpg", "geo.json", "flight/gt.csv"))


def _fixes_from(path: Path) -> pd.DataFrame:
    """name, x, y (map px) of the frames with a confirmed visual fix."""
    d = pd.read_csv(path)
    if "ok" in d.columns:  # experiments/oracle_roi
        ok = d["ok"].astype(str).str.lower().isin(["true", "1"])
    else:  # experiments/islands/*/positions.csv (an exact fix or a verified island) or a mission trajectory.csv
        ok = d["status"].isin(["map", "island_verified"])
    return d.loc[ok, ["name", "x", "y"]].astype({"name": str})


def _spoof_sources() -> list[dict]:
    out = []
    if _real_available():
        flight = "Реальний політ UAV-VisLoc #24"
        real = {"map": REAL / "map.jpg", "gt": REAL / "flight" / "gt.csv", "kind": "real"}
        by_fixes = sorted(ORACLE.glob("track_*.csv"), key=lambda p: -len(_fixes_from(p)))  # best method first
        for p in by_fixes:
            out.append({"id": f"oracle:{p.stem}", "group": f"{flight}: вікно навколо відомої позиції (верхня межа)",
                        "label": FIX_LABELS.get(p.stem, p.stem), "path": p, **real})
        for d in sorted(ISLANDS.glob("visloc_24_dinoft_track_*")):
            m = d.name.removeprefix("visloc_24_dinoft_")
            if (d / "positions.csv").exists():
                out.append({"id": f"islands:{d.name}", "group": f"{flight}: острови + DINOv2, масштаб з цього польоту",
                            "label": FIX_LABELS.get(m, m), "path": d / "positions.csv", **real})
        for d in sorted(ISLANDS.glob("visloc_24_scale*_dinov2")):
            if (d / "positions.csv").exists():
                s = d.name.split("scale")[1].split("_")[0]
                out.append({"id": f"islands:{d.name}", "group": f"{flight}: острови + DINOv2, масштаб без підказки",
                            "label": f"масштаб {s}", "path": d / "positions.csv", **real})
    if _manager is not None:  # finished missions on a georeferenced map with a GPS ground truth
        for j in sorted(_manager.jobs.values(), key=lambda j: -j.created):
            traj = j.out_dir / "trajectory.csv"
            if j.status == "done" and traj.exists() and _geo_for(j.map_path) and j.gt_path and j.gt_path.exists():
                if {"lon", "lat", "name"} <= set(pd.read_csv(j.gt_path, nrows=1).columns):
                    out.append({"id": f"job:{j.id}", "group": "Мої місії з GPS", "label": f"{j.title} · {j.method}",
                                "path": traj, "map": j.map_path, "gt": j.gt_path, "kind": j.data_kind})
    return out


@router.get("/api/spoof/sources")
def spoof_sources():
    out = []
    for s in _spoof_sources():
        n = len(pd.read_csv(s["gt"]))
        out.append({k: s[k] for k in ("id", "group", "label", "kind")}
                   | {"fixes_pct": round(100 * len(_fixes_from(s["path"])) / max(n, 1)), "n_frames": n,
                      "map": str(s["map"])})
    hint = "" if out else ("Немає польотів із GPS. Запустіть місію з GPS (вкладка «Місії»: «Координати» або «Свої дані» "
                           "+ CSV з GPS) або додайте реальний політ (scripts/import_uavvisloc.py).")
    return {"sources": out, "hint": hint}


def _source(source: str) -> dict:
    s = next((s for s in _spoof_sources() if s["id"] == source), None)
    if s is None:
        raise HTTPException(404, "unknown source")
    return s


@router.get("/api/spoof/map")
def spoof_map(source: str):
    s = _source(source)
    return FileResponse(_preview(Path(s["map"]), f"map_{_key(s['map'])}.jpg"), media_type="image/jpeg")


@router.get("/api/spoof/run")
def spoof_run(source: str, start: int = 80, rate: float = 3.0, max_m: float = 300.0, bearing: float = 135.0,
              threshold: float = 15.0, confirm: int = 3):
    from vo.spoofing import EARTH_RADIUS_M, detect, spoof, to_enu

    s = _source(source)
    geo = GeoRef.load(_geo_for(Path(s["map"])))
    gt = pd.read_csv(s["gt"])
    origin = (float(gt["lon"].iloc[0]), float(gt["lat"].iloc[0]))
    truth = to_enu(gt["lon"], gt["lat"], *origin)
    v = _fixes_from(s["path"]).set_index("name").reindex(gt["name"].astype(str))
    lon, lat = geo.to_lonlat(v["x"].to_numpy(float), v["y"].to_numpy(float))
    visual = to_enu(lon, lat, *origin)

    start = int(np.clip(start, 0, len(truth) - 1))
    gps = spoof(truth, start, rate, max_m, bearing)
    det = detect(gps, visual, threshold, max(1, confirm))
    honest = detect(truth, visual, threshold, max(1, confirm))
    err_gps = np.linalg.norm(gps - truth, axis=1)
    err_nav = np.linalg.norm(det.position - truth, axis=1)
    err_vis = np.linalg.norm(visual - truth, axis=1)
    stats = {"alarm": det.alarm, "false_alarm": honest.alarm, "fixes": int(np.isfinite(visual[:, 0]).sum()),
             "vis_err_median_m": float(np.nanmedian(err_vis)) if np.isfinite(err_vis).any() else None}
    if det.alarm is not None:
        after = slice(det.alarm, None)
        stats.update(delay_frames=det.alarm - start + 1, drag_m=min((det.alarm - start + 1) * rate, max_m),
                     err_gps_m=float(np.median(err_gps[after])), err_fixed_m=float(np.median(err_nav[after])))
    else:
        stats.update(err_gps_m=float(np.median(err_gps[start:])))

    def px(enu):
        la = origin[1] + np.degrees(enu[:, 1] / EARTH_RADIUS_M)
        lo = origin[0] + np.degrees(enu[:, 0] / (EARTH_RADIUS_M * np.cos(np.radians(origin[1]))))
        return np.stack(geo.to_px(lo, la), 1)

    return clean({"truth": px(truth), "gps": px(gps), "visual": px(visual), "position": px(det.position),
                  "residual_m": det.residual_m, "err_gps_m": err_gps, "err_nav_m": err_nav, "start": start,
                  "threshold": threshold, "stats": stats, "map_size": [geo.width, geo.height]})


# ---------------------------------------------------------------- degradations
STRESS_METHODS = {"sift_flann": "SIFT", "orb_bf": "ORB", "akaze_bf": "AKAZE"}  # the methods of the stress test
SUCCESS_M = 5.0
_stress_up: dict = {}  # uploaded frames: id -> {"frame", "map", "ref", "mpp", "georef", "label"}


def _stress_areas() -> list[dict]:
    out = []
    for p in presets():
        a = Path(p["map"]).parent
        if p["id"].startswith("area_") and (a / "detail.jpg").exists() and (a / "geo.json").exists():
            out.append({"id": p["id"], "label": p["label"].replace(" (є GT)", ""), "dir": a})
    return out


@lru_cache(maxsize=4)
def _area(area_dir: str, n_frames: int = 10):
    a = Path(area_dir)
    map_bgr, detail = imread(a / "map.jpg"), imread(a / "detail.jpg")
    gt = pd.read_csv(a / "flight" / "gt.csv")
    picks = np.unique(np.linspace(0, len(gt) - 1, n_frames).round().astype(int))
    return map_bgr, detail, detail.shape[1] / map_bgr.shape[1], gt, picks, GeoRef.load(a / "geo.json")


_localizers: dict = {}


def _map_localizer(map_path: Path, method: str):
    """One MapLocalizer per (map, method): the map features take seconds, every frame after that is fast."""
    from vo.localize import MapLocalizer

    key = (str(map_path), method)
    with _lock:
        if key not in _localizers:
            if len(_localizers) > 6:
                _localizers.pop(next(iter(_localizers)))
            cfg = load_config(ROOT / "configs" / "map" / "methods" / f"{method}.yaml",
                              [f"map.path={Path(map_path).as_posix()}"])
            _localizers[key] = MapLocalizer(imread(map_path), cfg, resolve_device(cfg["device"]))
        return _localizers[key]


def _stress_frame(source: str, axis: str, li: int, frame: int):
    """(degraded frame, map path, reference centre in map px, metres per map px, axis, level index)."""
    from vo.stress import AXES, degraded

    if axis not in AXES:
        raise HTTPException(400, "unknown axis")
    ax = AXES[axis]
    li = int(np.clip(li, 0, len(ax.levels) - 1))
    ai = list(AXES).index(axis)
    if source in _stress_up:
        u = _stress_up[source]
        if ax.geometric:
            raise HTTPException(400, "висота й нахил — лише для районів (кадр перемальовується зі знімка району)")
        img = degraded(ax, ax.levels[li], u["frame"], None, 1.0, None, seed=hash((ai, li, 0)) % 2**32)
        return img, u["map"], np.array(u["ref"]), u["mpp"], ax, li
    area = next((a for a in _stress_areas() if a["id"] == source), None)
    if area is None:
        raise HTTPException(404, "unknown source")
    map_bgr, detail, k, gt, picks, geo = _area(str(area["dir"]))
    i = int(picks[int(np.clip(frame, 0, len(picks) - 1))])
    clean_img = imread(area["dir"] / "flight" / "frames" / f"{gt.name[i]}.jpg")
    img = degraded(ax, ax.levels[li], clean_img, detail, k, gt.iloc[i], seed=hash((ai, li, i)) % 2**32)
    return img, area["dir"] / "map.jpg", gt.loc[i, ["x", "y"]].to_numpy(float), geo.meters_per_px(), ax, li


@router.get("/api/stress/info")
def stress_info():
    from vo.stress import AXES

    sources = [{"id": a["id"], "label": a["label"], "kind": "synthetic", "frames": 10, "geometric": True,
                "saved": _saved_curves(a["dir"].name)} for a in _stress_areas()]
    sources += [{"id": k, "label": u["label"], "kind": "own", "frames": 1, "geometric": False}
                for k, u in _stress_up.items()]
    return {"sources": sources, "methods": [{"id": k, "label": v} for k, v in STRESS_METHODS.items()],
            "axes": [{"name": a.name, "label": a.label, "levels": a.levels, "geometric": bool(a.geometric)}
                     for a in AXES.values()], "success_m": SUCCESS_M,
            "hint": "" if sources else "Немає районів локально (scripts/make_area.py --area lviv_old_town) — "
                                       "завантажте свій кадр."}


def _saved_curves(area_name: str) -> list[dict]:
    """Curves of the stress-test run (scripts/run_stress.py, 30 frames per level), if it was made for this area."""
    p = ROOT / "experiments" / "stress" / area_name / "summary.csv"
    if not p.exists():
        return []
    s = pd.read_csv(p).sort_values("level_idx")
    return [{"method": m, "axis": ax, "levels": g["level"].tolist(), "success": g["success"].tolist(), "frames": 30}
            for (m, ax), g in s.groupby(["method", "axis"], sort=False)]


@router.post("/api/stress/upload")
async def stress_upload(frame: UploadFile = File(...), map_file: UploadFile | None = File(None),
                        method: str = Form("sift_flann"), frame_scale: float = Form(0.5)):
    """Own frame: its clean localisation is the reference; success = the degraded frame lands within 5 m of it."""
    if method not in STRESS_METHODS:
        raise HTTPException(400, "unknown method")
    sid = "up_" + uuid.uuid4().hex[:6]
    d = UPLOADS / sid
    d.mkdir(parents=True, exist_ok=True)
    img = cv2.imdecode(np.frombuffer(await frame.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "кадр має бути зображенням")
    if map_file is not None and map_file.filename:
        m = cv2.imdecode(np.frombuffer(await map_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if m is None:
            raise HTTPException(400, "мапа має бути зображенням")
        map_path = d / "map.jpg"
        imwrite(map_path, m, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif DEFAULT_MAP.exists():
        map_path = DEFAULT_MAP
    else:
        raise HTTPException(400, "завантажте мапу (мапи завдання локально немає)")
    # the stress methods take frame_scale from their config; another scale is applied to the frame itself
    base = load_config(ROOT / "configs" / "map" / "methods" / f"{method}.yaml")["localize"]["frame_scale"]
    if abs(frame_scale - base) > 1e-6:
        img = cv2.resize(img, None, fx=frame_scale / base, fy=frame_scale / base, interpolation=cv2.INTER_AREA)
    fix = _map_localizer(map_path, method).localize(img)
    if not fix.ok:
        raise HTTPException(400, f"кадр не знайдено на мапі навіть без спотворень ({fix.reason}); "
                                 "перевірте мапу або масштаб кадру")
    geo = _geo_for(map_path)
    _stress_up[sid] = {"frame": img, "map": map_path, "ref": fix.center.tolist(), "georef": bool(geo),
                       "mpp": GeoRef.load(geo).meters_per_px() if geo else 1.0,  # without georeference: px
                       "label": f"Свій кадр: {Path(frame.filename or 'frame').name}"}
    return {"id": sid, "label": _stress_up[sid]["label"], "georef": bool(geo)}


@router.get("/api/stress/map")
def stress_map(source: str):
    if source in _stress_up:
        p = _stress_up[source]["map"]
    else:
        a = next((a for a in _stress_areas() if a["id"] == source), None)
        if a is None:
            raise HTTPException(404, "unknown source")
        p = a["dir"] / "map.jpg"
    return FileResponse(_preview(Path(p), f"map_{_key(p)}.jpg"), media_type="image/jpeg")


def _judge(fix, ref, mpp) -> tuple[bool, float | None]:
    if not fix.ok:
        return False, None
    err = float(np.hypot(*(fix.center - ref))) * mpp
    return err < SUCCESS_M, err


@router.get("/api/stress/try")
def stress_try(source: str, axis: str = "clouds", level_idx: int = 0, method: str = "sift_flann", frame: int = 0):
    if method not in STRESS_METHODS:
        raise HTTPException(400, "unknown method")
    img, map_path, ref, mpp, ax, li = _stress_frame(source, axis, level_idx, frame)
    t = time.perf_counter()
    fix = _map_localizer(map_path, method).localize(img)
    ms = 1000 * (time.perf_counter() - t)
    ok, err = _judge(fix, ref, mpp)
    georef = _stress_up[source]["georef"] if source in _stress_up else True
    loc = _map_localizer(map_path, method)
    return {"image": _jpeg_b64(img), "level": ax.levels[li], "ref": ref.tolist(), "ms": ms, "err": err,
            "units": "м" if georef else "px", "success": ok, "map_size": list(loc.map_size), **_pose(fix)}


_curves: dict = {}


@router.get("/api/stress/curve")
def stress_curve(source: str, axis: str, method: str):
    """Success rate at every level of the axis over the source's frames (10 frames of an area, 1 own frame)."""
    from vo.stress import AXES

    if method not in STRESS_METHODS or axis not in AXES:
        raise HTTPException(400, "unknown method or axis")
    key = (source, axis, method)
    if key not in _curves:
        ax = AXES[axis]
        if source in _stress_up:
            n = 1
        else:
            area = next((a for a in _stress_areas() if a["id"] == source), None)
            if area is None:
                raise HTTPException(404, "unknown source")
            n = len(_area(str(area["dir"]))[4])
        success = []
        for li in range(len(ax.levels)):
            hits = 0
            for f in range(n):
                img, map_path, ref, mpp, _, _ = _stress_frame(source, axis, li, f)
                hits += _judge(_map_localizer(map_path, method).localize(img), ref, mpp)[0]
            success.append(hits / n)
        _curves[key] = {"method": STRESS_METHODS[method], "levels": ax.levels, "success": success, "frames": n}
    return _curves[key]


# ---------------------------------------------------------------- live camera
LIVE_METHODS = {"track_sift": ("SIFT + FLANN, трекінг", "configs/map/methods/track_sift.yaml"),
                "track_akaze": ("AKAZE, трекінг", "configs/map/methods/track_akaze.yaml"),
                "track_orb": ("ORB, трекінг", "configs/map/methods/track_orb.yaml"),
                "baseline": ("SIFT + FLANN, вся мапа (без трекінгу)", "configs/map/baseline_sift.yaml")}
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
    return {"maps": [{"id": m["id"], "label": m["label"], "map": m["map"],
                      "geo": _geo_for(Path(m["map"])) is not None} for m in _live_maps()],
            "methods": [{"id": k, "label": v[0]} for k, v in LIVE_METHODS.items()]}


@router.get("/api/live/preview/{map_id}")
def live_preview(map_id: str):
    m = next((m for m in _live_maps() if m["id"] == map_id), None)
    if m is None:
        raise HTTPException(404, "unknown map")
    return FileResponse(_preview(Path(m["map"]), f"map_{_key(m['map'])}.jpg"), media_type="image/jpeg")


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
        _live[sid] = {"loc": loc, "map": Path(m["map"]), "n": 0}
    return {"id": sid, "map_size": [map_bgr.shape[1], map_bgr.shape[0]], "map": m["map"]}


def _session(sid: str) -> dict:
    s = _live.get(sid)
    if s is None:
        raise HTTPException(404, "live session not found: press Start again")
    return s


@router.post("/api/live/{sid}/frame")
async def live_frame(sid: str, request: Request):
    s = _session(sid)
    frame = cv2.imdecode(np.frombuffer(await request.body(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "not an image")
    t = time.perf_counter()
    with _lock:
        fix = s["loc"].localize(frame)
        s["n"] += 1
    return {"ms": 1000 * (time.perf_counter() - t), "k": s["n"], **_pose(fix)}
