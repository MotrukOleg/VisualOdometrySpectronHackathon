"""Build a localisation benchmark over ANY area: satellite map from public tiles + synthetic flight with exact GT.

The map is downloaded at `zoom`; the same ground box is downloaded again at `zoom + detail`, and the UAV frames are
rendered from that detailed image. So frames carry real fine detail that is absent from the map (as in the task
crops, frame -> map scale 2^-detail = 0.25), instead of being upsampled pieces of the map itself.
Rural areas often have no imagery at zoom 19; then the most detailed available zoom is used and upsampled.

--frames-from wayback:<release> (or wayback:oldest) cuts the frames from an ARCHIVED image of the same place
(Esri Wayback, back to ~2010) while the map stays current: the drone flies years after the map was made.
Output then goes to detail_<tag>.jpg and flight_<tag>/ (tag = capture year, e.g. wb2010), next to the
current-imagery flight; capture.json records the capture date and sensor of the frames' imagery.

Writes to --out (default data/areas/<name>):
    map.jpg          satellite map (the input for localisation)
    geo.json         georeference: map pixel <-> lon/lat (vo.tiles.GeoRef)
    detail.jpg       the same box at zoom + detail (frames are cut from it)
    flight/frames/   UAV frames
    flight/gt.csv    ground truth in MAP pixels (+ lat, lon)

Examples:
    python scripts/make_area.py --area lviv_old_town
    python scripts/make_area.py --name my_village --center 49.55,25.59 --size-km 2x1.5 --degrade screen
    python scripts/make_area.py --area lviv_old_town --frames-from wayback:oldest   # frames from ~2010 imagery
    python scripts/localize.py --map data/areas/lviv_old_town/map.jpg \
        --frames data/areas/lviv_old_town/flight/frames --gt data/areas/lviv_old_town/flight/gt.csv
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.io import imwrite  # noqa: E402
from vo.synthetic import make_flight  # noqa: E402
from vo.tiles import PROVIDERS, fetch_area, fetch_box, max_available_zoom  # noqa: E402
from vo.wayback import Release, Version, capture_info, releases, versions_at  # noqa: E402

AREAS = Path(__file__).resolve().parents[1] / "configs" / "areas.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", help=f"named area from {AREAS.name}")
    ap.add_argument("--name", help="output name for a custom --center")
    ap.add_argument("--center", help="lat,lon in degrees (overrides the area)")
    ap.add_argument("--size-km", help="WxH on the ground, e.g. 2.8x1.4")
    ap.add_argument("--zoom", type=int, help="map zoom level (17 ~ 0.75 m/px in Ukraine)")
    ap.add_argument("--detail", type=int, default=2, help="frames come from zoom + detail (2 -> scale 0.25)")
    ap.add_argument("--frames-zoom", type=int, help="cap the zoom the frames are cut from (a control for archived "
                                                     "imagery that only exists at a coarser zoom); adds _z<N> to the tag")
    ap.add_argument("--provider", default="esri", choices=sorted(PROVIDERS), help="imagery of the map")
    ap.add_argument("--frames-from", help="imagery of the frames if different from the map: wayback:<release> "
                                          "or wayback:oldest (see scripts/list_wayback.py)")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--frame-size", default="640x640")
    ap.add_argument("--turns", type=float, default=1.15)
    ap.add_argument("--tilt-std", type=float, default=0.0, help="random perspective (~1e-5 .. 1e-4)")
    ap.add_argument("--degrade", default="none", choices=["none", "easy", "screen"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="data/cache/tiles")
    ap.add_argument("--out", help="output directory (default data/areas/<name>)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    area = {}
    if a.area:
        areas = yaml.safe_load(AREAS.read_text(encoding="utf-8"))
        if a.area not in areas:
            ap.error(f"unknown area {a.area!r}; known: {', '.join(areas)}")
        area = areas[a.area]
    lat, lon = (float(v) for v in a.center.split(",")) if a.center else area.get("center", (None, None))
    if lat is None:
        ap.error("give --area or --center")
    size_km = tuple(float(v) for v in a.size_km.split("x")) if a.size_km else tuple(area.get("size_km", (2.8, 1.4)))
    zoom = a.zoom or area.get("zoom", 17)
    name = a.name or a.area or f"area_{lat:.4f}_{lon:.4f}"
    out = Path(a.out or f"data/areas/{name}")
    out.mkdir(parents=True, exist_ok=True)

    # 1. map at `zoom`
    m, geo = fetch_area(lat, lon, (1000 * size_km[0], 1000 * size_km[1]), zoom, a.provider, a.cache)
    imwrite(out / "map.jpg", m, [1, 95])
    geo.save(out / "geo.json")
    logging.info("map %dx%d px, %.2f m/px -> %s", geo.width, geo.height, geo.meters_per_px(), out / "map.jpg")

    # frames: same imagery as the map, or an archived version of the same place
    src, tag = a.provider, ""
    if a.frames_from:
        v = resolve_version(a.frames_from, lat, lon, zoom)
        src, tag = v.provider, f"_{v.tag}"
        logging.info("frames from %s: captured %s by %s (release %s)", src, v.capture_date or "?",
                     v.sensor or "?", v.release.date)
    flight, detail = out / f"flight{tag}", out / f"detail{tag}.jpg"

    # 2. exactly the same ground box at the most detailed zoom that really has imagery here (<= zoom + detail)
    zd = max_available_zoom(geo, min(zoom + a.detail, a.frames_zoom or 99), a.cache, provider=src)
    if a.frames_zoom:
        tag += f"_z{zd}"
        flight, detail = out / f"flight{tag}", out / f"detail{tag}.jpg"
    if zd < zoom + a.detail:
        logging.warning("frames come from zoom %d instead of %d (upsampled x%d, less fine detail)",
                        zd, zoom + a.detail, 2 ** (zoom + a.detail - zd))
    k = 2 ** (zd - zoom)
    d, _ = fetch_box(geo.x0 * k, geo.y0 * k, geo.width * k, geo.height * k, zd, src, a.cache)
    imwrite(detail, d, [1, 92])

    # 3. flight over the detailed image; frame -> map scale stays 2^-detail whatever zoom was available
    fs = tuple(int(v) for v in a.frame_size.split("x"))
    make_flight(flight, a.frames, fs, None, a.turns, a.tilt_std, 3.0, a.seed,
                detail, k / 2 ** a.detail, a.degrade)
    if a.frames_from:
        off = coregistration_offset(m, cv2.resize(d, (geo.width, geo.height), interpolation=cv2.INTER_AREA))
        logging.info("archived vs. map imagery: offset %.1f px (%.1f m) -> floor of the GT accuracy",
                     off, off * geo.meters_per_px())
        (flight / "capture.json").write_text(json.dumps(
            {"provider": src, "release_date": v.release.date, "capture_date": v.capture_date, "sensor": v.sensor,
             "resolution_m": v.resolution_m, "detail_zoom": zd, "coreg_offset_px": off,
             "coreg_offset_m": off * geo.meters_per_px()}, indent=2), encoding="utf-8")

    # 4. GT: detail px -> map px (pixel centres: u = (x + 0.5) / k - 0.5), plus lon/lat
    gt = pd.read_csv(flight / "gt.csv")
    gt["x"] = (gt["x"] + 0.5) / k - 0.5
    gt["y"] = (gt["y"] + 0.5) / k - 0.5
    gt["scale"] = gt["scale"] / k
    gt["lon"], gt["lat"] = geo.to_lonlat(gt["x"], gt["y"])
    gt.to_csv(flight / "gt.csv", index=False)
    logging.info("%d frames + GT -> %s", len(gt), flight)
    print(f"\nnext:\n  python scripts/localize.py --map {out / 'map.jpg'} --frames {flight / 'frames'} "
          f"--gt {flight / 'gt.csv'} --out experiments/results/area_{name}{tag}")


def coregistration_offset(map_bgr: np.ndarray, other_bgr: np.ndarray) -> float:
    """Median shift (map px) between two images of the same box, from SIFT matches + a robust similarity.

    Archived and current imagery are both georeferenced, but not perfectly: this is the error floor of the GT
    when the frames come from another epoch.
    """
    sift = cv2.SIFT_create(8000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(other_bgr, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        return float("nan")
    pairs = [m for m, n in cv2.BFMatcher().knnMatch(d2, d1, k=2) if m.distance < 0.75 * n.distance]
    if len(pairs) < 10:
        return float("nan")
    p2 = np.float32([k2[m.queryIdx].pt for m in pairs])
    p1 = np.float32([k1[m.trainIdx].pt for m in pairs])
    _, inl = cv2.estimateAffinePartial2D(p2, p1, method=cv2.RANSAC, ransacReprojThreshold=4)
    ok = inl.ravel() == 1 if inl is not None else np.zeros(len(p1), bool)
    return float(np.linalg.norm(np.median(p1[ok] - p2[ok], axis=0))) if ok.sum() >= 10 else float("nan")


def resolve_version(spec: str, lat: float, lon: float, zoom: int) -> Version:
    """'wayback:oldest' -> the oldest distinct imagery at the point; 'wayback:<release>' -> that release."""
    name, _, rid = spec.partition(":")
    if name != "wayback" or not rid:
        raise SystemExit(f"--frames-from expects wayback:<release> or wayback:oldest, got {spec!r}")
    if rid == "oldest":
        return versions_at(lat, lon, zoom)[-1]
    rel = next((r for r in releases() if r.id == rid), None) or Release(rid, "?", "")
    v = Version(rel)
    if rel.metadata_url:
        v.capture_date, v.sensor, v.resolution_m = capture_info(rel, lat, lon)
    return v


if __name__ == "__main__":
    main()
