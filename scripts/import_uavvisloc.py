"""Import a REAL UAV flight from the UAV-VisLoc dataset as a localisation benchmark with GPS ground truth.

UAV-VisLoc (https://zenodo.org/records/19603534, academic use only) has 30 real flights: nadir UAV images
(already rotated north-up) with the GPS position of every image, plus satellite image chips (Google Earth) of the
flight area. Unlike data/areas/*, frames and map come from different sensors and dates, so the error in metres is
a real navigation error.

The chips are stitched back into one image, georeferenced by a linear fit of the chip centres from
<id>-satellite-tile-coordinates.csv, and resampled onto the Web Mercator grid of `--zoom`, so the result has the
same geo.json as make_area.py and works with localize.py / export_gps.py unchanged.

Writes to --out (default data/real/visloc_<id>):
    map.jpg          satellite map
    geo.json         georeference (vo.tiles.GeoRef)
    flight/frames/   UAV images (symlinks)
    flight/gt.csv    name, x, y (map px), lon, lat, height

Example:
    python scripts/import_uavvisloc.py --flight data/uavvisloc/24
    python scripts/localize.py --map data/real/visloc_24/map.jpg --frames data/real/visloc_24/flight/frames \
        --gt data/real/visloc_24/flight/gt.csv
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.io import imread, imwrite  # noqa: E402
from vo.tiles import GeoRef, lonlat_to_px, px_to_lonlat  # noqa: E402

ATTRIBUTION = "UAV-VisLoc dataset (academic use); satellite imagery: Google Earth"


def stitch_chips(flight: Path, fid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mosaic of the satellite chips + linear fits lon(col) and lat(row) in mosaic px."""
    coords = pd.read_csv(flight / f"{fid}-satellite-tile-coordinates.csv")
    chips = {}
    for p in (flight / "satellite-tile").glob("*.jpg"):
        r, c = map(int, re.findall(r"_(\d+)_(\d+)\.jpg$", p.name)[0])
        chips[(r, c)] = p
    size = imread(next(iter(chips.values()))).shape[0]
    h = max(r for r, _ in chips) + size
    w = max(c for _, c in chips) + size
    mosaic = np.zeros((h, w, 3), np.uint8)
    for (r, c), p in chips.items():
        tile = imread(p)
        mosaic[r:r + tile.shape[0], c:c + tile.shape[1]] = tile
    rc = coords["patchname"].str.extract(r"(\d+)_(\d+)").astype(float).to_numpy() + size / 2  # chip centres
    fit_lon = np.polyfit(rc[:, 1], coords["longitude"], 1)
    fit_lat = np.polyfit(rc[:, 0], coords["latitude"], 1)
    return mosaic, fit_lon, fit_lat


def to_mercator(mosaic: np.ndarray, fit_lon, fit_lat, zoom: int) -> tuple[np.ndarray, GeoRef]:
    """Resample the lon/lat-linear mosaic onto the Web Mercator pixel grid of `zoom`."""
    h, w = mosaic.shape[:2]
    lon_a, lon_b = np.polyval(fit_lon, [0, w])
    lat_a, lat_b = np.polyval(fit_lat, [0, h])
    xs, ys = lonlat_to_px(np.array([lon_a, lon_b]), np.array([lat_a, lat_b]), zoom)
    x0, y0 = int(np.ceil(min(xs))), int(np.ceil(min(ys)))
    geo = GeoRef(zoom, x0, y0, int(max(xs)) - x0, int(max(ys)) - y0, "uavvisloc", ATTRIBUTION)
    lon, _ = px_to_lonlat(x0 + np.arange(geo.width, dtype=float), np.full(geo.width, y0, float), zoom)
    _, lat = px_to_lonlat(np.full(geo.height, x0, float), y0 + np.arange(geo.height, dtype=float), zoom)
    col = (lon - fit_lon[1]) / fit_lon[0]
    row = (lat - fit_lat[1]) / fit_lat[0]
    mx, my = np.meshgrid(col.astype(np.float32), row.astype(np.float32))
    out = cv2.remap(mosaic, mx, my, cv2.INTER_LINEAR)
    return out, geo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", required=True, help="extracted flight folder, e.g. data/uavvisloc/24")
    ap.add_argument("--zoom", type=int, default=17, help="Web Mercator zoom of the map (17 ~ 1 m/px)")
    ap.add_argument("--out", help="output folder (default data/real/visloc_<id>)")
    a = ap.parse_args()

    flight = Path(a.flight)
    fid = flight.name
    out = Path(a.out or f"data/real/visloc_{fid}")
    (out / "flight" / "frames").mkdir(parents=True, exist_ok=True)

    mosaic, fit_lon, fit_lat = stitch_chips(flight, fid)
    # Area-average to roughly the target resolution first: remap alone would alias.
    m_lon = abs(fit_lon[0]) * 111_320 * np.cos(np.radians(np.polyval(fit_lat, mosaic.shape[0] / 2)))
    target = 156_543.03 * np.cos(np.radians(np.polyval(fit_lat, mosaic.shape[0] / 2))) / 2 ** a.zoom
    f = m_lon / target
    if f < 1:
        mosaic = cv2.resize(mosaic, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        fit_lon, fit_lat = fit_lon * [1 / f, 1], fit_lat * [1 / f, 1]
    map_img, geo = to_mercator(mosaic, fit_lon, fit_lat, a.zoom)
    imwrite(out / "map.jpg", map_img)
    geo.save(out / "geo.json")

    uav = pd.read_csv(flight / f"{fid}-UAV-coordinates.csv")
    files = {p.stem.lower(): p for p in (flight / "UAV-image").iterdir()}
    rows = []
    for _, r in uav.iterrows():
        src = files.get(Path(r["filename"]).stem.lower())
        if src is None:
            continue
        link = out / "flight" / "frames" / src.name
        if not link.exists():
            link.symlink_to(src.resolve())
        x, y = geo.to_px(r["lon"], r["lat"])
        rows.append({"name": src.stem, "x": float(x), "y": float(y), "lon": r["lon"], "lat": r["lat"],
                     "height": r["height"]})
    gt = pd.DataFrame(rows)
    gt.to_csv(out / "flight" / "gt.csv", index=False, float_format="%.7f")
    inside = ((gt.x >= 0) & (gt.x < geo.width) & (gt.y >= 0) & (gt.y < geo.height)).mean()
    print(f"{out}: map {geo.width}x{geo.height} px at {geo.meters_per_px():.2f} m/px, {len(gt)} frames, "
          f"{100 * inside:.0f} % of GT inside the map")


if __name__ == "__main__":
    main()
