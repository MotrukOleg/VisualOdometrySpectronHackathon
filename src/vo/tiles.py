"""Satellite map of any area from public XYZ tile servers, with Web Mercator georeferencing.

A map is a rectangle of "global pixels" at a fixed zoom level z: the whole world at zoom z is a square of
256 * 2^z pixels (Web Mercator, EPSG:3857), x grows east, y grows south. `GeoRef` remembers where the map's
top-left corner lies in global pixels, so any map pixel converts to lon/lat and back exactly.

Tiles are cached on disk (`<cache>/<provider>/<z>/<x>/<y>.jpg`): a repeated run, or a larger area that
overlaps a previous one, does not download them again.
"""
from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

TILE = 256
EARTH_CIRCUMFERENCE_M = 40_075_016.686
PROVIDERS = {
    "esri": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        "max_zoom": 19,
    },
}
USER_AGENT = "VisualOdometrySpectronHackathon/0.1 (research)"


def lonlat_to_px(lon, lat, z: int):
    """Lon/lat (degrees) -> global pixel coordinates at zoom z (floats, vectorised)."""
    n = TILE * 2.0 ** z
    lat_r = np.radians(lat)
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(lat_r)) / np.pi) / 2.0 * n
    return x, y


def px_to_lonlat(x, y, z: int):
    """Global pixel coordinates at zoom z -> lon/lat (degrees)."""
    n = TILE * 2.0 ** z
    lon = np.asarray(x) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * np.asarray(y) / n))))
    return lon, lat


def meters_per_px(lat: float, z: int) -> float:
    """Ground resolution of a zoom-z pixel at the given latitude."""
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (TILE * 2 ** z)


@dataclass
class GeoRef:
    """Map pixel (u, v) <-> global pixel (x0 + u, y0 + v) at `zoom` <-> lon/lat."""
    zoom: int
    x0: int
    y0: int
    width: int
    height: int
    provider: str = "esri"
    attribution: str = ""

    def to_lonlat(self, u, v):
        return px_to_lonlat(np.asarray(u, float) + self.x0, np.asarray(v, float) + self.y0, self.zoom)

    def to_px(self, lon, lat):
        x, y = lonlat_to_px(lon, lat, self.zoom)
        return x - self.x0, y - self.y0

    def meters_per_px(self) -> float:
        _, lat = self.to_lonlat(self.width / 2, self.height / 2)
        return meters_per_px(float(lat), self.zoom)

    def save(self, path: str | Path) -> None:
        d = asdict(self)
        lon0, lat0 = self.to_lonlat(0, 0)
        lon1, lat1 = self.to_lonlat(self.width, self.height)
        d["bbox_lonlat"] = [float(lon0), float(lat1), float(lon1), float(lat0)]  # west, south, east, north
        d["meters_per_px"] = self.meters_per_px()
        Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GeoRef":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def _download(url: str, retries: int = 3) -> bytes | None:
    """Raw tile bytes; None if the server has no tile there (404)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(1.0 + attempt)
    return None


def is_placeholder(tile: np.ndarray) -> bool:
    """Esri answers 200 with a flat light-grey "Map data not yet available" tile where it has no imagery
    (rural areas at zoom 19). Real flat tiles (water, shadows) are dark, so brightness separates them."""
    g = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return 195 < g.mean() < 215 and g.std() < 8


def fetch_tile(provider: str, z: int, x: int, y: int, cache_dir: str | Path) -> np.ndarray | None:
    """One 256x256 BGR tile, from the disk cache or the server; None if there is no imagery."""
    path = Path(cache_dir) / provider / str(z) / str(x) / f"{y}.jpg"
    if path.exists():
        data = path.read_bytes()
    else:
        data = _download(PROVIDERS[provider]["url"].format(z=z, x=x, y=y))
        if data is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    tile = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if tile is None or is_placeholder(tile) else tile


def max_available_zoom(geo: GeoRef, z_max: int, cache_dir: str | Path = "data/cache/tiles", grid: int = 4) -> int:
    """Highest zoom in [geo.zoom, z_max] with real imagery at a grid x grid set of points over the map box."""
    us = (np.arange(grid) + 0.5) / grid * geo.width
    vs = (np.arange(grid) + 0.5) / grid * geo.height
    for z in range(min(z_max, PROVIDERS[geo.provider]["max_zoom"]), geo.zoom, -1):
        k = 2 ** (z - geo.zoom)
        if all(fetch_tile(geo.provider, z, int((geo.x0 + u) * k) // TILE, int((geo.y0 + v) * k) // TILE, cache_dir)
               is not None for u in us for v in vs):
            return z
        log.info("%s has no imagery at zoom %d here", geo.provider, z)
    return geo.zoom


def fetch_box(x0: int, y0: int, width: int, height: int, zoom: int, provider: str = "esri",
              cache_dir: str | Path = "data/cache/tiles", workers: int = 8) -> tuple[np.ndarray, GeoRef]:
    """Stitch the global-pixel box [x0, x0 + width) x [y0, y0 + height) at `zoom` into one BGR image."""
    if zoom > PROVIDERS[provider]["max_zoom"]:
        raise ValueError(f"{provider} has no tiles above zoom {PROVIDERS[provider]['max_zoom']}")
    tx0, ty0 = x0 // TILE, y0 // TILE
    tx1, ty1 = (x0 + width - 1) // TILE, (y0 + height - 1) // TILE
    jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    canvas = np.full(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 3), 127, np.uint8)
    missing = 0
    with ThreadPoolExecutor(workers) as pool:
        tiles = pool.map(lambda t: fetch_tile(provider, zoom, t[0], t[1], cache_dir), jobs)
        for (tx, ty), tile in tqdm(zip(jobs, tiles), total=len(jobs), desc=f"tiles z{zoom}", unit="tile"):
            if tile is None:
                missing += 1
                continue
            r, c = (ty - ty0) * TILE, (tx - tx0) * TILE
            canvas[r:r + TILE, c:c + TILE] = cv2.resize(tile, (TILE, TILE)) if tile.shape[:2] != (TILE, TILE) else tile
    if missing:
        log.warning("%d/%d tiles are missing at zoom %d (filled with grey)", missing, len(jobs), zoom)
    ox, oy = x0 - tx0 * TILE, y0 - ty0 * TILE
    img = canvas[oy:oy + height, ox:ox + width].copy()
    geo = GeoRef(zoom, x0, y0, width, height, provider, PROVIDERS[provider]["attribution"])
    return img, geo


def fetch_area(lat: float, lon: float, size_m: tuple[float, float], zoom: int, provider: str = "esri",
               cache_dir: str | Path = "data/cache/tiles", workers: int = 8) -> tuple[np.ndarray, GeoRef]:
    """Map centred at (lat, lon) covering size_m = (width, height) metres on the ground."""
    res = meters_per_px(lat, zoom)
    width, height = int(round(size_m[0] / res)), int(round(size_m[1] / res))
    cx, cy = lonlat_to_px(lon, lat, zoom)
    x0, y0 = int(round(float(cx) - width / 2)), int(round(float(cy) - height / 2))
    return fetch_box(x0, y0, width, height, zoom, provider, cache_dir, workers)
