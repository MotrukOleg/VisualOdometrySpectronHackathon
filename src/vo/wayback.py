"""Esri World Imagery Wayback: archived versions of the satellite basemap (2014 -> today).

A Wayback *release* is a full snapshot of World Imagery on a date; most releases repeat the imagery of the
previous one for any given place. `versions_at` walks the per-tile change history ("tilemap" service) from
the newest release back and keeps only releases whose imagery really differs there. `capture_info` asks the
release's metadata service when and by which sensor the imagery at a point was actually captured
(a 2014 release can show a 2010 image).

Tiles of a release are fetched through vo.tiles with provider "wayback:<release>".
"""
from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .tiles import TILE, _download, fetch_tile, lonlat_to_px

log = logging.getLogger(__name__)

CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
TILEMAP_URL = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tilemap/"
               "{release}/{z}/{y}/{x}")


@dataclass
class Release:
    id: str
    date: str  # release date YYYY-MM-DD (not the capture date)
    metadata_url: str


@dataclass
class Version:
    release: Release
    capture_date: str = ""  # YYYY-MM-DD from the metadata service, "" if unknown
    sensor: str = ""
    resolution_m: float | None = None

    @property
    def provider(self) -> str:
        return f"wayback:{self.release.id}"

    @property
    def tag(self) -> str:
        """Short name for output folders, e.g. 'wb2010' (capture year) or 'r2014-02-20' (release)."""
        return f"wb{self.capture_date[:4]}" if self.capture_date else f"r{self.release.date}"


def releases(cache_dir: str | Path = "data/cache/wayback") -> list[Release]:
    """All releases, oldest first. The list is cached; delete the cache file to pick up new releases."""
    path = Path(cache_dir) / "config.json"
    if not path.exists():
        data = _download(CONFIG_URL)
        if data is None:
            raise RuntimeError("Wayback config not found")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    out = [Release(k, v["itemTitle"].split("Wayback ")[-1].rstrip(")"), v["metadataLayerUrl"]) for k, v in cfg.items()]
    return sorted(out, key=lambda r: r.date)


def _tile_hash(release: str, z: int, x: int, y: int, cache_dir) -> str | None:
    tile = fetch_tile(f"wayback:{release}", z, x, y, cache_dir)
    return None if tile is None else hashlib.md5(tile.tobytes()).hexdigest()


def versions_at(lat: float, lon: float, zoom: int = 17, cache_dir: str | Path = "data/cache",
                with_capture: bool = True) -> list[Version]:
    """Releases with distinct imagery at (lat, lon), newest first."""
    rel = releases(Path(cache_dir) / "wayback")
    by_id = {r.id: r for r in rel}
    index = {r.id: k for k, r in enumerate(rel)}
    x, y = lonlat_to_px(lon, lat, zoom)
    tx, ty = int(x // TILE), int(y // TILE)
    out, seen, i = [], set(), len(rel) - 1
    while i >= 0:
        j = json.loads(_download(TILEMAP_URL.format(release=rel[i].id, z=zoom, y=ty, x=tx)) or b"{}")
        sel = str(j.get("select", [rel[i].id])[0])
        if j.get("data", [0])[0] and sel in by_id:
            h = _tile_hash(sel, zoom, tx, ty, Path(cache_dir) / "tiles")
            if h and h not in seen:
                seen.add(h)
                out.append(Version(by_id[sel]))
        i = min(i, index.get(sel, i)) - 1  # jump to the release before the last change of this tile
    if with_capture:
        for v in out:
            v.capture_date, v.sensor, v.resolution_m = capture_info(v.release, lat, lon)
    return out


def capture_info(release: Release, lat: float, lon: float) -> tuple[str, str, float | None]:
    """(capture date YYYY-MM-DD, sensor, source resolution in m) of the finest imagery at the point."""
    q = urllib.parse.urlencode({
        "f": "json", "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "sr": 4326,
        "layers": "all", "tolerance": 0, "returnGeometry": "false",
        "mapExtent": f"{lon - 0.01},{lat - 0.01},{lon + 0.01},{lat + 0.01}", "imageDisplay": "600,600,96"})
    try:
        data = _download(f"{release.metadata_url}/identify?{q}")
        res = sorted(json.loads(data or b"{}").get("results", []), key=lambda r: r["layerId"])
    except Exception as e:  # metadata is a nice-to-have, never fail the data build on it
        log.warning("no capture metadata for release %s: %s", release.id, e)
        return "", "", None
    if not res:
        return "", "", None
    a = res[0]["attributes"]  # lowest layer id = finest resolution
    d = str(a.get("SRC_DATE", ""))
    date = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else ""
    try:
        resolution = float(a.get("SRC_RES"))
    except (TypeError, ValueError):
        resolution = None
    return date, str(a.get("SRC_DESC", "")), resolution
