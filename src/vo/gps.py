"""Localisation result in world coordinates: lat/lon per frame, GeoJSON and an interactive web map.

Needs a georeferenced map (`geo.json` written by scripts/make_area.py, see vo.tiles.GeoRef). Map pixels
are converted exactly through Web Mercator, so the GPS track is as accurate as the pixel track.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .tiles import GeoRef

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, float)) for v in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def add_lonlat(traj: pd.DataFrame, geo: GeoRef) -> pd.DataFrame:
    """Copy of `traj` with lon/lat of the frame centre and of the 4 footprint corners."""
    out = traj.copy()
    out["lon"], out["lat"] = geo.to_lonlat(traj["x"], traj["y"])
    for k in range(4):
        out[f"c{k}lon"], out[f"c{k}lat"] = geo.to_lonlat(traj[f"c{k}x"], traj[f"c{k}y"])
    return out


def gps_errors(traj: pd.DataFrame, gt: pd.DataFrame) -> pd.Series:
    """Per-frame error in metres against a GT table with name, lon, lat (NaN where either is missing)."""
    g = gt.set_index("name")[["lon", "lat"]].reindex(traj["name"].astype(str))
    ok = traj["status"].isin(["map", "interp"]).to_numpy()
    err = haversine_m(traj["lon"], traj["lat"], g["lon"].to_numpy(), g["lat"].to_numpy())
    return pd.Series(np.where(ok, err, np.nan), index=traj.index, name="err_m")


def gps_metrics(err_m: pd.Series, thresholds=(1, 5, 25)) -> dict:
    e = err_m.to_numpy(float)
    has = np.isfinite(e)
    out = {"gps_err_median_m": float(np.median(e[has])) if has.any() else float("nan"),
           "gps_err_max_m": float(e[has].max()) if has.any() else float("nan")}
    for t in thresholds:
        out[f"gps_success@{t}m"] = float((np.nan_to_num(e, nan=np.inf) < t).mean())
    return out


def _rounded(v) -> list:
    return [round(float(x), 7) for x in v]


def to_geojson(traj: pd.DataFrame, gt: pd.DataFrame | None = None, footprints: bool = True) -> dict:
    """FeatureCollection: route line, one point per frame (with status/heading/inliers), footprints, GT line."""
    feats = []
    est = traj[traj["status"].isin(["map", "interp"])]
    feats.append({"type": "Feature", "properties": {"kind": "route"},
                  "geometry": {"type": "LineString", "coordinates": [_rounded(p) for p in est[["lon", "lat"]].values]}})
    for _, r in traj.iterrows():
        if r["status"] == "lost":
            continue
        props = {"kind": "frame", "frame": int(r["frame"]), "name": str(r["name"]), "status": r["status"],
                 "heading_deg": round(float(r["heading_deg"]), 1), "n_inliers": int(r["n_inliers"])}
        if "err_m" in r and np.isfinite(r["err_m"]):
            props["err_m"] = round(float(r["err_m"]), 2)
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point", "coordinates": _rounded([r["lon"], r["lat"]])}})
        if footprints and r["status"] == "map":
            ring = [_rounded([r[f"c{k}lon"], r[f"c{k}lat"]]) for k in (0, 1, 2, 3, 0)]
            feats.append({"type": "Feature", "properties": {"kind": "footprint", "frame": int(r["frame"])},
                          "geometry": {"type": "Polygon", "coordinates": [ring]}})
    if gt is not None and {"lon", "lat"} <= set(gt.columns):
        feats.append({"type": "Feature", "properties": {"kind": "ground_truth"},
                      "geometry": {"type": "LineString", "coordinates": [_rounded(p) for p in gt[["lon", "lat"]].values]}})
    return {"type": "FeatureCollection", "features": feats}


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map { height: 100%; margin: 0; background: #111; }
  .info { background: rgba(20,20,20,.85); color: #eee; padding: 8px 12px; border-radius: 6px;
          font: 13px/1.4 system-ui, sans-serif; }
  .info b { color: #fff; }
</style>
</head>
<body>
<div id="map"></div>
<script>
const data = __DATA__;
const summary = __SUMMARY__;
const map = L.map("map", {preferCanvas: true});
const sat = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {maxZoom: 20, maxNativeZoom: 19, attribution: "Imagery: Esri, Maxar, Earthstar Geographics"}).addTo(map);
// OSM tiles refuse pages opened from file:// (no Referer -> 403), Esri serves them
const streets = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
  {maxZoom: 20, maxNativeZoom: 19, attribution: "Esri, HERE, Garmin, &copy; OpenStreetMap contributors"});
const labels = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
  {maxZoom: 20, maxNativeZoom: 19});
const colour = {map: "#2ecc71", interp: "#f39c12"};
const layers = {footprints: L.layerGroup(), frames: L.layerGroup(), gt: L.layerGroup(), route: L.layerGroup()};
let frames = [];
for (const f of data.features) {
  const p = f.properties;
  if (p.kind === "route") {
    L.geoJSON(f, {style: {color: "#00e5ff", weight: 3}}).addTo(layers.route);
  } else if (p.kind === "ground_truth") {
    L.geoJSON(f, {style: {color: "#fff", weight: 2, dashArray: "6 6"}}).addTo(layers.gt);
  } else if (p.kind === "footprint") {
    L.geoJSON(f, {style: {color: "#2ecc71", weight: 1, fillOpacity: 0.05}}).addTo(layers.footprints);
  } else if (p.kind === "frame") {
    const [lon, lat] = f.geometry.coordinates;
    frames.push([lat, lon]);
    const err = p.err_m !== undefined ? `<br>error vs GT: <b>${p.err_m} m</b>` : "";
    L.circleMarker([lat, lon], {radius: 4, color: colour[p.status], fillOpacity: 1, weight: 1})
      .bindPopup(`<b>frame ${p.frame}</b> (${p.name})<br>status: ${p.status}<br>` +
                 `lat ${lat.toFixed(6)}, lon ${lon.toFixed(6)}<br>heading ${p.heading_deg}&deg;, ` +
                 `inliers ${p.n_inliers}${err}`)
      .addTo(layers.frames);
  }
}
Object.values(layers).forEach(l => l.addTo(map));
if (frames.length) {
  L.marker(frames[0]).bindTooltip("START", {permanent: true}).addTo(map);
  L.marker(frames[frames.length - 1]).bindTooltip("FINISH", {permanent: true}).addTo(map);
  map.fitBounds(L.latLngBounds(frames).pad(0.15));
}
L.control.layers({"Satellite": sat, "Streets": streets},
  {"Place names": labels, "Route": layers.route, "Frames": layers.frames, "Footprints": layers.footprints,
   "Ground truth": layers.gt},
  {collapsed: false}).addTo(map);
const info = L.control({position: "bottomleft"});
info.onAdd = () => {
  const d = L.DomUtil.create("div", "info");
  d.innerHTML = Object.entries(summary).map(([k, v]) => `${k}: <b>${v}</b>`).join("<br>");
  return d;
};
info.addTo(map);
L.control.scale({imperial: false}).addTo(map);
</script>
</body>
</html>
"""


def write_web_map(geojson: dict, path: str | Path, title: str, summary: dict) -> None:
    """Self-contained Leaflet page (data inlined): satellite / street base, route, frames with popups, GT."""
    html = (_HTML.replace("__TITLE__", title)
            .replace("__DATA__", json.dumps(geojson, separators=(",", ":")))
            .replace("__SUMMARY__", json.dumps(summary, ensure_ascii=False)))
    Path(path).write_text(html, encoding="utf-8")
