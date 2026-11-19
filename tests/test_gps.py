import numpy as np
import pandas as pd
import pytest

from vo.gps import add_lonlat, gps_errors, gps_metrics, haversine_m, to_geojson
from vo.tiles import GeoRef, lonlat_to_px, meters_per_px

LVIV = (24.0316, 49.8419)


def _geo():
    x, y = lonlat_to_px(*LVIV, 17)
    return GeoRef(17, int(x) - 1800, int(y) - 900, 3600, 1800)


def _traj(xy, status):
    rows = []
    for k, ((x, y), s) in enumerate(zip(xy, status)):
        r = {"frame": k, "name": f"f{k}", "x": x, "y": y, "heading_deg": 0.0, "n_inliers": 100, "status": s}
        for c, (dx, dy) in enumerate([(-80, -80), (80, -80), (80, 80), (-80, 80)]):
            r[f"c{c}x"], r[f"c{c}y"] = x + dx, y + dy
        rows.append(r)
    return pd.DataFrame(rows)


def test_haversine_known_distance():
    # one degree of latitude ~ 111.2 km
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)
    assert haversine_m(*LVIV, *LVIV) == 0


def test_pixel_offset_maps_to_metres():
    geo = _geo()
    t = add_lonlat(_traj([(1800, 900), (1900, 900)], ["map", "map"]), geo)
    d = haversine_m(t.lon[0], t.lat[0], t.lon[1], t.lat[1])
    # Web Mercator scale uses the equatorial radius, haversine the mean one: they differ by 0.11 %
    assert d == pytest.approx(100 * meters_per_px(LVIV[1], 17), rel=2e-3)
    assert t.lat[0] == pytest.approx(LVIV[1], abs=1e-4) and t.lon[0] == pytest.approx(LVIV[0], abs=1e-4)


def test_errors_and_geojson():
    geo = _geo()
    t = add_lonlat(_traj([(1800, 900), (1810, 900), (0, 0)], ["map", "interp", "lost"]), geo)
    gt = pd.DataFrame({"name": ["f0", "f1", "f2"]})
    gt["lon"], gt["lat"] = geo.to_lonlat(np.array([1800, 1800, 0]), np.array([900, 900, 0]))
    t["err_m"] = gps_errors(t, gt)
    m = meters_per_px(LVIV[1], 17)
    assert t.err_m[0] == pytest.approx(0, abs=1e-6)
    assert t.err_m[1] == pytest.approx(10 * m, rel=2e-3)
    assert np.isnan(t.err_m[2])  # lost frame has no estimate
    metrics = gps_metrics(t["err_m"])
    assert metrics["gps_success@25m"] == pytest.approx(2 / 3)  # lost counts as failure

    kinds = [f["properties"]["kind"] for f in to_geojson(t, gt)["features"]]
    assert kinds.count("frame") == 2 and kinds.count("footprint") == 1  # footprint only for 'map'
    assert "route" in kinds and "ground_truth" in kinds
