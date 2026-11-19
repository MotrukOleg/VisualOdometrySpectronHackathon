import numpy as np
import pytest

from vo import tiles
from vo.tiles import GeoRef, fetch_box, lonlat_to_px, meters_per_px, px_to_lonlat


def test_lonlat_px_roundtrip():
    lon, lat = np.array([24.0316, -73.9857, 151.2093]), np.array([49.8419, 40.7484, -33.8688])
    for z in (0, 12, 19):
        x, y = lonlat_to_px(lon, lat, z)
        lon2, lat2 = px_to_lonlat(x, y, z)
        assert np.allclose(lon2, lon, atol=1e-9) and np.allclose(lat2, lat, atol=1e-9)


def test_known_values():
    x, y = lonlat_to_px(0.0, 0.0, 1)  # equator / Greenwich is the centre of the world square
    assert x == pytest.approx(256) and y == pytest.approx(256)
    assert meters_per_px(0.0, 0) == pytest.approx(156543.03, rel=1e-6)
    assert meters_per_px(60.0, 0) == pytest.approx(156543.03 / 2, rel=1e-6)


def test_georef_save_load(tmp_path):
    g = GeoRef(17, 37_000_000 // 1000, 22_000_000 // 1000, 3600, 1800, "esri", "x")
    g.save(tmp_path / "geo.json")
    g2 = GeoRef.load(tmp_path / "geo.json")
    assert g2 == g
    lon, lat = g.to_lonlat(100.0, 200.0)
    u, v = g2.to_px(lon, lat)
    assert u == pytest.approx(100.0) and v == pytest.approx(200.0)


def test_placeholder_detection():
    rng = np.random.default_rng(0)
    placeholder = np.full((256, 256, 3), 205, np.uint8)
    placeholder[100:110, 50:150] = 240  # "Map data not yet available" text
    water = np.clip(rng.normal(10, 4, (256, 256, 3)), 0, 255).astype(np.uint8)  # flat but dark
    city = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    assert tiles.is_placeholder(placeholder)
    assert not tiles.is_placeholder(water) and not tiles.is_placeholder(city)


def test_fetch_box_stitches_and_crops(monkeypatch, tmp_path):
    # fake server: every tile is filled with its own (x, y) so placement can be checked
    def fake_tile(provider, z, x, y, cache_dir):
        t = np.zeros((256, 256, 3), np.uint8)
        t[..., 0], t[..., 1] = x % 256, y % 256
        return t

    monkeypatch.setattr(tiles, "fetch_tile", fake_tile)
    img, geo = fetch_box(300, 600, 500, 300, 5, cache_dir=tmp_path)
    assert img.shape == (300, 500, 3)
    assert (geo.x0, geo.y0, geo.width, geo.height) == (300, 600, 500, 300)
    # global pixel (300, 600) lies in tile (1, 2); (799, 899) in tile (3, 3)
    assert tuple(img[0, 0, :2]) == (1, 2)
    assert tuple(img[-1, -1, :2]) == (3, 3)
    # global x = 512 is the first column of tile 2
    assert img[0, 512 - 300 - 1, 0] == 1 and img[0, 512 - 300, 0] == 2
