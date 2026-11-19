"""Build a localisation benchmark over ANY area: satellite map from public tiles + synthetic flight with exact GT.

The map is downloaded at `zoom`; the same ground box is downloaded again at `zoom + detail`, and the UAV frames are
rendered from that detailed image. So frames carry real fine detail that is absent from the map (as in the task
crops, frame -> map scale 2^-detail = 0.25), instead of being upsampled pieces of the map itself.
Rural areas often have no imagery at zoom 19; then the most detailed available zoom is used and upsampled.

Writes to --out (default data/areas/<name>):
    map.jpg          satellite map (the input for localisation)
    geo.json         georeference: map pixel <-> lon/lat (vo.tiles.GeoRef)
    detail.jpg       the same box at zoom + detail (frames are cut from it)
    flight/frames/   UAV frames
    flight/gt.csv    ground truth in MAP pixels (+ lat, lon)

Examples:
    python scripts/make_area.py --area lviv_old_town
    python scripts/make_area.py --name my_village --center 49.55,25.59 --size-km 2x1.5 --degrade screen
    python scripts/localize.py --map data/areas/lviv_old_town/map.jpg \
        --frames data/areas/lviv_old_town/flight/frames --gt data/areas/lviv_old_town/flight/gt.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.io import imwrite  # noqa: E402
from vo.synthetic import make_flight  # noqa: E402
from vo.tiles import PROVIDERS, fetch_area, fetch_box, max_available_zoom  # noqa: E402

AREAS = Path(__file__).resolve().parents[1] / "configs" / "areas.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", help=f"named area from {AREAS.name}")
    ap.add_argument("--name", help="output name for a custom --center")
    ap.add_argument("--center", help="lat,lon in degrees (overrides the area)")
    ap.add_argument("--size-km", help="WxH on the ground, e.g. 2.8x1.4")
    ap.add_argument("--zoom", type=int, help="map zoom level (17 ~ 0.75 m/px in Ukraine)")
    ap.add_argument("--detail", type=int, default=2, help="frames come from zoom + detail (2 -> scale 0.25)")
    ap.add_argument("--provider", default="esri", choices=sorted(PROVIDERS))
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

    # 2. exactly the same ground box at the most detailed zoom that really has imagery here (<= zoom + detail)
    zd = max_available_zoom(geo, zoom + a.detail, a.cache)
    if zd < zoom + a.detail:
        logging.warning("frames come from zoom %d instead of %d (upsampled x%d, less fine detail)",
                        zd, zoom + a.detail, 2 ** (zoom + a.detail - zd))
    k = 2 ** (zd - zoom)
    d, _ = fetch_box(geo.x0 * k, geo.y0 * k, geo.width * k, geo.height * k, zd, a.provider, a.cache)
    imwrite(out / "detail.jpg", d, [1, 92])

    # 3. flight over the detailed image; frame -> map scale stays 2^-detail whatever zoom was available
    fs = tuple(int(v) for v in a.frame_size.split("x"))
    make_flight(out / "flight", a.frames, fs, None, a.turns, a.tilt_std, 3.0, a.seed,
                out / "detail.jpg", k / 2 ** a.detail, a.degrade)

    # 4. GT: detail px -> map px (pixel centres: u = (x + 0.5) / k - 0.5), plus lon/lat
    gt = pd.read_csv(out / "flight" / "gt.csv")
    gt["x"] = (gt["x"] + 0.5) / k - 0.5
    gt["y"] = (gt["y"] + 0.5) / k - 0.5
    gt["scale"] = gt["scale"] / k
    gt["lon"], gt["lat"] = geo.to_lonlat(gt["x"], gt["y"])
    gt.to_csv(out / "flight" / "gt.csv", index=False)
    logging.info("%d frames + GT -> %s", len(gt), out / "flight")
    print(f"\nnext:\n  python scripts/localize.py --map {out / 'map.jpg'} --frames {out / 'flight' / 'frames'} "
          f"--gt {out / 'flight' / 'gt.csv'} --out experiments/results/area_{name}")


if __name__ == "__main__":
    main()
