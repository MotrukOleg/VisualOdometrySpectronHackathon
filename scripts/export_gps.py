"""Export a localisation run to GPS: lat/lon per frame, GeoJSON and an interactive web map.

Reads <run>/trajectory.csv (written by scripts/localize.py) and the map's geo.json (written by
scripts/make_area.py). If a GT CSV with lon/lat is given, errors are reported in metres.

Writes into the run directory:
    trajectory_gps.csv   trajectory + lat/lon of the centre and footprint corners (+ err_m)
    route.geojson        route, frames, footprints, GT (opens in geojson.io, QGIS, Google Earth via import)
    route_map.html       self-contained interactive map (satellite / OSM, click a frame for details)
    gps_metrics.json     error in metres (only with GT)

Example:
    python scripts/export_gps.py --run experiments/results/area_kyiv_dnipro \
        --geo data/areas/kyiv_dnipro/geo.json --gt data/areas/kyiv_dnipro/flight/gt.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.gps import add_lonlat, gps_errors, gps_metrics, to_geojson, write_web_map  # noqa: E402
from vo.schema import read_trajectory  # noqa: E402
from vo.tiles import GeoRef  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory with trajectory.csv")
    ap.add_argument("--geo", required=True, help="geo.json of the map the run was localised on")
    ap.add_argument("--gt", help="optional GT CSV with name, lon, lat")
    ap.add_argument("--no-footprints", action="store_true", help="omit frame outlines (smaller files)")
    a = ap.parse_args()

    run = Path(a.run)
    geo = GeoRef.load(a.geo)
    traj = add_lonlat(read_trajectory(run / "trajectory.csv"), geo)
    gt = pd.read_csv(a.gt) if a.gt else None

    summary = {"run": run.name, "frames": len(traj),
               "registered": int((traj["status"] == "map").sum()),
               "interpolated": int((traj["status"] == "interp").sum()),
               "lost": int((traj["status"] == "lost").sum()),
               "map resolution": f"{geo.meters_per_px():.2f} m/px"}
    if gt is not None:
        traj["err_m"] = gps_errors(traj, gt)
        metrics = gps_metrics(traj["err_m"])
        (run / "gps_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary["median error"] = f"{metrics['gps_err_median_m']:.2f} m"
        summary["within 5 m"] = f"{100 * metrics['gps_success@5m']:.0f} %"
        print(json.dumps(metrics, indent=2))

    traj.to_csv(run / "trajectory_gps.csv", index=False, float_format="%.7f")
    gj = to_geojson(traj, gt, footprints=not a.no_footprints)
    (run / "route.geojson").write_text(json.dumps(gj, separators=(",", ":")), encoding="utf-8")
    write_web_map(gj, run / "route_map.html", f"UAV route: {run.name}", summary)
    print(f"written: {run / 'trajectory_gps.csv'}, {run / 'route.geojson'}, {run / 'route_map.html'}")


if __name__ == "__main__":
    main()
