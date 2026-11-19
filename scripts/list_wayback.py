"""List archived satellite imagery versions (Esri Wayback) available for an area.

Only releases whose imagery really differs at the area centre are shown, with the capture date and sensor
from Esri's metadata. Use the printed provider with scripts/make_area.py --frames-from.

Examples:
    python scripts/list_wayback.py --area poltava_fields
    python scripts/list_wayback.py --center 49.55,25.59
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vo.wayback import versions_at  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", help="named area from configs/areas.yaml")
    ap.add_argument("--center", help="lat,lon in degrees")
    ap.add_argument("--zoom", type=int, default=17)
    a = ap.parse_args()
    if a.center:
        lat, lon = (float(v) for v in a.center.split(","))
    elif a.area:
        lat, lon = yaml.safe_load((ROOT / "configs" / "areas.yaml").read_text(encoding="utf-8"))[a.area]["center"]
    else:
        ap.error("give --area or --center")
    print(f"{'captured':10s}  {'sensor':10s}  {'res, m':>6s}  {'release':10s}  provider")
    for v in versions_at(lat, lon, a.zoom):
        res = f"{v.resolution_m:.2f}" if v.resolution_m else "?"
        print(f"{v.capture_date or '?':10s}  {v.sensor or '?':10s}  {res:>6s}  {v.release.date}  {v.provider}")


if __name__ == "__main__":
    main()
