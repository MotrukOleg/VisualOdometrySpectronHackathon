"""GPS spoofing detection: cross-check the GPS track against visual map fixes.

A spoofer drags the receiver away slowly (a jump would trip the autopilot's own checks). Visual fixes do not
depend on GPS, so the distance between the reported GPS position and the visual fix stays at the matcher's error
(a few metres) until the spoofing starts and then grows with the drag. An alarm is raised after `confirm`
consecutive visual fixes disagree with GPS by more than `threshold_m`: one wrong fix cannot trigger it. After the
alarm, the spoofing offset (GPS - visual) is measured at every visual fix and subtracted from GPS, interpolated
between fixes: the drag is slow, so GPS minus the offset follows the real motion even where vision is blind.

All positions are local east/north metres, NaN where a frame has no visual fix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def to_enu(lon, lat, lon0: float, lat0: float) -> np.ndarray:
    """(N, 2) east/north metres relative to (lon0, lat0); equirectangular, fine for a few km."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    e = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    n = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.stack([e, n], 1)


def spoof(track: np.ndarray, start: int, rate_m: float, max_m: float, bearing_deg: float = 90.0) -> np.ndarray:
    """GPS track dragged from frame `start` by `rate_m` metres per frame towards `bearing_deg`, up to `max_m`."""
    k = np.arange(len(track))
    drag = np.clip((k - start + 1) * rate_m, 0, max_m)
    b = np.radians(bearing_deg)
    return track + drag[:, None] * np.array([np.sin(b), np.cos(b)])


@dataclass
class Detection:
    residual_m: np.ndarray  # |GPS - visual| per frame, NaN without a visual fix
    alarm: int | None  # frame index of the alarm, None if never raised
    position: np.ndarray  # (N, 2) navigation output: GPS before the alarm, GPS minus the offset after


def detect(gps: np.ndarray, visual: np.ndarray, threshold_m: float = 15.0, confirm: int = 3) -> Detection:
    residual = np.linalg.norm(gps - visual, axis=1)
    alarm, run = None, 0
    for k in np.flatnonzero(np.isfinite(residual)):
        run = run + 1 if residual[k] > threshold_m else 0
        if run >= confirm:
            alarm = int(k)
            break
    position = gps.copy()
    if alarm is not None:
        fixes = np.flatnonzero(np.isfinite(visual[:, 0]))
        offset = gps[fixes] - visual[fixes]
        after = np.arange(alarm, len(gps))
        for d in range(2):  # interpolate the offset between fixes, hold the last one
            position[after, d] = gps[after, d] - np.interp(after, fixes, offset[:, d])
    return Detection(residual, alarm, position)
