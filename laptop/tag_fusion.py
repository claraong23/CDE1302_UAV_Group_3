"""Final target-tag poses from many sightings.

Each time a target tag is seen, tag_mission.py records one Sighting (its arena
x/y plus how good the view was). At the end the sightings of one tag are
combined:

    median  ->  drop outliers  ->  weighted mean

Expected error of one sighting (metres):
    sigma = SIGMA0 + K_CENTRE * how far off the picture centre (0 = centre, 1 = corner;
                                the less centred of the target and the known tag(s))
                   + K_RANGE  * camera-to-target distance
                   + K_SPEED  * drone speed
                   + K_SEP    * distance flown since the last fix ('odom' sightings only)
    then / cos(viewing angle) (slanted views are worse), x0.7 with 2+ known tags.
    weight = 1 / sigma^2

A tag is CONFIRMED with >= CONFIRM_N kept sightings, from >= CONFIRM_VIEWS
viewpoints (drone positions at least VIEWPOINT_M apart; trajectories keep one
heading, so different places, not headings), and an uncertainty under CONFIRM_UNC.

Bench-tested in esp_camera/loc_test.py (4 Nov 2026): 0.4 cm and 1.1 cm error
on two targets.
"""
import math
from dataclasses import dataclass

import numpy as np

SIGMA0, K_SEP, K_RANGE, K_SPEED = 0.01, 0.05, 0.02, 0.10
K_CENTRE = 0.05            # a tag in the picture corner is ~5 cm worse than one in the middle
MAX_RANGE_M = 1.6          # sightings further than this from the lens are not used
                           # (at 1 m altitude every floor tag is >= 1 m away)
CONFIRM_N, CONFIRM_VIEWS, CONFIRM_UNC = 5, 2, 0.05
VIEWPOINT_M = 0.3          # drone positions this far apart count as different viewpoints
OUTLIER_MIN_M, OUTLIER_MAX_M = 0.05, 0.15


@dataclass
class Sighting:
    x: float
    y: float
    method: str            # "tag" (known tag in the same picture) or "odom"
    sep: float             # m to the nearest known tag used ("odom": m flown since the last fix)
    rng: float             # m from the lens to the target
    ang: float             # deg between the view ray and the tag's normal; nan if unknown
    nk: int                # known tags in the picture
    speed: float           # drone speed, m/s
    hdg: float             # drone heading, deg
    known: tuple = ()      # ids of the known tags in the same picture ("tag" method)
    pos: tuple = ()        # drone arena (x, y) when seen; () if unknown
    off: float = float("nan")  # 0 = picture centre .. 1 = corner (less centred of target / known tags)


def view_angle(det) -> float:
    """Degrees between the camera ray and the tag's normal (0 = seen face-on).
    nan when the detection has no rotation (the drone's own detections)."""
    if getattr(det, "R", None) is None or det.t is None:
        return float("nan")
    n = np.array(det.R, float).reshape(3, 3)[:, 2]
    ray = np.array(det.t, float) / max(det.range_m, 1e-9)
    return math.degrees(math.acos(min(1.0, abs(float(n @ ray)))))


def sigma_of(s: Sighting, k_sep: float = K_SEP) -> float:
    sig = SIGMA0 + K_RANGE * s.rng + K_SPEED * s.speed
    if not math.isnan(s.off):
        sig += K_CENTRE * s.off
    if s.method == "odom":                      # PX4 drift since the last known-tag fix
        sig += k_sep * s.sep
    if not math.isnan(s.ang):
        sig /= max(math.cos(math.radians(s.ang)), 0.3)
    if s.nk >= 2:
        sig *= 0.7
    return sig


def viewpoints(rows: list[Sighting]) -> int:
    """Distinct drone positions (>= VIEWPOINT_M apart) the sightings were taken from.
    Sightings without a position fall back to 30-degree heading bins."""
    spots = []
    for r in rows:
        if len(r.pos) == 2 and all(abs(v) < 1e6 for v in r.pos):
            if all(math.hypot(r.pos[0] - a, r.pos[1] - b) >= VIEWPOINT_M for a, b in spots):
                spots.append(r.pos)
    if spots:
        return len(spots)
    return len({int(((r.hdg + 15) % 360) // 30) for r in rows})


def final_pose(rows: list[Sighting], k_sep: float = K_SEP) -> dict:
    """Combine the sightings of one tag. Returns a dict:
    mean (x, y), med (x, y), n, kept, thr (outlier cut, m), unc (m),
    spread (m), views (viewpoints), sep (median, m), confirmed."""
    xy = np.array([(r.x, r.y) for r in rows], float)
    med = np.median(xy, axis=0)
    d = np.hypot(*(xy - med).T)
    thr = float(np.clip(3 * 1.4826 * float(np.median(d)), OUTLIER_MIN_M, OUTLIER_MAX_M))
    keep = d <= thr
    kept = [r for r, k in zip(rows, keep) if k]
    w = np.array([1.0 / sigma_of(r, k_sep) ** 2 for r in kept])
    xyk = xy[keep]
    mean = (w[:, None] * xyk).sum(axis=0) / w.sum()
    unc = math.sqrt(1.0 / w.sum())
    spread = math.sqrt(float((w * np.hypot(*(xyk - mean).T) ** 2).sum() / w.sum()))
    views = viewpoints(kept)
    return dict(mean=mean, med=med, n=len(rows), kept=len(kept), thr=thr,
                unc=max(unc, spread / 2), spread=spread, views=views,
                sep=float(np.median([r.sep for r in kept])),
                confirmed=(len(kept) >= CONFIRM_N and views >= CONFIRM_VIEWS
                           and max(unc, spread / 2) < CONFIRM_UNC))
