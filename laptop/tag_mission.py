#!/usr/bin/env python3
"""
Tag-localised mission: takeoff -> fly a route in the arena frame -> land.

Known floor tags (tag 1 at the pad centre plus landmarks, see tag_arena.yaml)
give the drone's true arena position whenever one is in view. Between
landmarks, PX4's position is used with the last correction applied, so:

  * waypoints are flown in the arena frame, and corrected for PX4 drift every
    time a landmark is seen;
  * target tags (4-7) are placed in the arena frame: from a known tag in the
    same picture when there is one ("tag", no drift at all), otherwise from
    the corrected PX4 position ("odom").

Writes a timestamped event log (takeoff command, each target found, land
command) and a CSV of every target sighting next to this script.

Tags come from the drone's own detector (UDP 5008) by default, so
tag_stream.py can show the video at the same time. Those detections carry tag
positions but not rotations: tags correct x/y, and the heading comes from PX4,
lined up at takeoff assuming the drone faces +x. Whenever two known tags are in
one picture (e.g. 20 and 21 at the start), the line between them corrects the
heading too; the mission checks this at the start and end of a trajectory. --laptop-camera detects on
the laptop instead (full pose, but then tag_stream.py cannot run).

The camera is assumed to look forward, tilted 60 deg down (--cam-pitch 60), with the top of
the image towards the drone's nose. Check that first with --dry-run: no
flight, it just prints the tag-based position while you carry the drone.

    python laptop/tag_mission.py --drone-id 3 --dry-run
    python laptop/tag_mission.py --drone-id 3
    python laptop/tag_mission.py --drone-id 3 --search     # search behaviour from tag_arena.yaml

--search flies the `search:` block of the arena file instead of the route:
entry -> pattern (red) -> decision at the TL corner -> next pattern (blue) only
if not every expected target has been found -> exit -> land. Patterns are
planned in search_patterns.py; preview them without a drone with
    python laptop/search_patterns.py --plot search_plan.png

Exit code: 0 all waypoints reached, 1 error or waypoint timeout, 130 Ctrl+C.
"""

import argparse
import csv
import logging
import math
import random
import select
import signal
import socket
import sys
import threading
import time
import types
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from apriltag_host import CAM_POS_M, Detector, passes_gate
from camera_stream import COMMAND_PORT, KEEPALIVE_S, FrameAssembler, drain_socket
from comms import CommsNode
from mission_v1 import PoseHistory
from protocol import (
    AT_DEBUG_PORT,
    CAMERA_STREAM_PORT,
    CMD_HOLD,
    CMD_LAND,
    CMD_START,
    NAV_FLYING,
    NAV_ROTATING,
    NAV_STUCK,
    NAV_TRAJ,
    TRAJ_MAX_PTS,
    TelemetryPacket,
    build_camera_stream_command,
    build_traj_packets,
    build_traj_start,
    parse_at_debug,
)
from search_patterns import SearchPlan, smooth_trajectory, write_csv
from tag_fusion import MAX_RANGE_M, Sighting, final_pose, view_angle
from send_trajectory import UPLOAD_GAP_S, load_trajectory
from simple_waypoint_mission import TelemetryTracker, send_command, send_goto

log = logging.getLogger("tag_mission")
HERE = Path(__file__).resolve().parent

CAMERA_FPS = 5             # keeps Wi-Fi free for telemetry
TAKEOFF_WAIT_S = 10.0      # CMD_START -> first waypoint
FIX_WAIT_S = 6.0           # then wait up to this long to see a known tag
ARRIVAL_RADIUS_M = 0.25
ARRIVAL_TIMEOUT_S = 25.0
CONNECT_TIMEOUT_S = 30.0
STALE_TIMEOUT_S = 3.0

BLEND = 0.5                # how much each new tag fix moves the correction (0-1)
MAX_JUMP_M = 1.0           # ignore a fix this far from where we think we are
RESEND_SHIFT_M = 0.10      # re-send the goal when the correction moves it this much
RESEND_MIN_S = 1.0         # ... but not more often than this
GOTO_RESEND_S = 2.0        # re-send if the drone has not started moving
# Heading from two known tags in one picture (e.g. 20 and 21, both on y = 0).
PAIR_MIN_SEP_M = 0.4       # the two tags must be at least this far apart
PAIR_DIST_TOL = 0.20       # ... and seen this close to their real spacing (20%), else a misread
HEADING_BLEND = 0.5        # each check moves the heading this fraction of the way
HEADING_MAX_STEP_DEG = 20  # a bigger correction needs 3 checks in a row that agree
HEADING_CHECK_S = 4.0      # at the start / end, wait up to this long for two-tag checks
HEADING_CHECK_N = 2        # ... and this many checks
HEADING_BURST_S = 2.0      # checks closer than this in time form one burst (after the flight)

# Floor tag with its top edge pointing +x. Rows are arena x, y, z(down) in tag
# axes (AprilTag tag frame: x right, y down the printed image, z into the floor).
FLOOR = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], float)


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def r_body_cam(pitch_deg: float) -> np.ndarray:
    """Camera axes in the drone frame; same formula as apriltag_host.R_BODY_CAM."""
    p = math.radians(pitch_deg)
    return np.array([[0.0, -math.sin(p), math.cos(p)],
                     [1.0, 0.0, 0.0],
                     [0.0, math.cos(p), math.sin(p)]])


# ---------------------------------------------------------------------------
# Arena config
# ---------------------------------------------------------------------------

class Arena:
    def __init__(self, path: Path):
        cfg = yaml.safe_load(path.read_text())
        self.tag_size = float(cfg.get("tag_size_m", 0.114))
        self.known = {}              # id -> (position (3,), tag->arena rotation)
        for tid, (x, y, yaw) in (cfg.get("known_tags") or {}).items():
            self.known[int(tid)] = (np.array([x, y, 0.0]), rot_z(math.radians(yaw)) @ FLOOR)
        self.targets = {int(t) for t in cfg.get("targets") or []}
        self.route = [(float(x), float(y)) for x, y in cfg.get("route") or []]
        self.search = SearchPlan(cfg["search"]) if cfg.get("search") else None
        # Optional tape-measured target positions (id: [x, y]) to score the result.
        self.truth = {int(k): (float(v[0]), float(v[1]))
                      for k, v in (cfg.get("target_truth") or {}).items()}
        if not self.known:
            raise SystemExit(f"{path}: no known_tags")
        if self.targets & set(self.known):
            raise SystemExit(f"{path}: a tag cannot be both known and a target")


# ---------------------------------------------------------------------------
# Pose maths
# ---------------------------------------------------------------------------

def tag_in_body(det, R_bc: np.ndarray, p_bc: np.ndarray):
    """Tag rotation (None if the detection has none) and centre in the drone
    frame (forward, right, down)."""
    t = np.array(det.t, float)
    if det.R is None:
        return None, R_bc @ t + p_bc
    return R_bc @ np.array(det.R, float).reshape(3, 3), R_bc @ t + p_bc


IMAGE_W, IMAGE_H = 320, 240        # QVGA, as at_detect.c


def off_centre(det) -> float:
    """How far a tag sits from the picture centre: 0 = centre, 1 = a corner."""
    c = getattr(det, "centre", None)
    if c is None:
        return float("nan")
    return math.hypot((c[0] - IMAGE_W / 2) / (IMAGE_W / 2), (c[1] - IMAGE_H / 2) / (IMAGE_H / 2)) / math.sqrt(2)


def drone_from_known(R_bt, t_b, P, M):
    """Drone position (3,) and body->arena rotation from one known tag."""
    R_ab = M @ R_bt.T
    return P - R_ab @ t_b, R_ab


def heading_of(R_ab: np.ndarray) -> float:
    return math.atan2(R_ab[1, 0], R_ab[0, 0])



class MissionClock:
    """Stopwatch from GO-<id> to LAND, with the time each stage started."""

    def __init__(self):
        self.t0 = None
        self.marks = []                          # (name, seconds since GO)

    def start(self):
        self.t0 = time.monotonic()
        self.mark("GO (confirmed)")

    def mark(self, name: str):
        if self.t0 is not None:
            self.marks.append((name, time.monotonic() - self.t0))

    def report(self):
        if self.t0 is None:
            return
        log.info("MISSION TIME:")
        for (name, t), nxt in zip(self.marks, self.marks[1:] + [None]):
            took = f"  ({nxt[1] - t:5.1f} s)" if nxt else ""
            log.info("  %6.1f s  %s%s", t, name, took)
        log.info("  total %.1f s (GO -> %s)", self.marks[-1][1], self.marks[-1][0])


CLOCK = MissionClock()


class Correction:
    """PX4 position -> arena: arena = rot(dyaw) * odom + T (horizontal only)."""

    def __init__(self):
        self.dyaw = 0.0
        self.T = np.zeros(2)
        self.have_fix = False
        self.last_fix = 0.0          # time.monotonic() of the last tag fix
        self._lock = threading.Lock()

    def assume_start(self, x: float, y: float, yaw: float) -> None:
        """Before any tag fix: the drone took off at (0, 0) facing +x."""
        with self._lock:
            self.dyaw = wrap(-yaw)
            self.T = -rot2(self.dyaw) @ np.array([x, y])

    def to_arena(self, x: float, y: float, yaw: float):
        with self._lock:
            p = rot2(self.dyaw) @ np.array([x, y]) + self.T
            return p[0], p[1], wrap(yaw + self.dyaw)

    def to_odom(self, ax: float, ay: float):
        with self._lock:
            p = rot2(-self.dyaw) @ (np.array([ax, ay]) - self.T)
            return p[0], p[1]

    @staticmethod
    def from_fix(fix_xy, fix_yaw: float, odom):
        """(T, dyaw) that one tag fix alone says: PX4's error at that moment."""
        ox, oy, oyaw = odom
        dyaw = wrap(fix_yaw - oyaw)
        return np.asarray(fix_xy, float) - rot2(dyaw) @ np.array([ox, oy]), dyaw

    def snap_heading(self, delta: float, odom) -> None:
        """Turn the heading by delta (rad), keeping the drone's arena x/y where it is."""
        o = np.array(odom[:2], float)
        with self._lock:
            here = rot2(self.dyaw) @ o + self.T
            self.dyaw = wrap(self.dyaw + delta)
            self.T = here - rot2(self.dyaw) @ o

    def update(self, fix_xy, fix_yaw: float, odom) -> bool:
        """Blend in a tag fix taken at PX4 pose odom = (x, y, yaw). False if rejected."""
        ox, oy, oyaw = odom
        new_T, new_dyaw = self.from_fix(fix_xy, fix_yaw, odom)
        ax, ay, _ = self.to_arena(ox, oy, oyaw)
        with self._lock:
            if self.have_fix:
                if math.hypot(fix_xy[0] - ax, fix_xy[1] - ay) > MAX_JUMP_M:
                    return False
                self.dyaw = wrap(self.dyaw + BLEND * wrap(new_dyaw - self.dyaw))
                self.T = self.T + BLEND * (new_T - self.T)
            else:
                self.dyaw, self.T, self.have_fix = new_dyaw, new_T, True
            self.last_fix = time.monotonic()
            return True


class FixHistory:
    """Every known-tag fix of the flight, for correcting 'odom' sightings afterwards.

    Live, an 'odom' sighting can only use the correction from the fixes BEFORE it, so
    it carries all the PX4 drift since then. Once the flight is over, the fixes AFTER
    it are known too: the correction at the sighting's time is blended from the last
    burst of fixes before it and the first burst after it, in proportion to time.
    A burst = fixes within BURST_S of each other; its median resists one bad fix.
    Fixes the live correction rejected (over MAX_JUMP_M) are kept here as well.
    """
    BURST_S = 1.0

    def __init__(self):
        # (time, T (2,), dyaw, drone arena xy (2,), PX4 odom, refs [(P xy, t_b xy, w)] or None)
        self.fixes = []

    def add(self, t: float, T, dyaw: float, drone_xy, odom=None, refs=None) -> None:
        self.fixes.append((t, np.asarray(T, float), dyaw, np.asarray(drone_xy, float)[:2], odom, refs))

    @staticmethod
    def _redo(f, model):
        """A fix as (time, T, dyaw, xy), redone with the after-flight heading model if possible."""
        t, T, dyaw, xy, odom, refs = f
        if model is None or odom is None or refs is None:
            return t, T, dyaw, xy
        dyaw = model(t)
        R = rot2(wrap(odom[2] + dyaw))
        w = np.array([r[2] for r in refs])
        w = w / w.sum()
        xy = sum(wi * (P - R @ tb) for wi, (P, tb, _) in zip(w, refs))
        return t, xy - rot2(dyaw) @ np.array(odom[:2]), dyaw, xy

    @staticmethod
    def _burst(group):
        T = np.median([f[1] for f in group], axis=0)
        dyaw = math.atan2(np.median([math.sin(f[2]) for f in group]),
                          np.median([math.cos(f[2]) for f in group]))
        xy = np.median([f[3] for f in group], axis=0)
        return group[0][0] if len(group) == 1 else float(np.median([f[0] for f in group])), T, dyaw, xy

    def at(self, t: float, model=None):
        """(T, dyaw, [arena xy of the fixes used], 'before'/'after'/'both') at time t, or None.
        model: optional after-flight heading (time -> dyaw); the fixes are redone with it."""
        fixes = [self._redo(f, model) for f in self.fixes]
        before = [f for f in fixes if f[0] <= t]
        after = [f for f in fixes if f[0] > t]
        b = self._burst([f for f in before if f[0] >= before[-1][0] - self.BURST_S]) if before else None
        a = self._burst([f for f in after if f[0] <= after[0][0] + self.BURST_S]) if after else None
        if b and a:
            w = min(max((t - b[0]) / max(a[0] - b[0], 1e-6), 0.0), 1.0)
            T = (1 - w) * b[1] + w * a[1]
            dyaw = wrap(b[2] + w * wrap(a[2] - b[2])) if model is None else model(t)
            return T, dyaw, [b[3], a[3]], "both"
        if b or a:
            f = b or a
            return f[1], f[2] if model is None else model(t), [f[3]], "before" if b else "after"
        return None


# ---------------------------------------------------------------------------
# Camera thread: localise + find targets
# ---------------------------------------------------------------------------

class TagLocaliser:
    """Localises from tag detections, taken from one of two sources:

    * detector None: the drone's own detections (UDP 5008, ~2 Hz). They carry
      each tag's position but not its rotation, so tags correct x/y while the
      heading comes from PX4 (lined up at takeoff). Leaves the camera stream
      free for tag_stream.py.
    * a Detector: the laptop streams the camera (UDP 5009) and detects every
      frame, with full pose (x/y and heading). Cannot run with tag_stream.py.
    """

    def __init__(self, esp_ip, drone_id: int, arena: Arena, poses: PoseHistory,
                 corr: Correction, detector, R_bc: np.ndarray, csv_path: Path):
        self.drone_id = drone_id
        self.esp_ip, self.arena, self.poses, self.corr = esp_ip, arena, poses, corr
        self.detector, self.R_bc, self.p_bc = detector, R_bc, np.array(CAM_POS_M, float)
        self.csv_path = csv_path
        self.t0 = time.monotonic()
        self.frames = self.fixes = self.rejected = 0
        self.last_fix_tags: list[int] = []
        self.sightings: dict[int, list[Sighting]] = {}
        self._prev_xy = None                    # (arena x/y, time) of the last tag fix, for speed
        self._last_fix_xy = None                # where the last tag fix was ("odom" sightings)
        self.first_seen: dict[int, float] = {}
        self.known_seen: dict[int, list] = {}  # known id -> [frames seen, first seen s, closest range m]
        self.history = FixHistory()             # every known-tag fix, to re-correct 'odom' sightings
        # target id -> one dict per sighting (same order as self.sightings): the raw data
        # needed to redo it after the flight (heading model, fixes before AND after).
        self.raw: dict[int, list] = {}
        # Two-known-tag heading checks: dicts(t, ids, err_deg, dyaw, applied).
        self.heading_checks: list[dict] = []
        self.heading_bursts: list = []          # after the flight: (time, dyaw) per burst
        self.heading_rejected = 0
        self._big_streak: list[float] = []      # recent large corrections waiting to agree
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tag-localiser", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def mark_takeoff(self):
        self.t0 = time.monotonic()

    def results(self):
        """(id, x, y, sightings, method, first seen s). x/y = tag_fusion.final_pose
        of the 'tag' sightings (known tag in the same picture), or of the 'odom'
        ones if a target was never seen with a known tag."""
        return [(tid, f["mean"][0], f["mean"][1], f["n"], method, first)
                for tid, f, method, first in self.final_poses()]

    def final_poses(self):
        """[(id, final_pose dict, method, first seen s)] for every target seen.
        The dict also has 'how': sighting counts by method and by known tag."""
        out = []
        with self._lock:
            model = self._heading_model()
            for tid, live_pts in sorted(self.sightings.items()):
                pts, used = [], dict(heading=0, both=0, before=0, after=0)
                for p, r in zip(live_pts, self.raw.get(tid, [])):
                    c = self._corrected(r, model)
                    if c is None:
                        pts.append(p)
                    else:
                        pts.append(c[0])
                        used[c[1]] += 1
                pts += live_pts[len(pts):]
                live_near = [q for q in live_pts if q.rng <= MAX_RANGE_M] or live_pts
                live = final_pose([q for q in live_near if q.method == "tag"] or live_near)["mean"]
                near = [p for p in pts if p.rng <= MAX_RANGE_M] or pts
                best = [p for p in near if p.method == "tag"] or near
                f = final_pose(best)
                by_known: dict[int, int] = {}
                for p in best:
                    for k in p.known:
                        by_known[k] = by_known.get(k, 0) + 1
                f["how"] = dict(tag=sum(p.method == "tag" for p in pts),
                                odom=sum(p.method == "odom" for p in pts),
                                too_far=len(pts) - len(near), by_known=by_known)
                if any(used.values()):
                    f["recorrected"] = dict(live=live, **used)
                out.append((tid, f, best[0].method, self.first_seen[tid]))
        return out

    def _check_heading(self, captured, known_dets, odom):
        """Heading from every pair of known tags in this picture; snap the correction to it."""
        tb = {d.id: tag_in_body(d, self.R_bc, self.p_bc)[1][:2] for d in known_dets}
        sins = coss = 0.0
        ids = set()
        for i, a in enumerate(known_dets):
            for b in known_dets[i + 1:]:
                v = self.arena.known[b.id][0][:2] - self.arena.known[a.id][0][:2]   # real, arena
                m = tb[b.id] - tb[a.id]                                             # seen, drone frame
                L, Lm = float(np.hypot(*v)), float(np.hypot(*m))
                if L < PAIR_MIN_SEP_M or abs(Lm - L) > PAIR_DIST_TOL * L:
                    continue
                psi = wrap(math.atan2(v[1], v[0]) - math.atan2(m[1], m[0]))         # true heading
                w = L * L / max(a.range_m * b.range_m, 0.01)                        # long + close = good
                sins += w * math.sin(psi)
                coss += w * math.cos(psi)
                ids |= {a.id, b.id}
        if not ids:
            return
        psi_true = math.atan2(sins, coss)
        err = wrap(psi_true - self.corr.to_arena(*odom)[2])
        err_deg = math.degrees(err)
        applied = abs(err_deg) <= HEADING_MAX_STEP_DEG
        if applied:
            self._big_streak = []
            self.corr.snap_heading(HEADING_BLEND * err, odom)
        else:                                    # big: only if 3 checks in a row agree
            self._big_streak.append(err)
            recent = self._big_streak[-3:]
            if len(recent) == 3 and max(abs(math.degrees(wrap(e - recent[-1]))) for e in recent) < 5:
                applied = True
                self._big_streak = []
                self.corr.snap_heading(float(np.median(recent)), odom)
                log.warning("Heading corrected by a large %+.1f deg (3 two-tag checks agreed)", err_deg)
            else:
                self.heading_rejected += 1
        rec = dict(t=captured, ids=tuple(sorted(ids)), err_deg=err_deg,
                   dyaw=wrap(psi_true - odom[2]), applied=applied)
        with self._lock:
            new_burst = (not self.heading_checks
                         or captured - self.heading_checks[-1]["t"] > HEADING_BURST_S)
            self.heading_checks.append(rec)
        if new_burst:
            log.info("Heading check from known tags %s: heading was off by %+.1f deg%s",
                     "+".join(map(str, rec["ids"])), err_deg, "" if applied else " (not applied yet)")

    def checks_since(self, t: float):
        """Two-tag heading checks made at or after time.monotonic() t."""
        with self._lock:
            return [c for c in self.heading_checks if c["t"] >= t]

    def _heading_model(self):
        """After the flight: PX4's true heading offset over time from the two-tag checks
        (median per burst, straight line between bursts, held flat before the first and
        after the last). None if there were no checks."""
        checks = self.heading_checks
        if not checks:
            return None
        bursts, cur = [], [checks[0]]
        for c in checks[1:]:
            if c["t"] - cur[-1]["t"] > HEADING_BURST_S:
                bursts.append(cur)
                cur = []
            cur.append(c)
        bursts.append(cur)
        pts = []
        for g in bursts:
            ref = g[0]["dyaw"]
            d = float(np.median([wrap(c["dyaw"] - ref) for c in g]))
            pts.append((float(np.median([c["t"] for c in g])), wrap(ref + d), len(g), g[0]["ids"]))
        self.heading_bursts = pts

        def model(t):
            if t <= pts[0][0]:
                return pts[0][1]
            for (t0, a0, *_), (t1, a1, *_) in zip(pts, pts[1:]):
                if t <= t1:
                    return wrap(a0 + (t - t0) / max(t1 - t0, 1e-6) * wrap(a1 - a0))
            return pts[-1][1]
        return model

    def _corrected(self, r, model):
        """(sighting redone after the flight, how) or None to keep the live one."""
        sight = r["sight"]
        if r["method"] == "tag":
            if model is None or r["refs"] is None or r["odom"] is None:
                return None
            hdg = wrap(r["odom"][2] + model(r["t"]))
            R = rot2(hdg)
            w = np.array([x[2] for x in r["refs"]])
            w = w / w.sum()
            est = sum(wi * (P + R @ (r["t_b"] - tb)) for wi, (P, tb, _) in zip(w, r["refs"]))
            return replace(sight, x=float(est[0]), y=float(est[1]), hdg=math.degrees(hdg)), "heading"
        c = self.history.at(r["t"], model) if r["odom"] is not None else None
        if c is None:
            return None
        T, dyaw, fix_xy, side = c
        ox, oy, oyaw = r["odom"]
        dxy = rot2(dyaw) @ np.array([ox, oy]) + T
        hdg = wrap(oyaw + dyaw)
        xy = dxy + rot2(hdg) @ r["t_b"]
        sep = min(float(np.hypot(*(dxy - q))) for q in fix_xy)
        return replace(sight, x=float(xy[0]), y=float(xy[1]), sep=sep, hdg=math.degrees(hdg),
                       pos=(float(dxy[0]), float(dxy[1]))), side

    def known_summary(self):
        """{known id: (frames seen, first seen s, closest range m)}."""
        with self._lock:
            return {k: tuple(v) for k, v in sorted(self.known_seen.items())}

    def _open_csv(self):
        f = self.csv_path.open("w", newline="")
        writer = csv.writer(f)
        writer.writerow(["wall_time", "t_since_takeoff_s", "tag_id", "x_m", "y_m",
                         "method", "drone_x_m", "drone_y_m", "margin",
                         "sep_m", "range_m", "view_deg", "known_in_view", "speed_ms", "heading_deg", "known_ids",
                         "off_centre"])
        return f, writer

    def _run(self):
        if self.detector is None:
            self._run_drone_detections()
        else:
            self._run_laptop_camera()

    def _run_drone_detections(self):
        """Use the drone's own detections (AprilTag debug stream, UDP 5008)."""
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csv_file = None
        try:
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rx.bind(("0.0.0.0", AT_DEBUG_PORT))
            rx.settimeout(0.2)
            csv_file, writer = self._open_csv()
            last_frame_ms = None
            while not self._stop.is_set():
                try:
                    data, _ = rx.recvfrom(2048)
                except socket.timeout:
                    continue
                arrived = time.monotonic()
                pkt = parse_at_debug(data)
                # Sent at 10 Hz; a new frame_ms means a newly processed frame.
                if pkt is None or pkt.drone_id != self.drone_id or pkt.frame_ms == 0 \
                        or pkt.frame_ms == last_frame_ms:
                    continue
                last_frame_ms = pkt.frame_ms
                self.frames += 1
                # The frame was taken about proc_ms before its result was sent.
                captured = arrived - pkt.proc_ms / 1000.0 - 0.05
                dets = [types.SimpleNamespace(id=d.id, t=(d.tx, d.ty, d.tz), R=None,
                                              range_m=d.range_m, decision_margin=d.margin,
                                              centre=(d.cx, d.cy))
                        for d in pkt.detections if d.has_pose]
                if dets:
                    self._process(captured, dets, writer)
                    csv_file.flush()
        except Exception as exc:        # the flight carries on without detections
            log.error("Tag localiser stopped: %s", exc)
        finally:
            if csv_file is not None:
                csv_file.close()
            rx.close()

    def _run_laptop_camera(self):
        """Stream the camera to the laptop and detect every frame (full pose)."""
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csv_file = None
        try:
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            rx.bind(("0.0.0.0", CAMERA_STREAM_PORT))
            rx.setblocking(False)
            start_cmd = build_camera_stream_command(True, CAMERA_FPS)
            assembler = FrameAssembler()
            csv_file, writer = self._open_csv()
            next_keepalive = 0.0
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= next_keepalive:         # drone streams only while it hears us
                    next_keepalive = now + KEEPALIVE_S
                    try:
                        tx.sendto(start_cmd, (self.esp_ip, COMMAND_PORT))
                    except OSError as exc:
                        log.warning("camera keepalive failed: %s", exc)
                select.select([rx], [], [], 0.05)
                frame, _, _ = drain_socket(rx, assembler, self.esp_ip)
                assembler.expire(time.monotonic())
                if frame is not None:
                    self._handle_frame(frame, writer)
                    csv_file.flush()
        except Exception as exc:        # the flight carries on without the camera
            log.error("Tag localiser stopped: %s", exc)
        finally:
            try:
                tx.sendto(build_camera_stream_command(False), (self.esp_ip, COMMAND_PORT))
            except OSError:
                pass
            if csv_file is not None:
                csv_file.close()
            rx.close()
            tx.close()

    def _handle_frame(self, frame, writer):
        captured = time.monotonic() - frame.age_ms / 1000.0
        gray = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return
        self.frames += 1
        dets = [d for d in self.detector.detect(gray)
                if passes_gate(d) and d.t is not None and d.R is not None]
        if dets:
            self._process(captured, dets, writer)

    def _process(self, captured, dets, writer):
        """Localise from the known tags in one frame, then place its targets."""
        odom = self.poses.at(captured)          # PX4 (x, y, yaw) when the frame was taken

        # 0. Heading from two known tags in one picture (drone detections carry no rotation).
        known_dets = [d for d in dets if d.id in self.arena.known]
        if odom is not None and len(known_dets) >= 2 and all(d.R is None for d in known_dets):
            self._check_heading(captured, known_dets, odom)

        # 1. Drone pose from every known tag in view (closer = more weight).
        refs = []                                # (drone position, R_ab, weight)
        refs_rel = []                            # (tag P xy, tag in drone frame xy, weight), to redo later
        for d in dets:
            if d.id in self.arena.known:
                R_bt, t_b = tag_in_body(d, self.R_bc, self.p_bc)
                P, M = self.arena.known[d.id]
                if R_bt is not None:
                    p, R_ab = drone_from_known(R_bt, t_b, P, M)
                elif odom is not None:           # no tag rotation: heading from PX4
                    R_ab = rot_z(self.corr.to_arena(*odom)[2])
                    p = P - R_ab @ t_b
                else:
                    continue
                refs.append((p, R_ab, 1.0 / max(d.range_m, 0.1) ** 2))
                if R_bt is None:
                    refs_rel.append((P[:2].copy(), t_b[:2].copy(), refs[-1][2]))
        if len(refs_rel) != len(refs):
            refs_rel = None                      # rotations from the tags: nothing to redo
        drone_xy, yaw, speed, ids = None, None, 0.0, []
        if refs:
            w = np.array([r[2] for r in refs])
            w /= w.sum()
            drone_xy = sum(wi * r[0][:2] for wi, r in zip(w, refs))
            yaw = math.atan2(sum(wi * math.sin(heading_of(r[1])) for wi, r in zip(w, refs)),
                             sum(wi * math.cos(heading_of(r[1])) for wi, r in zip(w, refs)))
            ids = sorted({d.id for d in dets if d.id in self.arena.known})
            self._note_known([d for d in dets if d.id in self.arena.known], captured)
            if self._prev_xy is not None and captured > self._prev_xy[1]:
                speed = float(np.hypot(*(drone_xy - self._prev_xy[0]))) / (captured - self._prev_xy[1])
            self._prev_xy = (drone_xy.copy(), captured)
            self._last_fix_xy = drone_xy.copy()
            if odom is not None:
                T_fix, dyaw_fix = Correction.from_fix(drone_xy, yaw, odom)
                self.history.add(captured, T_fix, dyaw_fix, drone_xy, odom, refs_rel)  # kept even if rejected
                first = not self.corr.have_fix
                if self.corr.update(drone_xy, yaw, odom):
                    self.fixes += 1
                    self.last_fix_tags = ids
                    if first:
                        log.info("First tag fix from tag(s) %s: drone at x=%.2f y=%.2f heading %.0f deg",
                                 ids, drone_xy[0], drone_xy[1], math.degrees(yaw))
                else:
                    self.rejected += 1
                    log.warning("Ignored fix from tag(s) %s at x=%.2f y=%.2f (over %.1f m from "
                                "the current estimate - check the tag positions)",
                                ids, drone_xy[0], drone_xy[1], MAX_JUMP_M)

        # 2. Targets.
        for d in dets:
            if d.id not in self.arena.targets:
                continue
            R_bt, t_b = tag_in_body(d, self.R_bc, self.p_bc)
            off = off_centre(d)
            if refs:                              # the less centred of target and known tags counts
                k_off = [off_centre(k) for k in known_dets]
                k_off = [v for v in k_off if not math.isnan(v)]
                if k_off and not math.isnan(off):
                    off = max(off, float(np.mean(k_off)))
            if refs:                              # same picture as a known tag: no drift
                w = np.array([r[2] for r in refs])
                est = sum(wi * (r[0] + r[1] @ t_b) for wi, r in zip(w / w.sum(), refs))
                x, y, method, dx, dy = est[0], est[1], "tag", drone_xy[0], drone_xy[1]
                sep = min(float(np.hypot(*(est - self.arena.known[i][0])[:2])) for i in ids)
                hdg = yaw
            elif odom is not None:                # corrected PX4 position + offset
                dx, dy, hdg = self.corr.to_arena(*odom)
                c, s = math.cos(hdg), math.sin(hdg)
                x = dx + t_b[0] * c - t_b[1] * s
                y = dy + t_b[0] * s + t_b[1] * c
                method = "odom"
                # PX4 drift grows with the distance flown since the last tag fix.
                sep = (float(np.hypot(dx - self._last_fix_xy[0], dy - self._last_fix_xy[1]))
                       if self._last_fix_xy is not None else 1.0)
            else:
                continue
            sight = Sighting(x=float(x), y=float(y), method=method, sep=sep, rng=float(d.range_m),
                             ang=view_angle(d), nk=len(ids) if refs else 0, speed=speed,
                             hdg=math.degrees(hdg), known=tuple(ids) if refs else (),
                             pos=(float(dx), float(dy)), off=off)
            since = captured - self.t0
            self._record(d.id, sight, since)
            with self._lock:
                self.raw.setdefault(d.id, []).append(dict(
                    t=captured, method=method, odom=odom, t_b=np.asarray(t_b, float)[:2].copy(),
                    refs=refs_rel if method == "tag" else None, sight=sight))
            writer.writerow([time.strftime("%H:%M:%S"), f"{since:.2f}", d.id, f"{x:.3f}",
                             f"{y:.3f}", method, f"{dx:.3f}", f"{dy:.3f}",
                             f"{d.decision_margin:.1f}", f"{sep:.3f}", f"{d.range_m:.3f}",
                             f"{sight.ang:.1f}", sight.nk, f"{speed:.2f}", f"{sight.hdg:.1f}",
                             ";".join(str(i) for i in sight.known), f"{sight.off:.2f}"])

    def _note_known(self, known_dets, captured):
        """Count known-tag sightings; log the first one of each."""
        since = captured - self.t0
        for d in known_dets:
            with self._lock:
                rec = self.known_seen.get(d.id)
                if rec is None:
                    self.known_seen[d.id] = [1, since, float(d.range_m)]
                else:
                    rec[0] += 1
                    rec[2] = min(rec[2], float(d.range_m))
            if rec is None:
                P = self.arena.known[d.id][0]
                log.info("KNOWN TAG %d seen at t=%.1f s (range %.2f m) - reference at x=%.2f y=%.2f",
                         d.id, since, d.range_m, P[0], P[1])

    def _record(self, tid, sight: Sighting, since):
        with self._lock:
            new = tid not in self.sightings
            self.sightings.setdefault(tid, []).append(sight)
            if new:
                self.first_seen[tid] = since
        if new:
            how = (f"from known tag(s) {', '.join(map(str, sight.known))}" if sight.method == "tag"
                   else "from PX4 position (no known tag in view)")
            log.info("TARGET %d FOUND at x=%.2f y=%.2f m (arena frame, %s) t=%.1f s",
                     tid, sight.x, sight.y, how, since)


# ---------------------------------------------------------------------------
# Flight
# ---------------------------------------------------------------------------

def fly_to(comms, drone_id, tracker, corr: Correction, ax: float, ay: float,
           radius: float = ARRIVAL_RADIUS_M, timeout: float = ARRIVAL_TIMEOUT_S) -> bool:
    """Fly to arena (ax, ay); the goal is re-sent as the drift correction changes."""
    start = time.monotonic()
    last_goal, last_sent, last_print, saw_moving = None, 0.0, 0.0, False
    while time.monotonic() - start < timeout:
        now = time.monotonic()
        snap = tracker.latest()
        if now - snap.received_at > STALE_TIMEOUT_S:
            raise ConnectionError(f"Telemetry is stale ({now - snap.received_at:.1f} s).")
        gx, gy = corr.to_odom(ax, ay)
        shifted = last_goal is not None and math.hypot(gx - last_goal[0], gy - last_goal[1]) > RESEND_SHIFT_M
        if (last_goal is None
                or (shifted and now - last_sent >= RESEND_MIN_S)
                or (not saw_moving and now - last_sent >= GOTO_RESEND_S)):
            send_goto(comms, drone_id, gx, gy)
            last_goal, last_sent = (gx, gy), now
        pkt = snap.packet
        if snap.received_at > start:
            if pkt.nav_state in (NAV_ROTATING, NAV_FLYING, NAV_STUCK):
                saw_moving = True
            cx, cy, _ = corr.to_arena(pkt.ned_x, pkt.ned_y, pkt.heading_rad)
            dist = math.hypot(cx - ax, cy - ay)
            if now - last_print >= 1.0:
                last_print = now
                log.info("  at x=%.2f y=%.2f  goal x=%.2f y=%.2f  dist %.2f m  (last tag fix %.1f s ago)",
                         cx, cy, ax, ay, dist, now - corr.last_fix if corr.have_fix else float("nan"))
            if dist <= radius:
                return True
        time.sleep(0.2)
    log.warning("Waypoint (%.2f, %.2f) not reached in %.0f s", ax, ay, timeout)
    return False


def play_trajectory(comms, drone_id, tracker, corr: Correction, pts: np.ndarray,
                    dt_ms: int, stop_when=None) -> bool:
    """Fly a trajectory given as offsets in the ARENA frame (x forward, y right).

    The ESP32 plays it on its own clock in PX4's frame, starting from wherever the
    drone hovers, so the offsets are rotated into PX4's frame with the current
    tag correction before upload. It cannot be steered once it has started.

    stop_when: optional function, checked every 0.2 s; when it returns True the
    drone is told to hold and this returns True early (used by --search)."""
    with corr._lock:
        dyaw = corr.dyaw
    xy = (rot2(-dyaw) @ pts[:, :2].T).T
    pts_px4 = np.column_stack([xy, pts[:, 2]])
    snap = tracker.latest().packet
    sx, sy, _ = corr.to_arena(snap.ned_x, snap.ned_y, snap.heading_rad)

    traj_id = random.randint(1, 255)
    packets = build_traj_packets(traj_id, pts_px4)
    start = build_traj_start(traj_id, dt_ms)
    for attempt in range(1, 6):
        for p in packets:
            comms.send_raw(drone_id, p)
            time.sleep(UPLOAD_GAP_S)
        sent = time.monotonic()
        comms.send_raw(drone_id, start)
        while time.monotonic() - sent < 1.5:
            snap = tracker.latest()
            if snap.received_at > sent and snap.packet.nav_state == NAV_TRAJ:
                break
            time.sleep(0.05)
        else:
            log.warning("Trajectory not started (attempt %d, see ESP log)", attempt)
            continue
        break
    else:
        raise RuntimeError("drone refused the trajectory")

    duration = (len(pts) - 1) * dt_ms / 1000.0
    log.info("Trajectory started from arena x=%.2f y=%.2f (%.1f s)", sx, sy, duration)
    CLOCK.mark("trajectory started")
    end = time.monotonic() + duration + 10.0
    last_print = 0.0
    stopped = False
    while time.monotonic() < end:
        snap = tracker.latest()
        if time.monotonic() - snap.received_at > STALE_TIMEOUT_S:
            raise ConnectionError("Telemetry is stale during the trajectory.")
        if snap.packet.nav_state != NAV_TRAJ:
            break
        if stop_when is not None and stop_when():
            send_command(comms, drone_id, CMD_HOLD, "CMD_HOLD")
            stopped = True
            break
        if time.monotonic() - last_print >= 1.0:
            last_print = time.monotonic()
            x, y, _ = corr.to_arena(snap.packet.ned_x, snap.packet.ned_y, snap.packet.heading_rad)
            age = time.monotonic() - corr.last_fix if corr.have_fix else float("nan")
            log.info("  at x=%.2f y=%.2f  (last tag fix %.1f s ago)", x, y, age)
        time.sleep(0.2)
    snap = tracker.latest().packet
    if stopped:
        x, y, _ = corr.to_arena(snap.ned_x, snap.ned_y, snap.heading_rad)
        log.info("Trajectory stopped early at arena x=%.2f y=%.2f", x, y)
        return True
    x, y, _ = corr.to_arena(snap.ned_x, snap.ned_y, snap.heading_rad)
    ex, ey = sx + pts[-1, 0], sy + pts[-1, 1]
    CLOCK.mark("trajectory ended")
    log.info("Trajectory ended at arena x=%.2f y=%.2f; planned end x=%.2f y=%.2f (off by %.2f m)",
             x, y, ex, ey, math.hypot(x - ex, y - ey))
    return False


# ---------------------------------------------------------------------------
# Search behaviour (--search, configured in the arena file's `search:` block)
# ---------------------------------------------------------------------------

def run_search(comms, drone_id, tracker, corr: Correction, loc, arena: Arena,
               dt_ms: int, stamp: str = "") -> int:
    """entry -> pattern -> decision -> (next pattern) -> exit. Returns waypoints missed.

    Each pattern trajectory that is uploaded is also saved next to this script as
    search_<pattern>_<stamp>.csv (t,x,y,z, arena frame), alongside the mission log."""
    plan = arena.search
    expected = plan.expected_tags or len(arena.targets)

    def found() -> int:
        return len(loc.results())              # target ids seen at least once

    def all_found() -> bool:
        return found() >= expected

    def goto(x, y, radius=ARRIVAL_RADIUS_M) -> bool:
        snap = tracker.latest().packet
        cx, cy, _ = corr.to_arena(snap.ned_x, snap.ned_y, snap.heading_rad)
        timeout = math.hypot(x - cx, y - cy) / plan.goto_speed * 1.5 + 15.0
        return fly_to(comms, drone_id, tracker, corr, x, y, radius=radius, timeout=timeout)

    def arena_pose():
        snap = tracker.latest()
        if time.monotonic() - snap.received_at > STALE_TIMEOUT_S:
            raise ConnectionError("Telemetry is stale.")
        p = snap.packet
        return corr.to_arena(p.ned_x, p.ned_y, p.heading_rad)    # x, y, heading (arena frame)

    def fly_smooth(points, stop_when=None, save_as=None) -> bool:
        """Trajectory from the current position through `points`; the heading does not change.
        save_as: pattern name -> also write the uploaded path to search_<name>_<stamp>.csv."""
        sx, sy, yaw = arena_pose()
        prof = smooth_trajectory([(sx, sy)] + list(points), plan.speed, dt_ms / 1000.0)
        if prof["length"] < 0.03:
            return False
        offsets = np.column_stack([prof["xy"] - prof["xy"][0], np.zeros(len(prof["xy"]))])
        log.info("  smooth path %.2f m, %.0f s, %d points", prof["length"], prof["duration"], len(offsets))
        if save_as:
            path = write_csv(HERE / f"search_{save_as}_{stamp or time.strftime('%Y%m%d_%H%M%S')}.csv", prof,
                             f"pattern '{save_as}' as uploaded, start x={sx:.2f} y={sy:.2f}, "
                             f"heading {math.degrees(yaw):.0f} deg, {time.strftime('%H:%M:%S')}")
            log.info("  trajectory saved to %s", path.name)
        return play_trajectory(comms, drone_id, tracker, corr, offsets,
                               int(round(prof["dt"] * 1000)), stop_when=stop_when)

    def face(heading_deg: float) -> float:
        """Turn the nose to `heading_deg` (arena frame). There is no yaw command: CMD_GOTO
        points the drone at its goal while it flies, so aim at a point `look_m` away in the
        wanted direction, wait until the heading is within tolerance, then HOLD (which keeps
        the heading). Returns the final heading error in degrees."""
        want = math.radians(heading_deg)
        tol = math.radians(plan.turn["tolerance_deg"])
        for attempt in range(1, plan.turn["attempts"] + 1):
            x, y, yaw = arena_pose()
            if abs(wrap(yaw - want)) <= tol:
                break
            gx, gy = corr.to_odom(x + plan.turn["look_m"] * math.cos(want),
                                  y + plan.turn["look_m"] * math.sin(want))
            send_goto(comms, drone_id, gx, gy)
            t_end = time.monotonic() + plan.turn["timeout_s"]
            while time.monotonic() < t_end and abs(wrap(arena_pose()[2] - want)) > tol:
                time.sleep(0.1)
            send_command(comms, drone_id, CMD_HOLD, "CMD_HOLD")
            time.sleep(0.8)                     # let it stop
            log.info("  turn attempt %d: heading %.0f deg (want %.0f)", attempt,
                     math.degrees(arena_pose()[2]), heading_deg)
        return math.degrees(wrap(arena_pose()[2] - want))

    stop = all_found if plan.stop_early else None
    missed = 0
    log.info("SEARCH: %s mode, %d pattern(s), expecting %d target(s), max %d loop(s)",
             plan.mode, len(plan.patterns), expected, plan.max_loops)
    for x, y in plan.entry:                     # drift-corrected, like the route
        if not goto(x, y):
            missed += 1
    for loop in range(plan.max_loops):
        name, pts, heading = plan.patterns[loop % len(plan.patterns)]
        log.info("SEARCH pattern '%s' (loop %d/%d)", name, loop + 1, plan.max_loops)
        if not goto(*pts[0]):                   # back near the decision corner (any heading)
            missed += 1
        if heading is not None and plan.mode == "trajectory":
            err = face(heading)
            fly_smooth([pts[0]])                # onto the exact corner without turning
            time.sleep(1.0)                     # settle before the pattern locks its heading
            _, _, yaw = arena_pose()
            log.info("SEARCH pattern '%s' facing %.0f deg (wanted %.0f, error %.0f)",
                     name, math.degrees(yaw), heading, err)
            if abs(err) > plan.turn["tolerance_deg"]:
                log.warning("Heading still %.0f deg off after turning; flying the pattern anyway.", err)
        else:
            if not goto(*pts[0], radius=0.10):  # start exactly on the decision corner
                missed += 1
            time.sleep(1.0)                     # settle before a trajectory locks its start
        if plan.mode == "trajectory":
            fly_smooth(pts[1:], stop_when=stop, save_as=name)
        else:
            for x, y in pts[1:]:
                if stop is not None and stop():
                    send_command(comms, drone_id, CMD_HOLD, "CMD_HOLD")
                    break
                if not goto(x, y):
                    missed += 1
                    send_command(comms, drone_id, CMD_HOLD, "CMD_HOLD")
        n = found()
        if all_found():
            log.info("DECISION: %d/%d targets found -> go back", n, expected)
            break
        last = loop + 1 == plan.max_loops
        log.info("DECISION: %d/%d targets found -> %s", n, expected,
                 "out of loops, go back" if last else "try again")
    for x, y in plan.exit:
        if not goto(x, y):
            missed += 1
    return missed


def heading_check(loc: TagLocaliser, label: str, since: float) -> bool:
    """Wait (hovering) for HEADING_CHECK_N two-known-tag heading checks, up to HEADING_CHECK_S."""
    deadline = time.monotonic() + HEADING_CHECK_S
    while True:
        got = [c for c in loc.checks_since(since) if c["applied"]]
        if len(got) >= HEADING_CHECK_N or time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    if not got:
        log.warning("%s HEADING CHECK: two known tags (e.g. 20 and 21) not seen together - "
                    "heading NOT checked, it comes from PX4 only.", label)
        return False
    ids = sorted({i for c in got for i in c["ids"]})
    log.info("%s HEADING CHECK: tags %s seen together in %d picture(s); heading was off by %+.1f deg, "
             "now %+.1f deg after correction", label, "+".join(map(str, ids)), len(got),
             got[0]["err_deg"], got[-1]["err_deg"] * (1 - HEADING_BLEND))
    return True


def print_results(loc: TagLocaliser, arena: Arena) -> None:
    log.info("Frames %d, tag fixes %d (%d ignored).", loc.frames, loc.fixes, loc.rejected)
    known = loc.known_summary()
    log.info("KNOWN TAGS (reference tags used to correct the drone's position):")
    for kid, (P, _) in sorted(loc.arena.known.items()):
        if kid in known:
            n, first, rmin = known[kid]
            log.info("  tag %d at (%+.2f, %+.2f): seen in %d frame(s), first t=%.1fs, closest %.2f m",
                     kid, P[0], P[1], n, first, rmin)
        else:
            log.info("  tag %d at (%+.2f, %+.2f): NOT seen", kid, P[0], P[1])
    rows = loc.final_poses()
    checks = loc.heading_checks
    if checks:
        log.info("HEADING CHECKS (two known tags in one picture): %d, %d not applied",
                 len(checks), loc.heading_rejected)
        prev = None
        for t, dyaw, n, ids in loc.heading_bursts:
            drift = f", drifted {math.degrees(wrap(dyaw - prev)):+.1f} deg since the previous one" \
                if prev is not None else ""
            log.info("  t=%.1fs: tags %s, %d check(s), first said heading off by %+.1f deg%s",
                     t - loc.t0, "+".join(map(str, ids)), n,
                     next(c["err_deg"] for c in checks if abs(c["t"] - t) <= HEADING_BURST_S), drift)
            prev = dyaw
    else:
        log.warning("HEADING CHECKS: none - two known tags were never seen in the same picture; "
                    "the heading came from PX4 only (lined up at power-on).")
    log.info("FINAL TARGET POSES (arena frame; median -> drop outliers -> weighted mean, see tag_fusion.py):")
    for tid, f, method, first in rows:
        (x, y), (mx, my) = f["mean"], f["med"]
        log.info("  tag %d: x %+.3f y %+.3f m  (median %+.3f, %+.3f)  first seen t=%.1fs",
                 tid, x, y, mx, my, first)
        how = f["how"]
        rc = f.get("recorrected")
        if method == "tag":
            used = ", ".join(f"{k} ({n})" for k, n in sorted(how["by_known"].items(), key=lambda kv: -kv[1]))
            log.info("     method TAG: placed relative to known tag(s) in the same picture: %s", used)
            if how["odom"]:
                log.info("     (%d PX4-only sighting(s) not used - known-tag sightings are more accurate)",
                         how["odom"])
        else:
            if rc and (rc["both"] or rc["before"] or rc["after"]):
                log.info("     method ODOM, re-corrected after the flight: PX4 position corrected by the "
                         "known-tag fixes before AND after each sighting (median %.2f m from the nearest fix)",
                         f["sep"])
                log.info("     sightings with a fix before+after %d, before only %d, after only %d",
                         rc["both"], rc["before"], rc["after"])
            else:
                log.info("     method ODOM: never seen with a known tag; placed from PX4 position "
                         "corrected by the last known-tag fix (median %.2f m flown since that fix)", f["sep"])
        if rc:
            lx, ly = rc["live"]
            log.info("     after-flight correction%s moved it %.1f cm from the live estimate (%+.3f, %+.3f)",
                     " (heading from the two-tag checks)" if rc["heading"] or loc.heading_bursts else "",
                     math.hypot(x - lx, y - ly) * 100, lx, ly)
            if tid in arena.truth:
                tx, ty = arena.truth[tid]
                log.info("     live estimate vs truth: error %.1f cm", math.hypot(lx - tx, ly - ty) * 100)
        if how["too_far"]:
            log.info("     (%d sighting(s) over %.1f m from the camera not used)", how["too_far"], MAX_RANGE_M)
        log.info("     combined %d/%d sightings (outlier cut %.0f cm), %d viewpoint(s); weights favour "
                 "close, slow views with the tags near the picture centre", f["kept"], f["n"], f["thr"] * 100,
                 f["views"])
        log.info("     uncertainty ~%.1f cm -> %s", f["unc"] * 100,
                 "CONFIRMED" if f["confirmed"] else "unconfirmed (needs more sightings / viewpoints)")
        if tid in arena.truth:
            tx, ty = arena.truth[tid]
            log.info("     vs truth (%+.3f, %+.3f): error %.1f cm (fwd/back %+.1f, left/right %+.1f)",
                     tx, ty, math.hypot(x - tx, y - ty) * 100, (x - tx) * 100, (y - ty) * 100)
    missing = sorted(arena.targets - {r[0] for r in rows})
    if missing:
        log.info("  not found: %s", missing)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tag-localised takeoff -> route -> land mission")
    ap.add_argument("--drone-id", type=int, required=True, help="Flashed ESP32 CONFIG_DRONE_ID.")
    ap.add_argument("--arena", default=str(HERE / "tag_arena.yaml"), help="Tag layout and route.")
    ap.add_argument("--cam-pitch", type=float, default=60.0,
                    help="Camera tilt below horizontal, deg (90 = straight down).")
    ap.add_argument("--dry-run", action="store_true",
                    help="No flight: print the tag-based position (carry the drone by hand).")
    ap.add_argument("--trajectory",
                    help="Fly this t,x,y,z CSV (like send_trajectory.py) instead of the route. "
                         "Its x/y are taken as arena forward/right from the start point.")
    ap.add_argument("--traj-rotate", type=float, default=0.0,
                    help="Turn the trajectory by this many degrees (180 = flip it to point +x).")
    ap.add_argument("--takeoff-wait", type=float, default=TAKEOFF_WAIT_S,
                    help="Seconds from CMD_START to the first move.")
    ap.add_argument("--dt", type=float, default=0.1,
                    help="Trajectory upload sample period (s). 0.1 = the drone's 10 Hz loop; it interpolates "
                         "between points, and fewer upload packets survive the ESP's 6-packet UDP queue.")
    ap.add_argument("--max-speed", type=float, default=0.30,
                    help="Refuse faster trajectories (m/s).")
    ap.add_argument("--search", action="store_true",
                    help="Fly the search behaviour in the arena file's `search:` block "
                         "instead of the route (see search_patterns.py).")
    ap.add_argument("--confirm", action="store_true",
                    help="Ask for confirmation before takeoff.")
    ap.add_argument("--laptop-camera", action="store_true",
                    help="Detect on the laptop from the camera stream (full pose incl. heading, "
                         "5 fps) instead of using the drone's detections. tag_stream.py "
                         "cannot run at the same time.")
    ap.add_argument("--telem-port", type=int, default=5005)
    ap.add_argument("--cmd-port", type=int, default=5006)
    args = ap.parse_args()

    traj_pts, dt_ms = None, round(args.dt * 1000)
    if args.trajectory:
        traj_pts = load_trajectory(args.trajectory, dt_ms / 1000)
        R = rot2(math.radians(args.traj_rotate))
        traj_pts[:, :2] = (R @ traj_pts[:, :2].T).T
        vmax = np.linalg.norm(np.diff(traj_pts, axis=0), axis=1).max() * 1000 / dt_ms
        if vmax > args.max_speed + 1e-6:
            raise SystemExit(f"max speed {vmax:.3f} m/s exceeds --max-speed {args.max_speed}")
        if len(traj_pts) > TRAJ_MAX_PTS:
            raise SystemExit(f"{len(traj_pts)} points > {TRAJ_MAX_PTS}; use a larger --dt")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s", "%H:%M:%S")
    logging.basicConfig(level=logging.INFO)
    root = logging.getLogger()
    for h in root.handlers:
        h.setFormatter(fmt)
    if not args.dry_run:
        fh = logging.FileHandler(HERE / f"tag_mission_{stamp}.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    arena = Arena(Path(args.arena))
    if args.search and arena.search is None:
        raise SystemExit(f"{args.arena}: no 'search:' block for --search")
    if args.search and args.trajectory:
        raise SystemExit("use either --search or --trajectory, not both")
    if traj_pts is None and not args.search and not arena.route and not args.dry_run:
        raise SystemExit(f"{args.arena}: no route (or pass --trajectory)")
    # Laptop-side detection only with --laptop-camera (built now, not mid-flight).
    detector = Detector(tag_size_m=arena.tag_size) if args.laptop_camera else None
    drone_id = args.drone_id

    poses, corr = PoseHistory(), Correction()
    tracker = TelemetryTracker(drone_id)

    def on_telemetry(packet: TelemetryPacket, source_ip: str) -> None:
        tracker.callback(packet, source_ip)
        if packet.drone_id == drone_id:
            poses.add(time.monotonic(), packet.ned_x, packet.ned_y, packet.heading_rad)

    comms = CommsNode(listen_port=args.telem_port, cmd_port=args.cmd_port)
    comms.set_nav_tags([], {})           # goals are in PX4's own frame (start offset 0, 0)
    comms.on_telemetry(on_telemetry)

    def handle_interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_interrupt)

    loc = None
    start_sent = land_sent = aborted = False
    comms.start()
    try:
        log.info("Waiting for telemetry from drone %d...", drone_id)
        snap = tracker.wait_for_first_packet(CONNECT_TIMEOUT_S, args.telem_port, comms.known_drones)
        corr.assume_start(snap.packet.ned_x, snap.packet.ned_y, snap.packet.heading_rad)
        esp_ip = comms.known_drones()[drone_id]
        loc = TagLocaliser(esp_ip, drone_id, arena, poses, corr, detector,
                           r_body_cam(args.cam_pitch), HERE / f"tag_sightings_{stamp}.csv")
        loc.start()
        source = ("laptop camera stream (UDP 5009): do not run tag_stream.py"
                  if detector else "drone's own detections (UDP 5008): tag_stream.py may run")
        log.info("Tags from %s. Known tags %s, targets %s, camera pitch %.0f deg.",
                 source, sorted(arena.known), sorted(arena.targets), args.cam_pitch)

        if args.dry_run:
            log.info("DRY RUN - no flight. Carry the drone over the tags; Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
                pkt = tracker.latest().packet
                x, y, yaw = corr.to_arena(pkt.ned_x, pkt.ned_y, pkt.heading_rad)
                age = time.monotonic() - corr.last_fix if corr.have_fix else float("nan")
                log.info("arena x=%.2f y=%.2f heading %.0f deg | last fix %.1f s ago from %s | PX4 x=%.2f y=%.2f",
                         x, y, math.degrees(yaw), age, loc.last_fix_tags or "-", pkt.ned_x, pkt.ned_y)

        if args.confirm:
            what = args.trajectory or ("the search" if args.search else "the route")
            if input(f"Type GO-{drone_id} to take off and fly {what}: ").strip() != f"GO-{drone_id}":
                log.info("Confirmation did not match; nothing sent.")
                return 0

        CLOCK.start()                        # GO typed (or straight away without --confirm)
        log.info("TAKEOFF command sent (CMD_START)")
        send_command(comms, drone_id, CMD_START, "CMD_START")
        loc.mark_takeoff()
        start_sent = True
        time.sleep(args.takeoff_wait)

        CLOCK.mark("takeoff wait over, looking for a known tag")
        deadline = time.monotonic() + FIX_WAIT_S
        while not corr.have_fix and time.monotonic() < deadline:
            time.sleep(0.2)
        if not corr.have_fix:
            log.warning("No known tag seen after takeoff: assuming the drone took off at "
                        "(0, 0) facing +x until a landmark is seen.")

        if traj_pts is not None:
            if corr.have_fix:
                log.info("Centring over the start point (0, 0) using the tags...")
                CLOCK.mark("centring over the start")
                fly_to(comms, drone_id, tracker, corr, 0.0, 0.0, radius=0.10)
                time.sleep(1.0)              # settle before the trajectory locks its start
            if detector is None:
                CLOCK.mark("start heading check")
                heading_check(loc, "START", since=loc.t0)
            play_trajectory(comms, drone_id, tracker, corr, traj_pts, dt_ms)
            if detector is None:
                CLOCK.mark("end heading check")
                heading_check(loc, "END", since=time.monotonic())
            log.info("LAND command sent (CMD_LAND)")
            CLOCK.mark("LAND sent")
            send_command(comms, drone_id, CMD_LAND, "CMD_LAND", repeat=3)
            land_sent = True
            return 0

        if args.search:
            CLOCK.mark("search started")
            missed = run_search(comms, drone_id, tracker, corr, loc, arena, dt_ms, stamp)
            log.info("LAND command sent (CMD_LAND)")
            CLOCK.mark("LAND sent")
            send_command(comms, drone_id, CMD_LAND, "CMD_LAND", repeat=3)
            land_sent = True
            if missed:
                log.warning("%d search waypoint(s) not reached.", missed)
                return 1
            return 0

        missed = 0
        CLOCK.mark("route started")
        for i, (ax, ay) in enumerate(arena.route, start=1):
            log.info("Waypoint %d/%d: arena x=%.2f y=%.2f", i, len(arena.route), ax, ay)
            if not fly_to(comms, drone_id, tracker, corr, ax, ay):
                missed += 1
                send_command(comms, drone_id, CMD_HOLD, "CMD_HOLD")
            time.sleep(0.5)

        log.info("LAND command sent (CMD_LAND)")
        CLOCK.mark("LAND sent")
        send_command(comms, drone_id, CMD_LAND, "CMD_LAND", repeat=3)
        land_sent = True
        if missed:
            log.warning("%d of %d waypoints not reached.", missed, len(arena.route))
            return 1
        return 0
    except KeyboardInterrupt:
        aborted = True
        log.warning("Stopped by Ctrl+C.")
        return 130
    except Exception as exc:
        aborted = True
        log.error("Mission aborted: %s", exc)
        return 1
    finally:
        if aborted and start_sent and not land_sent:
            try:
                log.info("LAND command sent (fallback CMD_LAND)")
                CLOCK.mark("LAND sent (fallback, mission stopped early)")
                send_command(comms, drone_id, CMD_LAND, "fallback CMD_LAND", repeat=3)
            except Exception as exc:
                log.error("Could not send fallback LAND: %s", exc)
        if loc is not None:
            loc.stop()
            print_results(loc, arena)
        CLOCK.report()
        if detector is not None:
            detector.close()
        comms.stop()


if __name__ == "__main__":
    sys.exit(main())
