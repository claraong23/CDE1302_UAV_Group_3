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
lined up at takeoff assuming the drone faces +x. --laptop-camera detects on
the laptop instead (full pose, but then tag_stream.py cannot run).

The camera is assumed to look straight down (--cam-pitch 90) with the top of
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
import statistics
import sys
import threading
import time
import types
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
from search_patterns import SearchPlan, smooth_trajectory
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
        self.tag_size = float(cfg.get("tag_size_m", 0.12))
        self.known = {}              # id -> (position (3,), tag->arena rotation)
        for tid, (x, y, yaw) in (cfg.get("known_tags") or {}).items():
            self.known[int(tid)] = (np.array([x, y, 0.0]), rot_z(math.radians(yaw)) @ FLOOR)
        self.targets = {int(t) for t in cfg.get("targets") or []}
        self.route = [(float(x), float(y)) for x, y in cfg.get("route") or []]
        self.search = SearchPlan(cfg["search"]) if cfg.get("search") else None
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


def drone_from_known(R_bt, t_b, P, M):
    """Drone position (3,) and body->arena rotation from one known tag."""
    R_ab = M @ R_bt.T
    return P - R_ab @ t_b, R_ab


def heading_of(R_ab: np.ndarray) -> float:
    return math.atan2(R_ab[1, 0], R_ab[0, 0])


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

    def update(self, fix_xy, fix_yaw: float, odom) -> bool:
        """Blend in a tag fix taken at PX4 pose odom = (x, y, yaw). False if rejected."""
        ox, oy, oyaw = odom
        new_dyaw = wrap(fix_yaw - oyaw)
        new_T = np.asarray(fix_xy) - rot2(new_dyaw) @ np.array([ox, oy])
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
        self.sightings: dict[int, list[tuple[float, float, str]]] = {}
        self.first_seen: dict[int, float] = {}
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
        """(id, x, y, sightings, method, first seen s) - 'tag' sightings preferred."""
        out = []
        with self._lock:
            for tid, pts in sorted(self.sightings.items()):
                best = [p for p in pts if p[2] == "tag"] or pts
                out.append((tid, statistics.median(p[0] for p in best),
                            statistics.median(p[1] for p in best),
                            len(pts), best[0][2], self.first_seen[tid]))
        return out

    def _open_csv(self):
        f = self.csv_path.open("w", newline="")
        writer = csv.writer(f)
        writer.writerow(["wall_time", "t_since_takeoff_s", "tag_id", "x_m", "y_m",
                         "method", "drone_x_m", "drone_y_m", "margin"])
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
                                              range_m=d.range_m, decision_margin=d.margin)
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

        # 1. Drone pose from every known tag in view (closer = more weight).
        refs = []                                # (drone position, R_ab, weight)
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
        drone_xy = None
        if refs:
            w = np.array([r[2] for r in refs])
            w /= w.sum()
            drone_xy = sum(wi * r[0][:2] for wi, r in zip(w, refs))
            yaw = math.atan2(sum(wi * math.sin(heading_of(r[1])) for wi, r in zip(w, refs)),
                             sum(wi * math.cos(heading_of(r[1])) for wi, r in zip(w, refs)))
            ids = sorted({d.id for d in dets if d.id in self.arena.known})
            if odom is not None:
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
            if refs:                              # same picture as a known tag: no drift
                w = np.array([r[2] for r in refs])
                est = sum(wi * (r[0] + r[1] @ t_b) for wi, r in zip(w / w.sum(), refs))
                x, y, method, dx, dy = est[0], est[1], "tag", drone_xy[0], drone_xy[1]
            elif odom is not None:                # corrected PX4 position + offset
                dx, dy, yaw = self.corr.to_arena(*odom)
                c, s = math.cos(yaw), math.sin(yaw)
                x = dx + t_b[0] * c - t_b[1] * s
                y = dy + t_b[0] * s + t_b[1] * c
                method = "odom"
            else:
                continue
            since = captured - self.t0
            self._record(d.id, x, y, method, since)
            writer.writerow([time.strftime("%H:%M:%S"), f"{since:.2f}", d.id, f"{x:.3f}",
                             f"{y:.3f}", method, f"{dx:.3f}", f"{dy:.3f}",
                             f"{d.decision_margin:.1f}"])

    def _record(self, tid, x, y, method, since):
        with self._lock:
            new = tid not in self.sightings
            self.sightings.setdefault(tid, []).append((x, y, method))
            if new:
                self.first_seen[tid] = since
        if new:
            log.info("TARGET %d FOUND at x=%.2f y=%.2f m (arena frame, %s) t=%.1f s",
                     tid, x, y, method, since)


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
    log.info("Trajectory ended at arena x=%.2f y=%.2f; planned end x=%.2f y=%.2f (off by %.2f m)",
             x, y, ex, ey, math.hypot(x - ex, y - ey))
    return False


# ---------------------------------------------------------------------------
# Search behaviour (--search, configured in the arena file's `search:` block)
# ---------------------------------------------------------------------------

def run_search(comms, drone_id, tracker, corr: Correction, loc, arena: Arena,
               dt_ms: int) -> int:
    """entry -> pattern -> decision -> (next pattern) -> exit. Returns waypoints missed."""
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

    stop = all_found if plan.stop_early else None
    missed = 0
    log.info("SEARCH: %s mode, %d pattern(s), expecting %d target(s), max %d loop(s)",
             plan.mode, len(plan.patterns), expected, plan.max_loops)
    for x, y in plan.entry:                     # drift-corrected, like the route
        if not goto(x, y):
            missed += 1
    for loop in range(plan.max_loops):
        name, pts = plan.patterns[loop % len(plan.patterns)]
        log.info("SEARCH pattern '%s' (loop %d/%d)", name, loop + 1, plan.max_loops)
        if not goto(*pts[0], radius=0.10):      # start exactly on the decision corner
            missed += 1
        time.sleep(1.0)                         # settle before a trajectory locks its start
        if plan.mode == "trajectory":
            snap = tracker.latest().packet
            sx, sy, _ = corr.to_arena(snap.ned_x, snap.ned_y, snap.heading_rad)
            prof = smooth_trajectory([(sx, sy)] + pts[1:], plan.speed, dt_ms / 1000.0)
            offsets = np.column_stack([prof["xy"] - prof["xy"][0], np.zeros(len(prof["xy"]))])
            log.info("  smooth path %.1f m, %.0f s, %d points", prof["length"], prof["duration"], len(offsets))
            play_trajectory(comms, drone_id, tracker, corr, offsets,
                            int(round(prof["dt"] * 1000)), stop_when=stop)
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


def print_results(loc: TagLocaliser, arena: Arena) -> None:
    log.info("Frames %d, tag fixes %d (%d ignored).", loc.frames, loc.fixes, loc.rejected)
    rows = loc.results()
    log.info("TARGETS (arena frame, median of sightings):")
    log.info("  %-4s %-8s %-8s %-10s %-7s %s", "ID", "x (m)", "y (m)", "sightings", "method", "first seen")
    for tid, x, y, n, method, first in rows:
        log.info("  %-4d %-8.2f %-8.2f %-10d %-7s t=%.1fs", tid, x, y, n, method, first)
    missing = sorted(arena.targets - {r[0] for r in rows})
    if missing:
        log.info("  not found: %s", missing)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tag-localised takeoff -> route -> land mission")
    ap.add_argument("--drone-id", type=int, required=True, help="Flashed ESP32 CONFIG_DRONE_ID.")
    ap.add_argument("--arena", default=str(HERE / "tag_arena.yaml"), help="Tag layout and route.")
    ap.add_argument("--cam-pitch", type=float, default=90.0,
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
    ap.add_argument("--dt", type=float, default=0.05, help="Trajectory upload sample period (s).")
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

        log.info("TAKEOFF command sent (CMD_START)")
        send_command(comms, drone_id, CMD_START, "CMD_START")
        loc.mark_takeoff()
        start_sent = True
        time.sleep(args.takeoff_wait)

        deadline = time.monotonic() + FIX_WAIT_S
        while not corr.have_fix and time.monotonic() < deadline:
            time.sleep(0.2)
        if not corr.have_fix:
            log.warning("No known tag seen after takeoff: assuming the drone took off at "
                        "(0, 0) facing +x until a landmark is seen.")

        if traj_pts is not None:
            if corr.have_fix:
                log.info("Centring over the start point (0, 0) using the tags...")
                fly_to(comms, drone_id, tracker, corr, 0.0, 0.0, radius=0.10)
                time.sleep(1.0)              # settle before the trajectory locks its start
            play_trajectory(comms, drone_id, tracker, corr, traj_pts, dt_ms)
            log.info("LAND command sent (CMD_LAND)")
            send_command(comms, drone_id, CMD_LAND, "CMD_LAND", repeat=3)
            land_sent = True
            return 0

        if args.search:
            missed = run_search(comms, drone_id, tracker, corr, loc, arena, dt_ms)
            log.info("LAND command sent (CMD_LAND)")
            send_command(comms, drone_id, CMD_LAND, "CMD_LAND", repeat=3)
            land_sent = True
            if missed:
                log.warning("%d search waypoint(s) not reached.", missed)
                return 1
            return 0

        missed = 0
        for i, (ax, ay) in enumerate(arena.route, start=1):
            log.info("Waypoint %d/%d: arena x=%.2f y=%.2f", i, len(arena.route), ax, ay)
            if not fly_to(comms, drone_id, tracker, corr, ax, ay):
                missed += 1
                send_command(comms, drone_id, CMD_HOLD, "CMD_HOLD")
            time.sleep(0.5)

        log.info("LAND command sent (CMD_LAND)")
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
                send_command(comms, drone_id, CMD_LAND, "fallback CMD_LAND", repeat=3)
            except Exception as exc:
                log.error("Could not send fallback LAND: %s", exc)
        if loc is not None:
            loc.stop()
            print_results(loc, arena)
        if detector is not None:
            detector.close()
        comms.stop()


if __name__ == "__main__":
    sys.exit(main())
