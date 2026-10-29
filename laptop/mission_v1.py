#!/usr/bin/env python3
"""
mission.py plus AprilTag detection on a background thread.

The flight is the same as mission.py: takeoff -> zig-zag path -> back to start
-> land.  While it flies, a second thread streams the drone camera, detects
tags with the firmware's detector (apriltag_host.py), converts each sighting
to the local (x = north, y = east) frame using the drone pose from telemetry,
prints every newly found tag with a timestamp, and writes all sightings to a
CSV.  A live camera window shows the detected tags (q or Esc closes it without
affecting the mission; --no-preview skips it).  A summary table is printed
after landing.

Example:
    python3 laptop/mission_v1.py --drone-id 3

Exit code: 0 all waypoints reached, 1 error or waypoint timeout, 130 Ctrl+C.
"""

import argparse
import bisect
import csv
import logging
import math
import select
import signal
import socket
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from apriltag_host import Detector, passes_gate
from camera_stream import COMMAND_PORT, KEEPALIVE_S, FrameAssembler, drain_socket
from comms import CommsNode
from mission import (
    ARRIVAL_RADIUS_M,
    ARRIVAL_TIMEOUT_S,
    CONNECT_TIMEOUT_S,
    STALE_TIMEOUT_S,
    TAKEOFF_WAIT_S,
    build_waypoints,
)
from protocol import (
    CAMERA_STREAM_PORT,
    CMD_HOLD,
    CMD_LAND,
    CMD_START,
    TelemetryPacket,
    build_camera_stream_command,
)
from simple_waypoint_mission import (
    TelemetryTracker,
    send_command,
    send_goto,
    wait_for_arrival,
)
from tag_stream import draw_tag

log = logging.getLogger("mission_v1")

CAMERA_FPS = 5            # keeps Wi-Fi free for MAVLink/telemetry
POSE_MAX_AGE_S = 0.5      # skip a sighting if no telemetry this close in time
PREVIEW_SCALE = 2.0       # live window size multiplier
PREVIEW_WINDOW = "Drone camera + AprilTags (mission_v1)"


def tag_in_local_frame(
    drone_x: float, drone_y: float, heading_rad: float,
    forward: float, right: float,
) -> tuple[float, float]:
    """Tag position (x north, y east) from its offset in the drone frame.

    heading_rad is the yaw from north, clockwise (0 = north, pi/2 = east).
    """
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    return (drone_x + forward * c - right * s,
            drone_y + forward * s + right * c)


class PoseHistory:
    """Recent drone poses from telemetry, looked up by time."""

    def __init__(self, maxlen: int = 300) -> None:
        self._poses: deque[tuple[float, float, float, float]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, t: float, x: float, y: float, heading: float) -> None:
        with self._lock:
            self._poses.append((t, x, y, heading))

    def at(self, t: float) -> tuple[float, float, float] | None:
        """(x, y, heading) of the telemetry sample nearest in time to t."""
        with self._lock:
            poses = list(self._poses)
        if not poses:
            return None
        times = [p[0] for p in poses]
        i = bisect.bisect_left(times, t)
        best = min(poses[max(0, i - 1):i + 1], key=lambda p: abs(p[0] - t))
        if abs(best[0] - t) > POSE_MAX_AGE_S:
            return None
        return best[1], best[2], best[3]


class TagLogger:
    """Streams the camera, detects tags and records where each one is."""

    def __init__(self, esp_ip: str, poses: PoseHistory, detector: Detector,
                 csv_path: Path, preview: bool = True) -> None:
        self.esp_ip = esp_ip
        self.poses = poses
        self.detector = detector
        self.csv_path = csv_path
        self.preview = preview
        self._window_shown = False
        self.t0 = time.monotonic()           # reset by mark_takeoff()
        self.sightings: dict[int, list[tuple[float, float]]] = {}
        self.first_seen: dict[int, float] = {}
        self.frames = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tag-logger", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def mark_takeoff(self) -> None:
        self.t0 = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def summary(self) -> list[tuple[int, int, float, float, float]]:
        """(id, sightings, median x, median y, seconds since takeoff of first sighting)."""
        with self._lock:
            return [
                (tag_id,
                 len(pts),
                 statistics.median(p[0] for p in pts),
                 statistics.median(p[1] for p in pts),
                 self.first_seen[tag_id])
                for tag_id, pts in sorted(self.sightings.items())
            ]

    def _run(self) -> None:
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csv_file = None
        try:
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            rx.bind(("0.0.0.0", CAMERA_STREAM_PORT))
            rx.setblocking(False)
            drone = (self.esp_ip, COMMAND_PORT)
            start_cmd = build_camera_stream_command(True, CAMERA_FPS)
            assembler = FrameAssembler()

            csv_file = self.csv_path.open("w", newline="")
            writer = csv.writer(csv_file)
            writer.writerow(["wall_time", "t_since_takeoff_s", "tag_id",
                             "tag_x_m", "tag_y_m", "drone_x_m", "drone_y_m",
                             "heading_rad", "fwd_m", "right_m", "down_m", "margin"])

            next_keepalive = 0.0
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= next_keepalive:        # drone streams only while it hears us
                    next_keepalive = now + KEEPALIVE_S
                    try:
                        tx.sendto(start_cmd, drone)
                    except OSError as exc:
                        log.warning("camera keepalive failed: %s", exc)

                select.select([rx], [], [], 0.05)
                frame, _, _ = drain_socket(rx, assembler, self.esp_ip)
                assembler.expire(time.monotonic())
                if frame is not None:
                    self._handle_frame(frame, writer)
                    csv_file.flush()
                if self.preview:
                    self._poll_window()      # keeps the window responsive
        except Exception as exc:     # the flight must carry on without the camera
            log.error("Tag logger stopped: %s", exc)
        finally:
            self._close_preview()
            try:
                tx.sendto(build_camera_stream_command(False), (self.esp_ip, COMMAND_PORT))
            except OSError:
                pass
            if csv_file is not None:
                csv_file.close()
            rx.close()
            tx.close()
            self.detector.close()

    def _handle_frame(self, frame, writer) -> None:
        # Capture time on the laptop clock: now minus the drone-side age.
        captured = time.monotonic() - frame.age_ms / 1000.0
        gray = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return
        self.frames += 1

        pose = self.poses.at(captured)
        if pose is None and not self.preview:
            return                           # nothing to record, nothing to draw
        detections = [d for d in self.detector.detect(gray) if passes_gate(d)]

        if pose is not None:
            drone_x, drone_y, heading = pose
            for det in detections:
                if det.t_body is None:
                    continue
                fwd, right, down = det.t_body
                tag_x, tag_y = tag_in_local_frame(drone_x, drone_y, heading, fwd, right)
                since = captured - self.t0
                self._record(det.id, tag_x, tag_y, since)
                writer.writerow([time.strftime("%H:%M:%S"), f"{since:.2f}", det.id,
                                 f"{tag_x:.3f}", f"{tag_y:.3f}",
                                 f"{drone_x:.3f}", f"{drone_y:.3f}", f"{heading:.3f}",
                                 f"{fwd:.3f}", f"{right:.3f}", f"{down:.3f}",
                                 f"{det.decision_margin:.1f}"])

        if self.preview:
            self._show(gray, detections, pose, frame.age_ms)

    def _show(self, gray, detections, pose, age_ms: int) -> None:
        """Camera frame with every accepted tag outlined, plus a status line."""
        try:
            h, w = gray.shape
            size = (round(w * PREVIEW_SCALE), round(h * PREVIEW_SCALE))
            view = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), size,
                              interpolation=cv2.INTER_NEAREST)
            for det in detections:
                draw_tag(view, det, PREVIEW_SCALE)

            if pose is None:
                status = "no telemetry"
            else:
                status = f"x {pose[0]:+.2f}  y {pose[1]:+.2f}  yaw {math.degrees(pose[2]):.0f}"
            with self._lock:
                found = len(self.sightings)
            status += f"   tags found {found}   cam {age_ms} ms"
            for colour, thickness in (((0, 0, 0), 3), ((0, 255, 255), 1)):   # outlined text
                cv2.putText(view, status, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colour, thickness, cv2.LINE_AA)
            cv2.imshow(PREVIEW_WINDOW, view)
            self._window_shown = True
        except cv2.error as exc:             # no display available: keep detecting
            log.warning("Live preview disabled: %s", exc)
            self.preview = False

    def _poll_window(self) -> None:
        """Process window events; q, Esc or closing the window turns the preview off."""
        try:
            key = cv2.waitKey(1) & 0xFF
            closed = (self._window_shown and
                      cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1)
            if key in (ord("q"), 27) or closed:
                log.info("Live preview closed (tag logging continues).")
                self._close_preview()
        except cv2.error:
            self.preview = False

    def _close_preview(self) -> None:
        self.preview = False
        if self._window_shown:
            self._window_shown = False
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except cv2.error:
                pass

    def _record(self, tag_id: int, x: float, y: float, since: float) -> None:
        with self._lock:
            new = tag_id not in self.sightings
            self.sightings.setdefault(tag_id, []).append((x, y))
            if new:
                self.first_seen[tag_id] = since
        if new:
            log.info("TAG %d detected at t=%.1fs: local position x=%.2f y=%.2f m",
                     tag_id, since, x, y)


def print_summary(tag_logger: TagLogger) -> None:
    rows = tag_logger.summary()
    log.info("Camera frames processed: %d", tag_logger.frames)
    if not rows:
        log.info("No AprilTags detected.")
        return
    log.info("AprilTags found: %d", len(rows))
    log.info("  %-4s %-10s %-9s %-9s %s", "ID", "sightings", "x (m)", "y (m)", "first seen")
    for tag_id, count, x, y, first in rows:
        log.info("  %-4d %-10d %-9.2f %-9.2f t=%.1fs", tag_id, count, x, y, first)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fixed-path mission with AprilTag detection on a second thread")
    parser.add_argument("--drone-id", type=int, required=True,
                        help="Flashed ESP32 CONFIG_DRONE_ID.")
    parser.add_argument("--no-preview", action="store_true",
                        help="Do not open the live camera window.")
    parser.add_argument("--telem-port", type=int, default=5005)
    parser.add_argument("--cmd-port", type=int, default=5006)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    waypoints = build_waypoints()
    drone_id = args.drone_id

    # Build the detector now so a missing compiler stops us before takeoff.
    detector = Detector()

    poses = PoseHistory()
    tracker = TelemetryTracker(drone_id)

    def on_telemetry(packet: TelemetryPacket, source_ip: str) -> None:
        tracker.callback(packet, source_ip)
        if packet.drone_id == drone_id:
            poses.add(time.monotonic(), packet.ned_x, packet.ned_y, packet.heading_rad)

    comms = CommsNode(listen_port=args.telem_port, cmd_port=args.cmd_port)
    # Start offset (0, 0) so waypoints are relative to the takeoff point.
    comms.set_nav_tags([], {})
    comms.on_telemetry(on_telemetry)

    def handle_interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_interrupt)

    tag_logger: TagLogger | None = None
    start_sent = land_sent = aborted = False
    comms.start()
    try:
        log.info("Waiting for telemetry from drone %d...", drone_id)
        tracker.wait_for_first_packet(CONNECT_TIMEOUT_S, args.telem_port, comms.known_drones)
        log.info("Telemetry link established.")

        # Camera goes to the drone's address, taken from its telemetry packets.
        esp_ip = comms.known_drones()[drone_id]
        csv_path = Path(__file__).with_name(time.strftime("tags_%Y%m%d_%H%M%S.csv"))
        tag_logger = TagLogger(esp_ip, poses, detector, csv_path,
                               preview=not args.no_preview)
        tag_logger.start()
        log.info("Tag logger started (camera %s, log %s).", esp_ip, csv_path.name)

        log.info("Sending CMD_START (arm + takeoff)")
        send_command(comms, drone_id, CMD_START, "CMD_START")
        tag_logger.mark_takeoff()
        start_sent = True
        time.sleep(TAKEOFF_WAIT_S)

        missed = 0
        for index, (goal_x, goal_y) in enumerate(waypoints, start=1):
            log.info("Sending waypoint %d/%d to (%.2f, %.2f)",
                     index, len(waypoints), goal_x, goal_y)

            def resend(gx: float = goal_x, gy: float = goal_y) -> None:
                send_goto(comms, drone_id, gx, gy)

            resend()
            if not wait_for_arrival(
                tracker,
                resend,
                timeout_s=ARRIVAL_TIMEOUT_S,
                stale_timeout_s=STALE_TIMEOUT_S,
                goal_x=goal_x,
                goal_y=goal_y,
                arrival_radius_m=ARRIVAL_RADIUS_M,
            ):
                missed += 1
                send_command(comms, drone_id, CMD_HOLD, "CMD_HOLD")   # stop before next leg
            time.sleep(0.5)

        log.info("Sending CMD_LAND")
        send_command(comms, drone_id, CMD_LAND, "CMD_LAND", repeat=3)
        land_sent = True

        if missed:
            log.warning("%d of %d waypoints not reached.", missed, len(waypoints))
            return 1
        return 0
    except KeyboardInterrupt:
        aborted = True
        log.warning("Mission interrupted.")
        return 130
    except Exception as exc:
        aborted = True
        log.error("Mission aborted: %s", exc)
        return 1
    finally:
        if aborted and start_sent and not land_sent:
            try:
                send_command(comms, drone_id, CMD_LAND, "fallback CMD_LAND", repeat=3)
            except Exception as exc:
                log.error("Could not send fallback LAND: %s", exc)
        if tag_logger is not None:
            tag_logger.stop()
            print_summary(tag_logger)
        else:
            detector.close()
        comms.stop()


if __name__ == "__main__":
    sys.exit(main())
