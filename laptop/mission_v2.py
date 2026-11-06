#!/usr/bin/env python3
"""
mission_v1.py with a different flight: forward -> slow 360 deg turn -> back.

    takeoff -> fly FORWARD m straight ahead (the way the drone faces at the
    start) -> turn a full 360 deg on the spot, slowly -> fly back to the start
    -> land

Tag detection, the live camera window, the sightings CSV and the summary are
the same as mission_v1.py (its TagLogger is reused).

The turn uses CMD_YAW (firmware: hold position, turn to a heading), stepped
round TURN_STEP_DEG at a time with TURN_STEP_S between steps, so one lap takes
about 360 / TURN_STEP_DEG * TURN_STEP_S seconds. Needs firmware with CMD_YAW
(main/wifi_task.c); older firmware ignores it and the drone just hovers.

Example:
    python laptop/mission_v2.py --drone-id 3
    python laptop/mission_v2.py --drone-id 3 --forward 2.25 --turn-step 15 --step-time 1.5

Exit code: 0 done, 1 error or waypoint timeout, 130 Ctrl+C.
"""

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path

from apriltag_host import Detector
from comms import CommsNode
from mission import ARRIVAL_RADIUS_M, CONNECT_TIMEOUT_S, STALE_TIMEOUT_S, TAKEOFF_WAIT_S
from mission_v1 import PoseHistory, TagLogger, print_summary
from protocol import CMD_HOLD, CMD_LAND, CMD_START, CMD_YAW, CommandPacket, TelemetryPacket
from simple_waypoint_mission import TelemetryTracker, send_command, send_goto, wait_for_arrival

log = logging.getLogger("mission_v2")

FORWARD_M = 2.25          # distance flown straight ahead
TURN_STEP_DEG = 15.0      # heading step per CMD_YAW
TURN_STEP_S = 1.5         # wait between steps -> 360 deg in ~36 s
TURN_SETTLE_S = 5.0       # after the last step, wait up to this long to face the start heading
TURN_TOL_DEG = 10.0       # "back to the start heading" when this close
LEG_TIMEOUT_S = 40.0      # give up on a leg after this long


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def send_yaw(comms: CommsNode, drone_id: int, heading_rad: float) -> None:
    """Hold position and turn to heading_rad (NED, CW from north). Sent twice: UDP."""
    for _ in range(2):
        if not comms.send_command(drone_id, CommandPacket(CMD_YAW, goal_x=wrap(heading_rad))):
            raise RuntimeError(f"Failed to send CMD_YAW: drone {drone_id} IP is unknown.")
        time.sleep(0.05)


def fly_to(comms, drone_id, tracker, x: float, y: float, name: str) -> bool:
    log.info("%s: CMD_GOTO (%.2f, %.2f)", name, x, y)

    def resend(gx: float = x, gy: float = y) -> None:
        send_goto(comms, drone_id, gx, gy)

    resend()
    ok = wait_for_arrival(tracker, resend, timeout_s=LEG_TIMEOUT_S, stale_timeout_s=STALE_TIMEOUT_S,
                          goal_x=x, goal_y=y, arrival_radius_m=ARRIVAL_RADIUS_M)
    if not ok:
        log.warning("%s: not reached in %.0f s, holding.", name, LEG_TIMEOUT_S)
        send_command(comms, drone_id, CMD_HOLD, "CMD_HOLD")
    return ok


def turn_360(comms, drone_id, tracker, step_deg: float, step_s: float) -> float:
    """Step the heading round one full lap (clockwise). Returns degrees actually turned."""
    def heading() -> float:
        snap = tracker.latest()
        if time.monotonic() - snap.received_at > STALE_TIMEOUT_S:
            raise ConnectionError("Telemetry is stale.")
        return snap.packet.heading_rad

    start = last = heading()
    turned = 0.0
    steps = max(1, math.ceil(360.0 / step_deg))
    log.info("TURN: 360 deg in %d steps of %.0f deg, ~%.0f s, starting at heading %.0f deg",
             steps, 360.0 / steps, steps * step_s, math.degrees(start))
    for k in range(1, steps + 1):
        send_yaw(comms, drone_id, start + math.radians(360.0 * k / steps))
        t_end = time.monotonic() + step_s
        while time.monotonic() < t_end:
            h = heading()
            turned += math.degrees(wrap(h - last))
            last = h
            time.sleep(0.1)
        if k % max(1, steps // 4) == 0:
            log.info("TURN: %3.0f / 360 deg", turned)
    t_end = time.monotonic() + TURN_SETTLE_S        # let it finish the last step
    while time.monotonic() < t_end and abs(math.degrees(wrap(heading() - start))) > TURN_TOL_DEG:
        h = heading()
        turned += math.degrees(wrap(h - last))
        last = h
        time.sleep(0.1)
    h = heading()
    turned += math.degrees(wrap(h - last))
    log.info("TURN done: turned %.0f deg, heading %.0f deg (start %.0f)",
             turned, math.degrees(h), math.degrees(start))
    if abs(turned) < 300:
        log.warning("Turned only %.0f deg - is the firmware flashed with CMD_YAW?", turned)
    return turned


def main() -> int:
    ap = argparse.ArgumentParser(description="Forward -> slow 360 turn -> back, with AprilTag logging")
    ap.add_argument("--drone-id", type=int, required=True, help="Flashed ESP32 CONFIG_DRONE_ID.")
    ap.add_argument("--forward", type=float, default=FORWARD_M, help="Metres straight ahead (default 2.25).")
    ap.add_argument("--turn-step", type=float, default=TURN_STEP_DEG, help="Degrees per turn step.")
    ap.add_argument("--step-time", type=float, default=TURN_STEP_S, help="Seconds per turn step.")
    ap.add_argument("--no-preview", action="store_true", help="Do not open the live camera window.")
    ap.add_argument("--telem-port", type=int, default=5005)
    ap.add_argument("--cmd-port", type=int, default=5006)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    drone_id = args.drone_id
    detector = Detector()                 # built now so a missing compiler stops us before takeoff

    poses = PoseHistory()
    tracker = TelemetryTracker(drone_id)

    def on_telemetry(packet: TelemetryPacket, source_ip: str) -> None:
        tracker.callback(packet, source_ip)
        if packet.drone_id == drone_id:
            poses.add(time.monotonic(), packet.ned_x, packet.ned_y, packet.heading_rad)

    comms = CommsNode(listen_port=args.telem_port, cmd_port=args.cmd_port)
    comms.set_nav_tags([], {})            # goals in PX4's own frame (start offset 0, 0)
    comms.on_telemetry(on_telemetry)

    def handle_interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_interrupt)

    tag_logger = None
    start_sent = land_sent = aborted = False
    comms.start()
    try:
        log.info("Waiting for telemetry from drone %d...", drone_id)
        snap = tracker.wait_for_first_packet(CONNECT_TIMEOUT_S, args.telem_port, comms.known_drones)
        home_x, home_y, home_h = snap.packet.ned_x, snap.packet.ned_y, snap.packet.heading_rad
        out_x = home_x + args.forward * math.cos(home_h)
        out_y = home_y + args.forward * math.sin(home_h)
        log.info("Start (%.2f, %.2f) facing %.0f deg -> forward point (%.2f, %.2f)",
                 home_x, home_y, math.degrees(home_h), out_x, out_y)

        esp_ip = comms.known_drones()[drone_id]
        csv_path = Path(__file__).with_name(time.strftime("tags_v2_%Y%m%d_%H%M%S.csv"))
        tag_logger = TagLogger(esp_ip, poses, detector, csv_path, preview=not args.no_preview)
        tag_logger.start()
        log.info("Tag logger started (camera %s, log %s).", esp_ip, csv_path.name)

        log.info("TAKEOFF command sent (CMD_START)")
        send_command(comms, drone_id, CMD_START, "CMD_START")
        tag_logger.mark_takeoff()
        start_sent = True
        time.sleep(TAKEOFF_WAIT_S)

        missed = 0
        if not fly_to(comms, drone_id, tracker, out_x, out_y, f"FORWARD {args.forward:.2f} m"):
            missed += 1
        time.sleep(1.5)                   # settle before turning
        turn_360(comms, drone_id, tracker, args.turn_step, args.step_time)
        if not fly_to(comms, drone_id, tracker, home_x, home_y, "BACK to start"):
            missed += 1

        log.info("LAND command sent (CMD_LAND)")
        send_command(comms, drone_id, CMD_LAND, "CMD_LAND", repeat=3)
        land_sent = True
        return 1 if missed else 0
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
                log.info("LAND command sent (fallback CMD_LAND)")
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
