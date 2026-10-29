import argparse
import logging
import signal
import sys
import time

from comms import CommsNode
from protocol import CMD_HOLD, CMD_LAND, CMD_START
from simple_waypoint_mission import (
    TelemetryTracker,
    send_command,
    send_goto,
    wait_for_arrival,
)

log = logging.getLogger("mission")

# Legs as (forward, right) steps in metres; the waypoints are their running sum.
LEGS = [
    (0.8, 0.0),     # forward 0.8
    (0.0, -0.4),    # left 0.4
    (0.8, 0.0),     # forward 0.8
    (0.0, 0.8),     # right 0.8
    (-0.8, 0.0),    # back 0.8
    (0.0, -0.4),    # left 0.4
]
HOME = (0.0, 0.0)

TAKEOFF_WAIT_S = 10.0     # wait after CMD_START before the first waypoint
ARRIVAL_TIMEOUT_S = 25.0  # give up on a waypoint after this long
ARRIVAL_RADIUS_M = 0.25   # a waypoint counts as reached within this distance
CONNECT_TIMEOUT_S = 30.0  # wait for the first telemetry packet
STALE_TIMEOUT_S = 3.0     # land if telemetry is older than this


def build_waypoints() -> list[tuple[float, float]]:
    x, y = HOME
    waypoints = []
    for dx, dy in LEGS:
        x, y = x + dx, y + dy
        waypoints.append((x, y))
    waypoints.append(HOME)
    return waypoints


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed-path takeoff -> zig-zag -> home -> land")
    parser.add_argument("--drone-id", type=int, required=True,
                        help="Flashed ESP32 CONFIG_DRONE_ID.")
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

    tracker = TelemetryTracker(drone_id)
    comms = CommsNode(listen_port=args.telem_port, cmd_port=args.cmd_port)
    # Start offset (0, 0) so waypoints are relative to the takeoff point.
    comms.set_nav_tags([], {})
    comms.on_telemetry(tracker.callback)

    def handle_interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_interrupt)

    start_sent = land_sent = aborted = False
    comms.start()
    try:
        log.info("Waiting for telemetry from drone %d...", drone_id)
        tracker.wait_for_first_packet(CONNECT_TIMEOUT_S, args.telem_port, comms.known_drones)
        log.info("Telemetry link established.")

        log.info("Sending CMD_START (arm + takeoff)")
        send_command(comms, drone_id, CMD_START, "CMD_START")
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
        comms.stop()


if __name__ == "__main__":
    sys.exit(main())
