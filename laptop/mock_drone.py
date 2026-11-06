#!/usr/bin/env python3
"""
Fake ESP32 for testing the mission scripts without hardware.

Speaks the lab's UDP protocol (laptop/protocol.py): telemetry on 5005 and
AprilTag detections on 5008 at 10 Hz, commands on 5006. Supports CMD_START,
CMD_GOTO (rotate then fly), CMD_HOLD, CMD_LAND and trajectory upload/playback
(CMD_TRAJ_DATA / CMD_TRAJ_START, heading held, like nav_task.c).

Ground-truth tags are given in the ARENA frame (launch point, x forward,
y right) and converted with --heading-deg / --start-n / --start-e, so you can
test a drone that is not at the NED origin or not facing north.

    python laptop/mock_drone.py --drone-id 2          # lab defaults (60 deg camera)
    python laptop/mock_drone.py --drone-id 3 --pitch-deg 90 \
        --tags "1:0,0;2:0.65,-1.35;8:4.75,-1.35;9:0.65,0.90;4:1.37,-1.32;5:4.03,-1.38;6:2.72,0.93"
    # then in a second terminal:
    python laptop/tag_mission.py --drone-id 3 --search
"""
import argparse
import math
import random
import socket
import struct
import time

from protocol import (CMD_GOTO, CMD_HOLD, CMD_LAND, CMD_START, CMD_TRAJ_DATA, CMD_TRAJ_START,
                      NAV_ARRIVED, NAV_FLYING, NAV_IDLE, NAV_ROTATING, NAV_TRAJ,
                      PKT_AT_DEBUG, PKT_CMD, PKT_TELEM)

# Three of the six candidate spots in the briefing figure, a few cm off (as if placed by hand)
TRUE_TAGS_ARENA = {3: (1.37, -1.32), 5: (4.03, -1.38), 8: (2.72, 0.93)}
FX, FY, CX, CY, W, H = 175.57, 174.59, 157.16, 116.21, 320, 240   # apriltag_host.FIRMWARE_INTRINSICS
CAM_PITCH = math.radians(60.0)   # overridden by --pitch-deg
CAM_POS = (0.02, 0.0, 0.0)
ALT = 0.5
SPEED, TURN = 0.4, 1.5       # CMD_GOTO cruise m/s, rotate rad/s
TAU = 0.25                   # s, first-order lag of position tracking


def body_to_cam(b):
    s, c = math.sin(CAM_PITCH), math.cos(CAM_PITCH)
    R = ((0, -s, c), (1, 0, 0), (0, c, s))
    v = [b[i] - CAM_POS[i] for i in range(3)]
    return tuple(sum(R[i][j] * v[i] for i in range(3)) for j in range(3))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone-id", type=int, default=2)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--noise", type=float, default=0.02, help="tag translation noise std (m)")
    ap.add_argument("--heading-deg", type=float, default=0.0, help="drone heading at launch (NED yaw)")
    ap.add_argument("--start-n", type=float, default=0.0, help="launch position north in PX4 local frame")
    ap.add_argument("--start-e", type=float, default=0.0)
    ap.add_argument("--drift", type=float, default=0.05, help="random hover drift after take-off (m)")
    ap.add_argument("--pitch-deg", type=float, default=60.0, help="camera tilt below horizontal (90 = down)")
    ap.add_argument("--tags", help="ground-truth tags in the arena frame, e.g. '1:0,0;4:1.35,-1.35' "
                                   "(default: three targets at candidate spots)")
    ap.add_argument("--traj-bias", type=float, default=0.0,
                    help="shift trajectory playback north by this much (m) to test the return correction")
    args = ap.parse_args()

    global CAM_PITCH
    CAM_PITCH = math.radians(args.pitch_deg)
    truth = TRUE_TAGS_ARENA
    if args.tags:
        truth = {int(k): tuple(map(float, v.split(","))) for k, v in
                 (item.split(":") for item in args.tags.split(";") if item.strip())}
    h0 = math.radians(args.heading_deg)
    tags = {tid: (args.start_n + x * math.cos(h0) - y * math.sin(h0),
                  args.start_e + x * math.sin(h0) + y * math.cos(h0))
            for tid, (x, y) in truth.items()}

    cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    cmd.bind(("0.0.0.0", 5006))
    cmd.setblocking(False)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    x, y, h = args.start_n, args.start_e, h0      # actual pose
    sx, sy = x, y                                 # position setpoint
    state, airborne, takeoff_at, goal = NAV_IDLE, False, None, None
    traj = {"id": 0, "n": 0, "pts": {}}
    play = None                                    # (origin, pts, dt, t0)
    dt = 0.1
    print(f"mock drone {args.drone_id}; true tags (NED): "
          + ", ".join(f"{k}:({v[0]:.2f},{v[1]:.2f})" for k, v in tags.items()))
    while True:
        t_loop = time.time()
        # ---- commands -----------------------------------------------------
        try:
            while True:
                data, _ = cmd.recvfrom(2048)
                if len(data) < 2 or data[0] != PKT_CMD:
                    continue
                ctype = data[1]
                if len(data) == 22 and ctype in (CMD_START, CMD_GOTO, CMD_HOLD, CMD_LAND):
                    _, _, gx, gy, *_ = struct.unpack("<BBff12b", data)
                    if ctype == CMD_START and not airborne and takeoff_at is None:
                        takeoff_at = time.time() + 3.0
                        print("CMD_START")
                    elif ctype == CMD_GOTO and airborne and state != NAV_TRAJ:
                        goal = (gx, gy)
                    elif ctype == CMD_HOLD:
                        goal, play, state, sx, sy = None, None, NAV_IDLE, x, y
                        print("CMD_HOLD")
                    elif ctype == CMD_LAND:
                        goal, play, airborne, state = None, None, False, NAV_IDLE
                        print(f"CMD_LAND at ({x:.2f}, {y:.2f})")
                elif ctype == CMD_TRAJ_DATA:
                    tid, total, off, n = struct.unpack_from("<BHHB", data, 2)
                    if tid != traj["id"] or total != traj["n"]:
                        traj = {"id": tid, "n": total, "pts": {}}
                    for k in range(n):
                        traj["pts"][off + k] = struct.unpack_from("<fff", data, 8 + 12 * k)
                elif ctype == CMD_TRAJ_START:
                    tid, dt_ms = struct.unpack_from("<BH", data, 2)
                    if airborne and state != NAV_TRAJ and tid == traj["id"] and len(traj["pts"]) == traj["n"]:
                        pts = [traj["pts"][i] for i in range(traj["n"])]
                        play = ((x + args.traj_bias, y), pts, dt_ms / 1000.0, time.time())
                        traj["id"] = 0
                        state, goal = NAV_TRAJ, None
                        print(f"TRAJ start: {len(pts)} pts, {dt_ms} ms, heading held {math.degrees(h):.1f} deg")
        except BlockingIOError:
            pass
        if takeoff_at and time.time() >= takeoff_at and not airborne:
            airborne, takeoff_at, state = True, None, NAV_IDLE
            sx = x + random.uniform(-args.drift, args.drift)
            sy = y + random.uniform(-args.drift, args.drift)
            print("airborne")
        # ---- motion -------------------------------------------------------
        if airborne and play:
            (ox, oy), pts, pdt, t0 = play
            k = (time.time() - t0) / pdt
            if k >= len(pts) - 1:
                sx, sy = ox + pts[-1][0], oy + pts[-1][1]
                state, play = NAV_ARRIVED, None
                print("TRAJ done")
            else:
                i, a = int(k), k - int(k)
                sx = ox + pts[i][0] + a * (pts[i + 1][0] - pts[i][0])
                sy = oy + pts[i][1] + a * (pts[i + 1][1] - pts[i][1])
        elif airborne and goal:
            dx, dy = goal[0] - x, goal[1] - y
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                state, goal, sx, sy = NAV_ARRIVED, None, goal[0], goal[1]
            else:
                err = wrap(math.atan2(dy, dx) - h)
                if abs(err) > 0.15:
                    state = NAV_ROTATING
                    h = (h + math.copysign(min(TURN * dt, abs(err)), err)) % (2 * math.pi)
                    sx, sy = x, y
                else:
                    state = NAV_FLYING
                    step = min(SPEED * dt, dist)
                    sx, sy = x + dx / dist * step, y + dy / dist * step
        if airborne:
            a = dt / (TAU + dt)
            x += a * (sx - x)
            y += a * (sy - y)
        # ---- telemetry ----------------------------------------------------
        pkt = struct.pack("<BBfffBbf32sBH", PKT_TELEM, args.drone_id, x, y, h, state, -1, 0.0,
                          bytes(32), 0, 0xFFFF)
        out.sendto(pkt, (args.host, 5005))
        # ---- AprilTag debug stream (firmware intrinsics, 15 px edge margin) ---
        dets, n = b"", 0
        if airborne:
            for tid, (tn, te) in tags.items():
                dn, de = tn - x, te - y
                fwd = dn * math.cos(h) + de * math.sin(h)
                right = -dn * math.sin(h) + de * math.cos(h)
                tx, ty, tz = body_to_cam((fwd, right, ALT))
                if tz <= 0.05:
                    continue
                u, v = FX * tx / tz + CX, FY * ty / tz + CY
                if 15 < u < W - 15 and 15 < v < H - 15 and math.sqrt(tx * tx + ty * ty + tz * tz) < 2.0:
                    tx += random.gauss(0, args.noise)
                    ty += random.gauss(0, args.noise)
                    tz += random.gauss(0, args.noise)
                    dets += struct.pack("<bB7f", tid, 0, 80.0, u, v, tx, ty, tz, 0.01)
                    n += 1
        hdr = struct.pack("<BBIHbBB", PKT_AT_DEBUG, args.drone_id, int(time.time() * 1000) & 0xFFFFFFFF,
                          20, -1, n, n)
        out.sendto(hdr + dets, (args.host, 5008))
        time.sleep(max(0.0, dt - (time.time() - t_loop)))


if __name__ == "__main__":
    main()
