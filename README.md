# CDE1302 UAV – Group 3: AprilTag search drone

A small indoor drone that flies a pre-planned path over the arena, spots
AprilTags on the floor with its camera, and reports where the target tags are.

**Group 3**
- Ong Miao Ying, Clara
- Yow Keng Yang, Daniel
- Kugapryan Kumaresan
- Loke Ee Zhe

Built on the course's base code,
[linfengfrank/esp-everything-qgc-bridge](https://github.com/linfengfrank/esp-everything-qgc-bridge),
and extended by our group (see [What we changed](#what-we-changed)).

## How it works

```
 PLAN (offline)            LAPTOP (Python)                 DRONE
 MATLAB / Python  ──CSV──▶ tag_mission.py  ◀──Wi-Fi UDP──▶ ESP32-S3 ──UART MAVLink──▶ PX4 ──▶ motors
 trajectory                tag_stream.py (live video)       camera + 8 ToF sensors
```

1. **Plan:** a trajectory is designed offline (MATLAB minimum-snap, or Python)
   and saved as a CSV of `t, x, y, z`. The units are seconds and metres from
   the start point: x is forward and y is right.
2. **Take off:** `tag_mission.py` uploads the CSV to the drone. When you type
   `GO-3`, the ESP32 arms PX4 and climbs to 1 m.
3. **Fly:** the ESP32 plays the trajectory through PX4. The drone holds its
   heading the whole way and moves sideways and backwards as needed.
4. **Detect:** the ESP32 runs the AprilTag detector (tag16h5) on every camera
   frame and sends each sighting to the laptop.
5. **Correct and map:** the laptop uses the **known tags 15–20**, which are at
   measured positions, to correct PX4's position drift. It places each
   **target tag (0–14)** on the arena map.
6. **Land:** optionally the drone first slides back to (0, 0), then it lands.
   The laptop prints the target tags it found and their positions.

Full diagrams: `system_architecture_simple.png` and
`checkerboard_calibration_flowchart.png`. They were made alongside the project
and are not in this repo.

## Final runs

Run from the repo root on the laptop, connected to the drone's Wi-Fi. Drone 3
is at `172.20.10.13`.

```bash
# Live camera feed with tags drawn (optional, runs alongside a mission)
python laptop/tag_stream.py --esp-ip 172.20.10.13

# Ellipse
python laptop/tag_mission.py --drone-id 3 --trajectory trajectory/ellipse_asym_traj.csv --no-return-home --takeoff-wait 14 --confirm
# Elephant
python laptop/tag_mission.py --drone-id 3 --trajectory trajectory/elephant_min_snap_traj.csv --takeoff-wait 14 --no-return-home --confirm
# Rectangle
python laptop/tag_mission.py --drone-id 3 --trajectory trajectory/rectangle_min_snap_traj.csv --takeoff-wait 14 --confirm
# Petals
python laptop/tag_mission.py --drone-id 3 --trajectory trajectory/petals_traj.csv --takeoff-wait 14 --confirm
```

- `--confirm` asks you to type `GO-3` before anything flies.
- `--takeoff-wait 14` is the time in seconds from takeoff to the start of the
  path. It has to be long enough for the drone to reach 1 m, or the path is
  flown low.
- `--return-home` (the default) slides the drone back to (0, 0) in up to two
  short hops before landing, which costs about 10 s.
  `--no-return-home` lands where the path ends.
- `--dry-run` doesn't fly. It prints the tag-corrected position while you
  carry the drone around, which is a quick check of the camera and the tag
  layout.
- Each run writes a log and a CSV of tag sightings to
  `laptop/final_run_logs/`. Older test runs are in `laptop/logged_runs/`.

## Trajectories

All paths start and end at (0, 0) and cruise at up to 0.297 m/s;
`tag_mission.py` refuses anything faster than 0.30 m/s.

| Path | CSV | Made with | Time |
|---|---|---|---|
| Lopsided ellipse (1.3 m left / 0.9 m right, shifted 0.2 m left) | `ellipse_asym_traj.csv` | `make_ellipse.py` (constant speed) | 38 s |
| Elephant (15 points) | `elephant_min_snap_traj.csv` | `elephant_min_snap_demo.m` (MATLAB minimum-snap) | 79 s |
| Rectangle over the likely tag spots | `rectangle_min_snap_traj.csv` | `rectangle_min_snap_demo.m` (MATLAB minimum-snap) | 50 s |
| Petals, passing over tags 17 and 19 | `petals_traj.csv` | `make_petals.py` (Catmull-Rom, curvature-limited speed) | 83 s |

Minimum-snap uses one 7th-order polynomial per leg and minimises the snap (the
fourth derivative of position). The result is smooth enough for PX4 to track
closely. Open a `.m` file in MATLAB and press Run to regenerate its CSV and
plot. Other paths in `trajectory/` (line, stem-square, triangle, lawnmower,
min-snap ellipses) were for testing.

## Arena set-up

Tag positions are in `laptop/tag_arena.yaml`, in metres from the start point
(x forward, y right):

| Known tag | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|
| x | 0.91 | 1.91 | 1.91 | 3.19 | 1.91 | 3.19 |
| y | 0.00 | 0.00 | 0.915 | −1.37 | −1.37 | 0.915 |

- **Targets** are any tag with ID 0–14; only a few are placed.
- **Tag size:** all tags are tag16h5 with a 114 mm black square.
- **Adding a known tag** needs no reflash: add a line under `known_tags:`.
  Use ID 21, 22 or 23 (the drone detects IDs 0–23), and don't reuse a target
  ID.
- **Start position:** place the drone at (0, 0) facing +x, towards tag 15.

## Repo layout

| Folder | What's in it |
|---|---|
| `main/` | ESP32-S3 firmware (C, FreeRTOS): Wi-Fi, PX4 MAVLink link, takeoff/landing, trajectory player, obstacle avoidance, AprilTag detection, camera stream |
| `components/` | Libraries used by the firmware: esp-apriltag, MAVLink, VL53L5CX ToF driver |
| `laptop/` | Python: `tag_mission.py` (the mission), `tag_stream.py` (live feed), and the modules they use |
| `laptop/tag_arena.yaml` | Known tag positions and target IDs |
| `trajectory/` | Trajectory generators (MATLAB `.m`, Python) and their CSVs and plots |
| `tools/` | ESP-IDF version pin |

Main laptop modules:
- `comms.py`, `protocol.py`: UDP messages to and from the drone.
- `apriltag_host.py`: the drone's own AprilTag C code compiled for the
  laptop, so `tag_stream.py` decodes tags exactly as the drone does.
- `tag_fusion.py`: combines many sightings of a target into one position.
- `search_patterns.py`: smooths paths and checks camera coverage.
- `send_trajectory.py`: loads trajectory CSVs.
- `simple_waypoint_mission.py`: telemetry tracking and commands.

## What we changed

Compared with the base repo:

**Drone firmware.** These changes only reach the drone after a rebuild and
reflash.
- **Camera calibration** (`main/at_detect.c`): fx, fy, cx, cy measured on
  drone 3's own camera with a checkerboard (fx 188.9, fy 188.2, cx 160.5,
  cy 118.1; 0.30 px error). The base code had fx 163.5 and fy 153.2. These
  numbers turn a tag's size in the picture into its distance, so wrong values
  give wrong positions. The same numbers are in `laptop/apriltag_host.py`.
- **Tag size** 0.12 m → 0.114 m, the measured black square.
- **Fixed camera exposure** (exposure 50, auto gain): less motion blur on the
  tags while flying.
- **Tag IDs 0–23** instead of 0–21
  (`components/esp-apriltag/apriltag/tag16h5.c`).
- **Takeoff height** 0.5 m → 1.0 m (`CRUISE_ALT_M` in `main/main.c`,
  `WIFI_CRUISE_ALT_M` in `main/wifi_task.c`). Higher means the camera sees
  more floor.
- **New command `CMD_YAW`:** hold position and turn to a given heading.

**Laptop** (`laptop/tag_mission.py`, new):
- **Tag-based localisation:** each sighting of a known tag corrects PX4's
  drift. Two known tags in the same picture also correct the heading.
- **Target mapping:** each target is placed from a known tag in the same
  picture when possible (no drift), otherwise from the corrected drone
  position. All sightings of a target are combined by `tag_fusion.py`.
- **Trajectory flying from CSV**, with a speed check, timing per stage, and
  an optional slide back to (0, 0) before landing (`--return-home`).
- **Logs** for every run in `laptop/final_run_logs/`.

**Trajectories:** the four final paths above, plus test paths.

## Setup

**Laptop:** Python 3.10+.
```bash
python -m pip install -r laptop/requirements.txt
```
`tag_stream.py` compiles the AprilTag detector the first time it runs, so it
needs a C compiler. On Windows that's MinGW-w64 gcc, for example
`conda install -c conda-forge gcc`.

**Firmware:** needs ESP-IDF; the pinned version is in `tools/idf-pin.env`.
```bash
idf.py menuconfig                 # Drone Configuration: drone ID, Wi-Fi, laptop IP
idf.py build
idf.py -p COM5 flash monitor      # use the drone's COM port (Device Manager)
```
