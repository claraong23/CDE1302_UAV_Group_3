#!/usr/bin/env python3
"""Lawnmower (back-and-forth) trajectory over a square in front of the start.

Writes t,x,y,z (s, m; x forward, y right, z 0) for tag_mission.py --trajectory.
The path starts and ends at the start point (0, 0); the drone keeps its
takeoff heading the whole way (trajectory mode), so the 60 deg camera always
looks forward (+x).

    python trajectory/make_lawnmower.py                       # 1.5 x 1.5 m, 4 lanes
    python trajectory/make_lawnmower.py --length 1.5 --width 1.5 --lanes 4 --plot

Then fly it (no --traj-rotate needed, it already points +x):
    python laptop/tag_mission.py --drone-id 3 --arena laptop/tag_arena_ellipse.yaml \
        --trajectory trajectory/lawnmower_1p5_traj.csv --cam-pitch 60 --confirm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "laptop"))
from search_patterns import smooth_trajectory, write_csv  # noqa: E402

# Same speeds as the search patterns in tag_arena.yaml (lab limit 0.30 m/s).
SPEED = {"cruise": 0.20, "corner": 0.08, "corner_radius": 0.20, "accel": 0.08, "smoothing_s": 1.2}


def lawnmower(length: float, width: float, lanes: int, x0: float = 0.0):
    """Corner points: lanes run along x (forward/back), stepping across y from left to right."""
    ys = np.linspace(-width / 2, width / 2, lanes)
    pts = [(0.0, 0.0)]
    for i, y in enumerate(ys):
        near, far = (x0, x0 + length) if i % 2 == 0 else (x0 + length, x0)
        pts += [(near, y), (far, y)]
    pts.append((0.0, 0.0))                     # home
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=float, default=1.5, help="m along x (forward)")
    ap.add_argument("--width", type=float, default=1.5, help="m along y (sideways), centred on the start line")
    ap.add_argument("--lanes", type=int, default=4, help="number of forward/back lanes")
    ap.add_argument("--x0", type=float, default=0.0, help="m forward where the lanes begin")
    ap.add_argument("--out", type=Path, default=HERE / "lawnmower_1p5_traj.csv")
    ap.add_argument("--plot", action="store_true", help="also save a PNG next to the CSV")
    args = ap.parse_args()

    pts = lawnmower(args.length, args.width, args.lanes, args.x0)
    prof = smooth_trajectory(pts, SPEED, 0.05)
    write_csv(args.out, prof, f"lawnmower {args.length} x {args.width} m, {args.lanes} lanes, "
                              f"x {args.x0:.2f}..{args.x0 + args.length:.2f}, y +-{args.width / 2:.2f}")
    print(f"{args.out.name}: {prof['length']:.2f} m, {prof['duration']:.0f} s, "
          f"max {prof['max_speed']:.2f} m/s, lanes at y = "
          + ", ".join(f"{y:+.2f}" for y in np.linspace(-args.width / 2, args.width / 2, args.lanes)))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xy = prof["xy"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(xy[:, 1], xy[:, 0], "-", color="tab:green", lw=2, label="path")
        c = np.array(pts)
        ax.plot(c[:, 1], c[:, 0], "o", ms=4, color="grey", label="corners")
        ax.plot(0, 0, "o", ms=10, color="tab:orange", label="start / land (0,0)")
        for i in range(0, len(xy) - 1, max(1, len(xy) // 25)):
            ax.annotate("", xy=(xy[i + 1, 1], xy[i + 1, 0]), xytext=(xy[i, 1], xy[i, 0]),
                        arrowprops=dict(arrowstyle="->", color="tab:green"))
        ax.set_xlabel("y right (m)")
        ax.set_ylabel("x forward (m)")
        ax.set_title(f"Lawnmower {args.length} x {args.width} m, {args.lanes} lanes, ~{prof['duration']:.0f} s")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"plot: {png.name}")


if __name__ == "__main__":
    main()
