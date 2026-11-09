#!/usr/bin/env python3
"""Straight out-and-back line: fly forward, stop, fly back to the start.

Writes t,x,y,z (s, m; x forward, y right, z 0). The drone keeps its takeoff
heading the whole way, so it flies backwards on the return leg.

    python trajectory/make_line.py                  # 4.6 m forward and back
    python trajectory/make_line.py --length 3 --plot

Then fly it:
    python laptop/send_trajectory.py --drone-id 22 --takeoff --takeoff-wait 12 --confirm \
        --trajectory trajectory/line_traj.csv
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "laptop"))
from search_patterns import smooth_trajectory, write_csv  # noqa: E402

# Same speeds as make_lawnmower.py (lab limit 0.30 m/s).
SPEED = {"cruise": 0.20, "corner": 0.08, "corner_radius": 0.20, "accel": 0.08, "smoothing_s": 1.2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=float, default=4.6, help="m forward (+x) before turning back")
    ap.add_argument("--out", type=Path, default=HERE / "line_traj.csv")
    ap.add_argument("--plot", action="store_true", help="also save a PNG of x vs time next to the CSV")
    args = ap.parse_args()

    pts = [(0.0, 0.0), (args.length, 0.0), (0.0, 0.0)]
    prof = smooth_trajectory(pts, SPEED, 0.05)
    write_csv(args.out, prof, f"line {args.length} m forward (+x) then back to start")
    print(f"{args.out.name}: {prof['length']:.2f} m, {prof['duration']:.0f} s, "
          f"max {prof['max_speed']:.2f} m/s, {len(prof['xy'])} points")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = prof["xy"][:, 0]
        t = [i * prof["dt"] for i in range(len(x))]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(t, x, color="tab:green", lw=2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("x forward (m)")
        ax.set_title(f"Line {args.length} m out and back, ~{prof['duration']:.0f} s")
        ax.grid(alpha=0.3)
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"plot: {png.name}")


if __name__ == "__main__":
    main()
