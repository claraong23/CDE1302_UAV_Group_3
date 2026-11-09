#!/usr/bin/env python3
"""Stem then square: fly forward up a stem, fly a square centred on the stem
line (left half a side, forward, right, back, left half a side), then fly
back down the stem to the start.

Writes t,x,y,z (s, m; x forward, y right, z 0). The drone keeps its takeoff
heading the whole way, so it flies sideways/backwards on some legs.

    python trajectory/make_stem_square.py                      # 1.6 m stem, 3 x 3 m square
    python trajectory/make_stem_square.py --stem 1 --side 2 --plot

Square spans x = stem..stem+side forward and y = -side/2..+side/2.

Then fly it:
    python laptop/send_trajectory.py --drone-id 22 --takeoff --takeoff-wait 12 --confirm \
        --trajectory trajectory/stem_square_traj.csv
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
    ap.add_argument("--stem", type=float, default=1.6, help="m forward (+x) before the square")
    ap.add_argument("--side", type=float, default=3.0, help="square side length (m)")
    ap.add_argument("--out", type=Path, default=HERE / "stem_square_traj.csv")
    ap.add_argument("--plot", action="store_true", help="also save a top-down PNG next to the CSV")
    args = ap.parse_args()

    a, h = args.stem, args.side / 2
    # (x forward, y right): stem, left, forward, right, back, left, back down the stem
    pts = [(0, 0), (a, 0), (a, -h), (a + 2 * h, -h), (a + 2 * h, h), (a, h), (a, 0), (0, 0)]
    prof = smooth_trajectory(pts, SPEED, 0.05)
    write_csv(args.out, prof, f"stem {a} m then {args.side} x {args.side} m square, "
                              f"x {a:.2f}..{a + 2 * h:.2f}, y +-{h:.2f}")
    print(f"{args.out.name}: {prof['length']:.2f} m, {prof['duration']:.0f} s, "
          f"max {prof['max_speed']:.2f} m/s, {len(prof['xy'])} points")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xy = prof["xy"]
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.plot(xy[:, 1], xy[:, 0], "-", color="tab:green", lw=2, label="path")
        for k in range(0, len(xy) - 1, max(1, len(xy) // 30)):
            ax.annotate("", xy=(xy[k + 1, 1], xy[k + 1, 0]), xytext=(xy[k, 1], xy[k, 0]),
                        arrowprops=dict(arrowstyle="->", color="tab:green"))
        ax.plot(0, 0, "o", ms=10, color="tab:orange", label="start / land (0,0)")
        ax.set_xlabel("y right (m)")
        ax.set_ylabel("x forward (m)")
        ax.set_title(f"Stem {a} m + {args.side} x {args.side} m square, ~{prof['duration']:.0f} s")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"plot: {png.name}")


if __name__ == "__main__":
    main()
