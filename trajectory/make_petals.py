#!/usr/bin/env python3
r"""Two-petal trajectory: up a stem, a loop to the left, a loop to the right, back down.

Shape traced from a hand sketch (Nov 2026). Seen from above, x forward, y right:

            _          _
          /   \      /   \
         |  L  |    |  R  |        L: up the inner edge, over the top to the left,
          \     \  /     /            down the outside, back in along the bottom.
            `----J----'            R: the mirror image, on the right.
                 |
                 |  stem
                 X  start / land (0, 0)

Writes t,x,y,z (s, m; x forward, y right, z 0) for tag_mission.py --trajectory.
The drone keeps its takeoff heading the whole way (trajectory mode), so the
60 deg camera always looks forward (+x).

    python trajectory/make_petals.py                              # 2.0 x 2.0 m, stem 0.5 m
    python trajectory/make_petals.py --length 2.0 --width 2.0 --stem 0.5 --plot
    python trajectory/make_petals.py --plot --arena laptop/tag_arena.yaml   # show the known tags

Then fly it (no --traj-rotate needed, it already points +x):
    python laptop/tag_mission.py --drone-id 3 --arena laptop/tag_arena.yaml \
        --trajectory trajectory/petals_traj.csv --cam-pitch 60 --confirm
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "laptop"))
from search_patterns import SPEED_HARD_LIMIT, TRAJ_MAX_PTS  # noqa: E402

# Same gentle speeds as make_lawnmower.py / the search patterns (lab limit 0.30 m/s).
CRUISE_MS = 0.295     # on the straighter parts (just under the 0.30 m/s lab limit after resampling)
ACCEL_MS2 = 0.08      # speeding up / slowing down along the path
LATERAL_MS2 = 0.04    # sideways acceleration allowed in bends (slows down on tight curves)

# Left petal traced from the sketch, in petal units:
#   a = 0 at the junction J .. 1 at the top of the petal (forward),
#   b = 0 on the stem line .. -1 at the petal's far left edge.
# Order: up the inner edge, over the top to the left, down the outside, back to J.
LEFT_PETAL = [
    (0.00, 0.00), (0.25, 0.00), (0.48, -0.05), (0.69, -0.16), (0.84, -0.31),
    (0.95, -0.51), (1.00, -0.71), (0.95, -0.85), (0.78, -0.93), (0.56, -1.00),
    (0.33, -0.98), (0.16, -0.81), (0.04, -0.51), (0.00, -0.16), (0.00, 0.00),
]


def control_points(length: float, width: float, stem: float):
    """Points through which the path passes, in metres (x forward, y right)."""
    petal_len, half_w = length - stem, width / 2
    left = [(stem + a * petal_len, b * half_w) for a, b in LEFT_PETAL]
    right = [(x, -y) for x, y in left]                   # mirror image
    up = [(0.0, 0.0), (stem / 2, 0.0)]
    down = [(stem / 2, 0.0), (0.0, 0.0)]
    return up + left + right[1:] + down


def catmull_rom(pts, per_seg: int = 60):
    """Smooth curve through every point (centripetal Catmull-Rom, no overshoot loops)."""
    P = np.asarray(pts, float)
    P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]

        def tj(ti, a, b):
            return ti + max(np.linalg.norm(b - a), 1e-6) ** 0.5
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        for t in np.linspace(t1, t2, per_seg, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[-2])
    return np.array(out)


def time_path(path: np.ndarray, dt: float):
    """Speed along the path (cruise, slower in bends, gentle start/stop), sampled every dt."""
    ds_raw = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_raw = np.r_[0.0, np.cumsum(ds_raw)]
    L = float(s_raw[-1])
    step = 0.005
    s = np.arange(0.0, L, step)
    s = np.r_[s, L]
    P = np.column_stack([np.interp(s, s_raw, path[:, 0]), np.interp(s, s_raw, path[:, 1])])
    ds = np.diff(s)

    # Curvature = turn angle per metre, smoothed over ~10 cm.
    heading = np.unwrap(np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0])))
    kappa = np.abs(np.gradient(heading, s[:-1]))
    k = 21
    kappa = np.convolve(np.r_[np.full(k, kappa[0]), kappa, np.full(k, kappa[-1])], np.ones(k) / k, "same")[k:-k]
    kappa = np.r_[kappa, kappa[-1]]

    v_cruise = min(CRUISE_MS, SPEED_HARD_LIMIT)
    v_lim = np.minimum(v_cruise, np.sqrt(LATERAL_MS2 / np.maximum(kappa, 1e-6)))
    v_lim[0] = v_lim[-1] = 0.0
    v2 = v_lim ** 2
    for i in range(1, len(v2)):                          # speeding up
        v2[i] = min(v2[i], v2[i - 1] + 2 * ACCEL_MS2 * ds[i - 1])
    for i in range(len(v2) - 2, -1, -1):                 # slowing down
        v2[i] = min(v2[i], v2[i + 1] + 2 * ACCEL_MS2 * ds[i])
    v = np.sqrt(v2)
    t = np.r_[0.0, np.cumsum(ds / np.maximum((v[:-1] + v[1:]) / 2, 1e-4))]
    duration = float(t[-1])

    n = int(math.ceil(duration / dt)) + 1
    if n > TRAJ_MAX_PTS:                                 # firmware buffer: use a longer step
        dt = math.ceil(duration / (TRAJ_MAX_PTS - 1) * 1000) / 1000
        n = int(math.ceil(duration / dt)) + 1
    sk = np.interp(np.arange(n) * dt, t, s, right=L)
    xy = np.column_stack([np.interp(sk, s, P[:, 0]), np.interp(sk, s, P[:, 1])])
    spd = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    return {"xy": xy, "dt": dt, "duration": duration, "length": L, "max_speed": float(spd.max())}


def write_csv(path: Path, prof, note: str):
    with path.open("w", newline="") as f:
        f.write("# Two-petal trajectory from make_petals.py (x forward, y right, from the start point)\n")
        f.write(f"# {note}\n")
        f.write(f"# {prof['length']:.2f} m, {prof['duration']:.1f} s, dt {prof['dt']:.3f} s\n")
        f.write("t,x,y,z\n")
        for i, (x, y) in enumerate(prof["xy"]):
            f.write(f"{i * prof['dt']:.3f},{x:.4f},{y:.4f},0\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=float, default=2.0, help="m forward, from the start X to the petal tops")
    ap.add_argument("--width", type=float, default=2.0, help="m side to side, across both petals (centred)")
    ap.add_argument("--stem", type=float, default=0.5, help="m forward from the start X to where the petals meet")
    ap.add_argument("--dt", type=float, default=0.1, help="sample period (s); 0.1 keeps the upload small")
    ap.add_argument("--out", type=Path, default=HERE / "petals_traj.csv")
    ap.add_argument("--plot", action="store_true", help="also save a PNG next to the CSV")
    ap.add_argument("--arena", type=Path, default=None, help="arena yaml: draw its known tags on the plot")
    args = ap.parse_args()
    if not 0 <= args.stem < args.length:
        raise SystemExit("--stem must be between 0 and --length")

    ctrl = control_points(args.length, args.width, args.stem)
    prof = time_path(catmull_rom(ctrl), args.dt)
    note = f"petals {args.length} m long x {args.width} m wide, stem {args.stem} m"
    write_csv(args.out, prof, note)
    xy = prof["xy"]
    print(f"{args.out.name}: {prof['length']:.2f} m, {prof['duration']:.0f} s, max {prof['max_speed']:.2f} m/s, "
          f"x {xy[:, 0].min():.2f}..{xy[:, 0].max():.2f}, y {xy[:, 1].min():+.2f}..{xy[:, 1].max():+.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.plot(xy[:, 1], xy[:, 0], "-", color="tab:green", lw=2, label="path")
        step = max(1, len(xy) // 30)
        for i in range(0, len(xy) - 1, step):
            ax.annotate("", xy=(xy[i + 1, 1], xy[i + 1, 0]), xytext=(xy[i, 1], xy[i, 0]),
                        arrowprops=dict(arrowstyle="->", color="tab:green"))
        ax.plot(0, 0, "x", ms=14, mew=3, color="tab:orange", label="start / land (0,0)")
        if args.arena:
            import yaml
            cfg = yaml.safe_load(args.arena.read_text())
            for tid, (tx, ty, *_) in (cfg.get("known_tags") or {}).items():
                ax.plot(ty, tx, "s", ms=9, color="black")
                ax.annotate(str(tid), (ty, tx), xytext=(6, 4), textcoords="offset points")
            ax.plot([], [], "s", color="black", label="known tags")
        ax.set_xlabel("y right (m)")
        ax.set_ylabel("x forward (m)")
        ax.set_title(f"Petals {args.length} x {args.width} m, stem {args.stem} m, ~{prof['duration']:.0f} s")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"plot: {png.name}")


if __name__ == "__main__":
    main()
