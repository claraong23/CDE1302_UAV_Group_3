#!/usr/bin/env python3
r"""Two-petal search path traced from a hand sketch (9 Nov 2026), over the cover rectangle.

Seen from above (x forward = up, y right = right), [ ] = the cover rectangle:

         .--.    .--.
        /    \  /    \         1. up a stem left of centre, through the rectangle
       |   .--\/--.   |        2. right petal: over the top, down the right side,
     [ |  /   /\   \  | ]          back left along the near part of the rectangle
     [ |  |  |  |  |  | ]       3. left petal: up the left side, over the top-left,
     [  \  \ |  | /  /  ]          then down the middle
     [   `--+-|--+-'   ]       4. out of the rectangle back to the start
            |  |
            S  E   start / land at (0, 0)

The sketch is stored in rectangle units (u = 0 at the path's left side .. 1 at its
right side, v = 0 at the near edge .. 1 at the far edge, above 1 = past the far
edge) and mapped onto --near/--far/--top/--left/--right in metres.

Writes t,x,y,z (s, m; x forward, y right, z 0) for tag_mission.py --trajectory.
The drone keeps its takeoff heading the whole way (trajectory mode), so the
60 deg camera always looks forward (+x).

    python trajectory/make_petals.py --plot
    python trajectory/make_petals.py --near 1.4 --far 4.1 --top 4.6 --left -1.4 --right 0.93 --plot

Then fly it (no --traj-rotate needed, it already points +x):
    python laptop/tag_mission.py --drone-id 3 --trajectory trajectory/petals_traj.csv --takeoff-wait 16 --confirm
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

# Path traced from the sketch, in order, as (u, v): u 0 = path's left side .. 1 = right
# side; v 0 = near edge of the rectangle .. 1 = far edge (> 1 past it). Start and end
# are added at (0, 0) in metres. ("m", x, y) is a fixed point in metres (e.g. a tag
# to fly over) that does not move with --near/--left/... .
SKETCH = [
    # 1. stem up, left of centre
    (0.34, 0.11), (0.31, 0.43), (0.36, 0.69), (0.44, 0.96), (0.55, 1.19), (0.63, 1.30),
    # 2. right petal: over the top, down the right side, back along the near part
    (0.79, 1.34), (0.96, 1.30), (1.00, 1.19), (1.00, 0.81), (1.00, 0.49),
    ("m", 1.91, 0.915),                              # right over tag 17 (metres)
    (0.78, 0.04), (0.50, 0.00), (0.24, 0.05),        # dips to the near edge (v = 0)
    # 3. left petal: up the left side, over the top-left, down the middle
    ("m", 1.91, -1.37),                              # right over tag 19 (metres)
    (0.00, 0.47),
    (0.00, 0.85), (0.01, 1.12), (0.07, 1.34), (0.17, 1.42),
    (0.29, 1.40), (0.42, 1.30), (0.53, 1.12), (0.61, 0.89), (0.66, 0.63), (0.71, 0.34),
    (0.70, 0.02),
]
V_TOP = max(p[1] for p in SKETCH if p[0] != "m")


def control_points(near: float, far: float, top: float, left: float, right: float):
    """Points through which the path passes, in metres (x forward, y right)."""
    def to_m(u, v):
        x = near + v * (far - near) if v <= 1 else far + (v - 1) / (V_TOP - 1) * (top - far)
        return (x, left + u * (right - left))
    pts = [(p[1], p[2]) if p[0] == "m" else to_m(*p) for p in SKETCH]
    return [(0.0, 0.0)] + pts + [(0.0, 0.0)]


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
    ap.add_argument("--near", type=float, default=1.40, help="m forward: near edge of the cover rectangle")
    ap.add_argument("--far", type=float, default=4.10, help="m forward: far edge of the cover rectangle")
    ap.add_argument("--top", type=float, default=4.60, help="m forward: tops of the petals (past --far)")
    ap.add_argument("--left", type=float, default=-1.40, help="m: path's left side (left = negative)")
    ap.add_argument("--right", type=float, default=0.93, help="m: path's right side")
    ap.add_argument("--dt", type=float, default=0.1, help="sample period (s); 0.1 keeps the upload small")
    ap.add_argument("--out", type=Path, default=HERE / "petals_traj.csv")
    ap.add_argument("--plot", action="store_true", help="also save a PNG next to the CSV")
    ap.add_argument("--arena", type=Path, default=None, help="arena yaml: draw its known tags on the plot")
    args = ap.parse_args()
    if not 0 < args.near < args.far < args.top or not args.left < 0 < args.right:
        raise SystemExit("need 0 < --near < --far < --top and --left < 0 < --right")

    ctrl = control_points(args.near, args.far, args.top, args.left, args.right)
    prof = time_path(catmull_rom(ctrl), args.dt)
    note = (f"petals over x {args.near}..{args.far} (tops {args.top}), "
            f"y {args.left}..{args.right} m, from the 9 Nov sketch")
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
        ax.plot([args.left, args.right, args.right, args.left, args.left],
                [args.near, args.near, args.far, args.far, args.near], ":", color="grey", label="cover rectangle")
        ax.set_title(f"Petals over x {args.near}..{args.far} m, y {args.left}..{args.right} m, "
                     f"~{prof['duration']:.0f} s")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"plot: {png.name}")


if __name__ == "__main__":
    main()
