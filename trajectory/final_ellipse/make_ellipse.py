#!/usr/bin/env python3
"""Constant-speed ellipse: the same speed for the whole lap (optional smooth ramps).

Default: same layout as ellipse_min_snap_demo.m: starts and ends at the +x tip
of the major axis, the ellipse lies behind the start (-x), clockwise seen from
above (first moves towards +y). Fly it with tag_mission.py --traj-rotate 180 to
put it in front of the drone. --forward writes it already in front instead. Unlike the min-snap version, the speed stays at
--speed for the whole lap instead of surging near the start/end.
With --ramp 0 (default) it starts and stops at full speed; the drone
hovers before and after, so PX4 takes that step itself.

    python trajectory/final_ellipse/make_ellipse.py                         # 4.6 x 3.0 m at ~0.30 m/s, no ramps
    python trajectory/final_ellipse/make_ellipse.py --length 2 --width 1 --speed 0.2 --plot

Lopsided, in front of the drone (no --traj-rotate needed), as flown with
--traj-rotate 180 (first moves left), 1.3 m out on the left, 0.9 m on the right:
    python trajectory/final_ellipse/make_ellipse.py --forward --left 1.3 --right 0.9 --out trajectory/final_ellipse/ellipse_asym_traj.csv

--shift 0.3 moves the whole ellipse 0.3 m left (as flown): the drone first flies
0.3 m left (straight into the ellipse, same direction, same speed) and laps; over
the last eighth of the lap the shift fades out, so the path curves straight into
(0, 0) and slows to a stop there (--end-ramp, 2 s by default when shifted).
No stop-and-reverse at the end, so no overshoot.
"""
import argparse
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=float, default=4.6, help="major axis along x (m)")
    ap.add_argument("--width", type=float, default=3.0, help="minor axis along y (m)")
    ap.add_argument("--left", type=float, help="m out to the left (overrides width/2), as flown")
    ap.add_argument("--right", type=float, help="m out to the right (overrides width/2), as flown")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="move the ellipse this many m left (as flown); adds a lead-in from the start")
    ap.add_argument("--forward", action="store_true",
                    help="put the ellipse in front of the start (+x), already turned: fly WITHOUT --traj-rotate")
    ap.add_argument("--speed", type=float, default=0.297, help="cruise speed (m/s); a hair under the 0.30 limit so resampling stays under it")
    ap.add_argument("--ramp", type=float, default=0.0, help="seconds to speed up / slow down (0 = none)")
    ap.add_argument("--end-ramp", type=float, default=None,
                    help="seconds to slow down at the end (default: --ramp, or 2 s when --shift is used)")
    ap.add_argument("--home-speed", type=float, default=0.15,
                    help="peak speed (m/s) of the slide from the shifted tip back to (0, 0)")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=HERE / "ellipse_fast_traj.csv")
    ap.add_argument("--plot", action="store_true", help="also save a speed/top-down PNG next to the CSV")
    args = ap.parse_args()

    a, v, tr = args.length / 2, args.speed, args.ramp
    td = args.end_ramp if args.end_ramp is not None else (max(tr, 2.0) if args.shift > 0 else tr)
    bl = args.left if args.left is not None else args.width / 2
    br = args.right if args.right is not None else args.width / 2

    # Ellipse by angle, then arc length along it. Built as flown in front of the start
    # (start at the back tip, first moves left, clockwise from above); each side has its
    # own half-width. Without --forward it is turned 180 deg (the old behind-the-start
    # layout, meant for --traj-rotate 180).
    th = np.linspace(0.0, 2 * math.pi, 20001)
    ex = a - a * np.cos(th)
    # Shift fades out over the last eighth of the lap (smoothstep), so the lap itself
    # comes back to (0, 0) still heading the same way: no stop-and-reverse, no overshoot.
    u = np.clip((th - 1.75 * math.pi) / (0.25 * math.pi), 0.0, 1.0)
    keep = 1 - (35 * u**4 - 84 * u**5 + 70 * u**6 - 20 * u**7)
    ey = -np.where(np.sin(th) >= 0, bl, br) * np.sin(th) - args.shift * keep
    if args.shift > 0:                                    # lead-in: straight left onto the tip
        n = max(2, int(args.shift / 0.001))
        ex = np.r_[np.zeros(n), ex]
        ey = np.r_[-np.linspace(0.0, args.shift, n, endpoint=False), ey]
    if not args.forward:
        ex, ey = -ex, -ey
    s_th = np.r_[0.0, np.cumsum(np.hypot(np.diff(ex), np.diff(ey)))]
    L = s_th[-1]

    # Speed: sin^2 ramp up over tr, cruise, sin^2 ramp down over td (a ramp covers v*ramp/2).
    if L < v * (tr + td) / 2:
        raise SystemExit("ellipse too short for this speed/ramp")
    T = (L - v * (tr + td) / 2) / v + tr + td
    t = np.arange(0.0, T + 1e-9, args.dt)
    if t[-1] < T:
        t = np.r_[t, T]

    def ramp(tt, r):                                       # integral of v*sin^2(pi t / 2r)
        return v * (tt / 2 - r / (2 * math.pi) * math.sin(math.pi * tt / r))

    def dist(tt):
        if tr > 0 and tt <= tr:
            return ramp(tt, tr)
        if td > 0 and tt >= T - td:
            return L - ramp(T - tt, td)
        return v * tr / 2 + v * (tt - tr)

    s = np.array([dist(tt) for tt in t])
    x, y = np.interp(s, s_th, ex), np.interp(s, s_th, ey)
    x[-1], y[-1] = ex[-1], ey[-1]

    # Shifted: slide from the tip (now at rest) back to (0, 0), rest to rest.
    # Position follows the 7th-order smoothstep, peak speed 35/16 * d / Th.
    d_home = math.hypot(x[-1], y[-1])
    if d_home > 1e-3:
        Th = max(35 / 16 * d_home / args.home_speed, 1.0)
        th_ = np.arange(args.dt, Th + 1e-9, args.dt)
        if th_[-1] < Th:
            th_ = np.r_[th_, Th]
        u = th_ / Th
        f = 35 * u**4 - 84 * u**5 + 70 * u**6 - 20 * u**7
        x = np.r_[x, x[-1] * (1 - f)]
        y = np.r_[y, y[-1] * (1 - f)]
        t = np.r_[t, T + th_]
        x[-1] = y[-1] = 0.0
        T = T + Th
        print(f"  + {d_home:.2f} m slide back to (0, 0) in {Th:.1f} s")

    sp = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
    with args.out.open("w", newline="") as f:
        f.write("# Constant-speed ellipse from make_ellipse.py (x forward, y right, from the start tip)\n")
        where = ("in front of the start (+x), fly without --traj-rotate" if args.forward
                 else "behind the start (-x), fly with --traj-rotate 180")
        f.write(f"# {args.length} m long, {bl} m left / {br} m right (as flown), shifted {args.shift} m left, "
                f"{where}, clockwise; "
                f"cruise {v:.2f} m/s, ramps {tr:.0f} s up / {td:.0f} s down, ends at (0, 0)\n")
        f.write(f"# {L:.2f} m, {T:.1f} s, dt {args.dt:.3f} s\n")
        f.write("t,x,y,z\n")
        for ti, xi, yi in zip(t, x, y):
            f.write(f"{ti:.3f},{xi:.4f},{yi:.4f},0\n")
    print(f"{args.out.name}: {L:.2f} m, {T:.1f} s, peak {sp.max():.3f} m/s, {len(t)} points")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(y, x, color="tab:green", lw=2)
        ax1.plot(0, 0, "o", ms=10, color="tab:orange", label="start")
        ax1.plot(y[-1], x[-1], "s", ms=7, color="tab:red", label="end")
        ax1.set_xlabel("y right (m)"); ax1.set_ylabel("x forward (m)")
        ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
        ax2.plot(t[1:], sp, color="tab:blue")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("speed (m/s)"); ax2.grid(alpha=0.3)
        fig.suptitle(f"Ellipse {args.length} m, {bl} m left / {br} m right, {v:.2f} m/s, ~{T:.0f} s")
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"plot: {png.name}")


if __name__ == "__main__":
    main()
