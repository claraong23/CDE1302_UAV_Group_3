#!/usr/bin/env python3
"""
Search patterns for tag_mission.py --search (planning only, no drone I/O).

The behaviour (configured in the `search:` block of tag_arena.yaml):

    pad -> forward to the search area -> up its left edge to TL   (entry)
    turn to face +x at TL
    pattern "red":  TL -> BR -> BL -> TR -> TL   (the X plus top and bottom edges)
    DECISION at TL: every expected target found? -> exit and land
                    otherwise                    -> turn to face -x, next pattern
    pattern "blue": TL -> TR -> BL -> BR -> TL   (same lines, opposite direction)
    exit: down the left edge to (0.9, 0) -> pad, land

Heading: in trajectory mode the drone holds its heading for a whole pattern,
so each pattern sets the heading it is flown with (`heading:` in degrees in
the arena frame, 0 = facing +x, 180 = facing -x).

Corner names are as seen on the handout drawing (x forward = right on the
drawing, y right of the drone = down on the drawing):
    TL = (x_min, y_min)   TR = (x_max, y_min)
    BL = (x_min, y_max)   BR = (x_max, y_max)

This module turns a list of arena points into a smooth trajectory (rounded
corners, gentle acceleration, speed limits) for play_trajectory(), and can
preview the plan without a drone:

    python laptop/search_patterns.py                      # print timings + camera coverage
    python laptop/search_patterns.py --plot search_plan.png
    python laptop/search_patterns.py --csv search_plans    # save each pattern as t,x,y,z CSV
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
TRAJ_MAX_PTS = 2400            # firmware buffer (protocol.TRAJ_MAX_PTS)
SPEED_HARD_LIMIT = 0.30        # m/s, lab guidance for trajectories
CORNERS = ("TL", "TR", "BL", "BR")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class SearchPlan:
    """The `search:` block of tag_arena.yaml, with corner names resolved to (x, y)."""

    def __init__(self, cfg: dict):
        a = cfg["area"]
        if "length" in a:                      # centre + length (along x) + width (along y)
            cx, cy = (float(v) for v in a["centre"])
            length, width = float(a["length"]), float(a["width"])
            if length <= 0 or width <= 0:
                raise ValueError("search.area length and width must be positive")
            self.area = {"x_min": cx - length / 2, "x_max": cx + length / 2,
                         "y_min": cy - width / 2, "y_max": cy + width / 2}
        else:                                  # or the four edges directly
            self.area = {k: float(a[k]) for k in ("x_min", "x_max", "y_min", "y_max")}
        self.length = self.area["x_max"] - self.area["x_min"]
        self.width = self.area["y_max"] - self.area["y_min"]
        self.corners = {
            "TL": (self.area["x_min"], self.area["y_min"]),
            "TR": (self.area["x_max"], self.area["y_min"]),
            "BL": (self.area["x_min"], self.area["y_max"]),
            "BR": (self.area["x_max"], self.area["y_max"]),
        }
        tl = self.corners["TL"]
        # auto: straight forward along y = 0 to the box's near edge, then sideways to TL
        # (and the reverse for the exit), so they follow the box when its size changes.
        entry, exit_ = cfg.get("entry", "auto"), cfg.get("exit", "auto")
        self.entry = ([(tl[0], 0.0), tl] if entry == "auto" else [self.point(p) for p in entry])
        self.exit = ([(tl[0], 0.0), (0.0, 0.0)] if exit_ == "auto" else [self.point(p) for p in exit_])
        # (name, points, heading_deg or None); heading None = keep whatever heading it has
        self.patterns = [(p["name"], [self.point(c) for c in p["corners"]],
                          None if p.get("heading") is None else float(p["heading"]))
                         for p in cfg["patterns"]]
        turn = cfg.get("turn") or {}
        self.turn = {"tolerance_deg": float(turn.get("tolerance_deg", 10.0)),
                     "look_m": float(turn.get("look_m", 0.6)),
                     "timeout_s": float(turn.get("timeout_s", 8.0)),
                     "attempts": int(turn.get("attempts", 3))}
        for name, pts, _ in self.patterns:
            if math.dist(pts[0], pts[-1]) > 1e-6:
                raise ValueError(f"search pattern '{name}' must start and end at the same corner "
                                 f"(the decision point)")
        self.expected_tags = int(cfg.get("expected_tags") or 0)
        self.stop_early = bool(cfg.get("stop_early", True))
        self.max_loops = int(cfg.get("max_loops", len(self.patterns)))
        self.mode = cfg.get("mode", "trajectory")
        if self.mode not in ("trajectory", "waypoints"):
            raise ValueError("search.mode must be 'trajectory' or 'waypoints'")
        self.speed = {"cruise": 0.20, "corner": 0.08, "corner_radius": 0.30,
                      "accel": 0.08, "smoothing_s": 1.2, **(cfg.get("speed") or {})}
        self.goto_speed = float(cfg.get("goto_speed", 0.10))
        self.spots = [tuple(map(float, s)) for s in cfg.get("spots") or []]
        self.camera = {"pitch_deg": 60.0, "altitude": 0.5, **(cfg.get("camera") or {})}

    def point(self, p):
        if isinstance(p, str):
            if p not in self.corners:
                raise ValueError(f"unknown corner '{p}' (use {', '.join(CORNERS)} or [x, y])")
            return self.corners[p]
        return (float(p[0]), float(p[1]))


def load_plan(arena_yaml: Path) -> SearchPlan:
    cfg = yaml.safe_load(Path(arena_yaml).read_text())
    if not cfg.get("search"):
        raise SystemExit(f"{arena_yaml}: no 'search:' block")
    return SearchPlan(cfg["search"])


# ---------------------------------------------------------------------------
# Smooth trajectory
# ---------------------------------------------------------------------------
def _dedupe(points):
    out = [tuple(map(float, points[0]))]
    for p in points[1:]:
        if math.dist(p, out[-1]) > 1e-6:
            out.append(tuple(map(float, p)))
    return out


def _dense_path(vertices, radius, ds=0.005):
    """Polyline with circular arcs at the corners, sampled about every ds metres.
    Returns points (N, 2), per-point kind (0 straight, 1 arc, 2 stop) and arc index ranges."""
    V = np.asarray(vertices, float)
    n = len(V)
    seg = V[1:] - V[:-1]
    lens = np.linalg.norm(seg, axis=1)
    dirs = seg / lens[:, None]
    corner = [None] * n
    for i in range(1, n - 1):
        u1, u2 = dirs[i - 1], dirs[i]
        theta = math.acos(max(-1.0, min(1.0, float(u1 @ u2))))
        if theta < 1e-3:
            continue
        if theta > math.radians(170):
            corner[i] = ("stop",)                       # full reversal: stop and go back
            continue
        avail_in = lens[i - 1] if i - 1 == 0 else lens[i - 1] / 2
        avail_out = lens[i] if i == n - 2 else lens[i] / 2
        # Tangent length capped at `radius`: right angles get the full radius, sharp
        # turns (edge -> diagonal) a smaller arc instead of cutting far inside the corner.
        t = min(radius * math.tan(theta / 2), radius, avail_in, avail_out)
        corner[i] = ("arc", t, t / math.tan(theta / 2), theta)

    pts, kind, arcs = [], [], []

    def line(p, q):
        m = max(1, int(math.ceil(np.linalg.norm(q - p) / ds)))
        for k in range(m):
            pts.append(p + (q - p) * k / m)
            kind.append(0)

    for i in range(n - 1):
        u = dirs[i]
        t0 = corner[i][1] if corner[i] and corner[i][0] == "arc" else 0.0
        t1 = corner[i + 1][1] if corner[i + 1] and corner[i + 1][0] == "arc" else 0.0
        line(V[i] + u * t0, V[i + 1] - u * t1)
        c = corner[i + 1]
        if c is None:
            continue
        if c[0] == "stop":
            pts.append(V[i + 1].copy())
            kind.append(2)
            continue
        _, t, R, theta = c
        u2 = dirs[i + 1]
        p1 = V[i + 1] - u * t
        sgn = 1.0 if (u[0] * u2[1] - u[1] * u2[0]) > 0 else -1.0
        centre = p1 + sgn * R * np.array([-u[1], u[0]])
        phi0 = math.atan2(p1[1] - centre[1], p1[0] - centre[0])
        m = max(2, int(math.ceil(R * theta / ds)))
        start = len(pts)
        for k in range(m):
            phi = phi0 + sgn * theta * k / m
            pts.append(centre + R * np.array([math.cos(phi), math.sin(phi)]))
            kind.append(1)
        arcs.append((start, len(pts)))
    pts.append(V[-1].copy())
    kind.append(0)

    P = np.asarray(pts)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
    remap = np.cumsum(keep) - 1
    arcs = [(int(remap[a]), int(remap[min(b, len(P) - 1)])) for a, b in arcs]
    return P[keep], np.asarray(kind)[keep], arcs


def smooth_trajectory(points, speed: dict, dt: float = 0.05):
    """Points (arena frame) -> dict with samples 'xy' (N, 2) every 'dt' s, plus stats.

    Speed limit on straights and through corners, acceleration limit, then the
    position along the path is smoothed in time so speed changes gently."""
    verts = _dedupe(points)
    if len(verts) < 2:
        xy = np.array([verts[0], verts[0]])
        return {"xy": xy, "dt": dt, "duration": dt, "length": 0.0, "max_speed": 0.0, "max_corner_speed": 0.0}
    v_cruise = min(float(speed["cruise"]), SPEED_HARD_LIMIT)
    v_corner = min(float(speed["corner"]), v_cruise)
    a_max = float(speed["accel"])
    w = float(speed["smoothing_s"])

    P, kind, arcs = _dense_path(verts, float(speed["corner_radius"]))
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    L = float(s[-1])

    v_lim = np.full(len(P), v_cruise)
    margin = 0.5 * v_cruise * w                       # slow down a little before each arc
    for a, b in arcs:
        v_lim[(s >= s[a] - margin) & (s <= s[b] + margin)] = v_corner
    v_lim[kind == 2] = 0.0
    v_lim[0] = v_lim[-1] = 0.0
    v2 = v_lim ** 2
    for i in range(1, len(v2)):
        v2[i] = min(v2[i], v2[i - 1] + 2 * a_max * ds[i - 1])
    for i in range(len(v2) - 2, -1, -1):
        v2[i] = min(v2[i], v2[i + 1] + 2 * a_max * ds[i])
    v = np.sqrt(v2)
    t = np.r_[0.0, np.cumsum(ds / np.maximum((v[:-1] + v[1:]) / 2, 1e-6))]

    fdt = 0.01
    tf = np.arange(0.0, t[-1] + fdt, fdt)
    sf = np.interp(tf, t, s)
    if w > 0:
        k = max(1, int(round(w / 2 / fdt)))
        k += (k + 1) % 2
        kern = np.convolve(np.ones(k) / k, np.ones(k) / k)
        pad = len(kern) * 3
        sm = np.convolve(np.r_[np.zeros(pad), sf, np.full(pad, L)], kern, mode="same")
        i0 = max(int(np.argmax(sm > 1e-7)) - 1, 0)
        i1 = int(np.argmax(sm >= L - 1e-7))
        sf = sm[i0:i1 + 1].copy()
        sf[0], sf[-1] = 0.0, L
        tf = np.arange(len(sf)) * fdt
    duration = float(tf[-1])

    n_pts = int(math.ceil(duration / dt)) + 1
    if n_pts > TRAJ_MAX_PTS:                          # firmware buffer: use a longer step
        dt = math.ceil(duration / (TRAJ_MAX_PTS - 1) * 1000) / 1000
        n_pts = int(math.ceil(duration / dt)) + 1
    sk = np.interp(np.arange(n_pts) * dt, tf, sf, right=L)
    xy = np.column_stack([np.interp(sk, s, P[:, 0]), np.interp(sk, s, P[:, 1])])

    vel = np.diff(xy, axis=0) / dt
    spd = np.linalg.norm(vel, axis=1)
    mid = (sk[:-1] + sk[1:]) / 2
    in_arc = np.zeros(len(mid), bool)
    for a, b in arcs:
        in_arc |= (mid >= s[a]) & (mid <= s[b])
    return {"xy": xy, "dt": dt, "duration": duration, "length": L,
            "max_speed": float(spd.max()) if len(spd) else 0.0,
            "max_corner_speed": float(spd[in_arc].max()) if in_arc.any() else 0.0,
            "speed": spd}


def write_csv(path, prof, note=""):
    """Save a planned trajectory as t,x,y,z (s, m; arena frame: x forward, y right, z 0).

    Same columns as the lab's trajectory CSVs, so it can be replayed with
    `tag_mission.py --trajectory <file>` (x/y read as arena forward/right) or plotted
    against the PX4 .ulg log."""
    path = Path(path)
    with path.open("w", newline="") as f:
        f.write("# Search trajectory from search_patterns.py (arena frame, origin = tag 1)\n")
        if note:
            f.write(f"# {note}\n")
        f.write(f"# {prof['length']:.2f} m, {prof['duration']:.1f} s, dt {prof['dt']:.3f} s\n")
        f.write("t,x,y,z\n")
        for i, (x, y) in enumerate(prof["xy"]):
            f.write(f"{i * prof['dt']:.3f},{x:.4f},{y:.4f},0\n")
    return path


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------
def camera_coverage(xy, dt, spots, pitch_deg=60.0, altitude=0.5, heading_deg=0.0,
                    fx=175.57304290, fy=174.59024332, cx=157.15561504, cy=116.21082668,
                    width=320, height=240, margin_px=15, cam_pos=(0.02, 0.0, 0.0), max_range=2.0):
    """Seconds each floor spot is fully inside the image, flying with a fixed heading
    (arena frame, 0 = facing +x)."""
    p = math.radians(pitch_deg)
    R = np.array([[0.0, -math.sin(p), math.cos(p)], [1.0, 0.0, 0.0], [0.0, math.cos(p), math.sin(p)]])
    off = np.asarray(cam_pos, float)
    h = math.radians(heading_deg)
    ch, sh = math.cos(h), math.sin(h)
    out = []
    for sx, sy in spots:
        dx, dy = sx - xy[:, 0], sy - xy[:, 1]
        fwd, right = dx * ch + dy * sh, -dx * sh + dy * ch
        body = np.column_stack([fwd, right, np.full(len(xy), altitude)]) - off
        c = body @ R
        ok = c[:, 2] > 0.05
        z = np.where(ok, c[:, 2], 1.0)
        u, v = fx * c[:, 0] / z + cx, fy * c[:, 1] / z + cy
        vis = ok & (u > margin_px) & (u < width - margin_px) & (v > margin_px) & (v < height - margin_px) \
            & (np.linalg.norm(c, axis=1) <= max_range)
        out.append(((sx, sy), float(vis.sum() * dt)))
    return out


def plan_legs(plan: SearchPlan, start=(0.0, 0.0)):
    """[(label, points, heading_deg)] for preview: entry, each pattern, exit from the decision corner."""
    legs = [("entry", [start] + plan.entry, None)]
    for name, pts, heading in plan.patterns:
        legs.append((name, pts, heading))
    legs.append(("exit", [plan.patterns[0][1][-1]] + plan.exit, None))
    return legs


def preview(plan: SearchPlan, plot_path=None, csv_dir=None):
    print(f"Box: length {plan.length:.2f} m (x {plan.area['x_min']:.2f}..{plan.area['x_max']:.2f}), "
          f"width {plan.width:.2f} m (y {plan.area['y_min']:.2f}..{plan.area['y_max']:.2f})")
    print("Corners: " + ", ".join(f"{k}=({x:.2f}, {y:.2f})" for k, (x, y) in plan.corners.items()))
    print(f"Mode: {plan.mode}; expected tags: {plan.expected_tags or 'number of targets'}; "
          f"stop early: {plan.stop_early}; max loops: {plan.max_loops}")
    results = []
    for label, pts, heading in plan_legs(plan):
        if plan.mode == "trajectory" or label in ("entry", "exit"):
            prof = smooth_trajectory(pts, plan.speed)
            secs = prof["duration"] if label not in ("entry", "exit") else prof["length"] / plan.goto_speed
        else:
            prof = smooth_trajectory(pts, plan.speed)
            secs = prof["length"] / plan.goto_speed
        results.append((label, prof, secs, heading))
        how = "trajectory" if (plan.mode == "trajectory" and label not in ("entry", "exit")) else "CMD_GOTO"
        facing = "" if heading is None else f", facing {heading:.0f} deg"
        print(f"\n{label:6s} {prof['length']:5.2f} m  ~{secs:4.0f} s  ({how}{facing})")
        if plan.spots and label not in ("entry", "exit"):
            for (sx, sy), t_in in camera_coverage(prof["xy"], prof["dt"], plan.spots,
                                                  plan.camera["pitch_deg"], plan.camera["altitude"],
                                                  heading if heading is not None else 0.0):
                flag = "" if t_in >= 1.0 else "   <-- NOT SEEN"
                print(f"   spot ({sx:5.2f}, {sy:5.2f}) in view {t_in:5.1f} s{flag}")
    entry, exit_ = results[0][2], results[-1][2]
    total = entry
    for label, prof, secs, _ in results[1:-1]:
        total += secs
        print(f"\nStop after {label}: ~{total + exit_:.0f} s of flight (excl. take-off/landing)")
    if csv_dir:
        out = Path(csv_dir)
        out.mkdir(parents=True, exist_ok=True)
        for label, prof, _, heading in results[1:-1]:
            facing = "" if heading is None else f", flown facing {heading:.0f} deg"
            p = write_csv(out / f"search_plan_{label}.csv", prof,
                          f"pattern '{label}' planned from the TL corner{facing}")
            print(f"CSV saved to {p}")
    if plot_path:
        _plot(plan, results, plot_path)


def _plot(plan, results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not installed - no plot")
        return
    pats = [r for r in results if r[0] not in ("entry", "exit")]
    fig, axes = plt.subplots(1, len(pats), figsize=(6.2 * len(pats), 5.6), squeeze=False)
    colours = {"red": "#c62828", "blue": "#1d4ed8"}
    a = plan.area
    for ax, (label, prof, secs, heading) in zip(axes[0], pats):
        ax.add_patch(Rectangle((a["x_min"], a["y_min"]), a["x_max"] - a["x_min"], a["y_max"] - a["y_min"],
                               fill=False, ec="#4caf50", lw=2))
        ax.add_patch(Rectangle((-0.45, -0.45), 0.9, 0.9, fc="#4caf5033", ec="#2e7d32"))
        if plan.spots:
            ax.scatter(*zip(*plan.spots), s=70, facecolors="none", edgecolors="k", zorder=3)
        for lab, pr, _, _ in results:
            if lab in ("entry", "exit"):
                ax.plot(pr["xy"][:, 0], pr["xy"][:, 1], color="#888", ls="--", lw=1.2)
        xy = prof["xy"]
        col = colours.get(label, "#6a1b9a")
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=2.4, label=label)
        step = max(1, len(xy) // 10)
        for j in range(step // 2, len(xy) - 15, step):
            ax.annotate("", xy=xy[j + 12], xytext=xy[j], arrowprops=dict(arrowstyle="->", color=col, lw=2))
        for k, (x, y) in plan.corners.items():
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(-14 if x == a["x_min"] else 6, -10 if y == a["y_min"] else 6))
        ax.plot(0, 0, "o", color="#ff6d00", ms=8)
        if heading is not None:                         # camera/nose direction for this pattern
            tl = plan.corners["TL"]
            h = math.radians(heading)
            ax.annotate("", xy=(tl[0] + 0.6 * math.cos(h), tl[1] + 0.6 * math.sin(h)), xytext=tl,
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5))
            ax.annotate(f"nose {heading:.0f}°", tl, textcoords="offset points", xytext=(4, -22), fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()                               # y (drone's right) drawn downward, like the handout
        ax.set_xlabel("x forward (m)")
        ax.set_ylabel("y right (m)")
        title_h = "" if heading is None else f", facing {heading:.0f}°"
        ax.set_title(f"{label}: ~{secs:.0f} s{title_h}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preview the tag_mission.py --search patterns")
    ap.add_argument("--arena", default=str(HERE / "tag_arena.yaml"))
    ap.add_argument("--plot", help="save a PNG of the patterns")
    ap.add_argument("--csv", metavar="DIR", help="save each pattern as search_plan_<name>.csv (t,x,y,z) in DIR")
    args = ap.parse_args()
    preview(load_plan(Path(args.arena)), args.plot, args.csv)
