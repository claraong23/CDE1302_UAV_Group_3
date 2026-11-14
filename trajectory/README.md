# Trajectories

CSV files are `t,x,y,z`: seconds, and metres from the start point (x forward,
y right; z = 0 holds the takeoff height). The drone keeps its takeoff heading
(nose +x) for the whole path. All paths start and end at rest at (0, 0) and
peak at 0.297 m/s; `tag_mission.py` refuses anything faster than 0.30 m/s.

Open a `.m` file in MATLAB and press Run to regenerate its CSV and PNG.

## Final runs

| Path | Script | CSV | Time |
|---|---|---|---|
| Lopsided ellipse | [`make_ellipse.py`](make_ellipse.py) | [`ellipse_asym_traj.csv`](ellipse_asym_traj.csv) | 38.4 s |
| Elephant | [`elephant_min_snap_demo.m`](elephant_min_snap_demo.m) | [`elephant_min_snap_traj.csv`](elephant_min_snap_traj.csv) | 79.0 s |
| Triangle | [`triangle_min_snap_demo.m`](triangle_min_snap_demo.m) | [`triangle_min_snap_traj.csv`](triangle_min_snap_traj.csv) | 37.8 s |
| Rectangle | [`rectangle_min_snap_demo.m`](rectangle_min_snap_demo.m) | [`rectangle_min_snap_traj.csv`](rectangle_min_snap_traj.csv) | 50.4 s |

**Lopsided ellipse** ([`ellipse_asym_traj.png`](ellipse_asym_traj.png)): 4.6 m long, 1.3 m out on the
left and 0.9 m on the right, shifted 0.2 m left, clockwise. Constant speed for
the whole lap, sampled at equal arc length, with a 2 s slow-down into (0, 0).

```bash
python trajectory/make_ellipse.py --forward --left 1.3 --right 0.9 --shift 0.2 --plot --out trajectory/ellipse_asym_traj.csv
```

**Elephant** ([`elephant_min_snap.png`](elephant_min_snap.png)): minimum-snap through 15 waypoints.
One 7th-order polynomial per leg; position passes through every waypoint and
velocity, acceleration and jerk are continuous. Leg times are balanced so each
leg peaks at about the same speed.

**Triangle** ([`triangle_min_snap.png`](triangle_min_snap.png)): (0, 0) → (3.8, 1.3) → (1.0, −1.5) →
(0, 0). Edges are straight at a steady cruise speed; each corner is a
minimum-snap turn between the two edges, rounded by `corner_radius` (0.3 m).

**Rectangle** ([`rectangle_min_snap.png`](rectangle_min_snap.png)): same method as the triangle. A
diagonal from (0, 0) to the near-left corner, clockwise round the rectangle,
then a diagonal home. The rectangle is set over the likely target spots, pulled
in where the camera still covers them (checked with
`search_patterns.camera_coverage`).

Fly any of them with, for example:

```bash
python laptop/tag_mission.py --drone-id 3 --trajectory trajectory/triangle_min_snap_traj.csv --takeoff-wait 14 --no-return-home --confirm
```

## How the elephant was made

These two scripts are for comparison only and were not flown (report
Section 3.2):

- [`elephant_sparse_min_snap_demo.m`](elephant_sparse_min_snap_demo.m) → [`elephant_sparse_min_snap_traj.csv`](elephant_sparse_min_snap_traj.csv):
  the same minimum-snap solver through only 8 of the 15 waypoints. With so
  few constraints the path swings far outside the waypoints.
- [`elephant_catmull_rom_demo.m`](elephant_catmull_rom_demo.m) → [`elephant_catmull_rom_traj.csv`](elephant_catmull_rom_traj.csv): a
  Catmull-Rom spline through the same 8 waypoints. Each piece depends only on
  its neighbouring points, so the curve stays close to the intended shape, but
  it is only C¹ and does not minimise snap. We used it as a reference to pick
  the 7 extra waypoints that make the minimum-snap elephant follow the shape.

## Test paths

Not used in the final runs: [`ellipse_min_snap_demo.m`](ellipse_min_snap_demo.m) and
[`ellipse_big_min_snap_demo.m`](ellipse_big_min_snap_demo.m), minimum-snap ellipses of 2.0 × 1.0 m and
4.6 × 3.0 m.
