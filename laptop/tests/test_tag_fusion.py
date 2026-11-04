"""tag_fusion.final_pose and its use in tag_mission.TagLocaliser."""
import csv
import io
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tag_mission as tm  # noqa: E402
from tag_fusion import Sighting, final_pose  # noqa: E402


def sight(x, y, sep=0.3, rng=0.6, hdg=0.0, nk=1, speed=0.0, ang=float("nan")):
    return Sighting(x=x, y=y, method="tag", sep=sep, rng=rng, ang=ang, nk=nk, speed=speed, hdg=hdg)


def test_outlier_is_dropped_and_mean_is_close():
    rng = np.random.default_rng(0)
    rows = [sight(0.30 + e[0], -0.30 + e[1], hdg=h)
            for e, h in zip(rng.normal(0, 0.005, (20, 2)), np.linspace(0, 120, 20))]
    rows.append(sight(0.60, 0.10))                    # a bad sighting 50 cm off
    f = final_pose(rows)
    assert f["kept"] == 20 and f["n"] == 21
    assert math.hypot(f["mean"][0] - 0.30, f["mean"][1] + 0.30) < 0.005
    assert f["confirmed"]


def test_near_sightings_outweigh_far_ones():
    rows = [sight(0.30, 0.0, sep=0.1, rng=0.4)] * 5 + [sight(0.34, 0.0, sep=1.0, rng=1.1)] * 5
    assert final_pose(rows)["mean"][0] < 0.31


def test_one_heading_is_not_confirmed():
    assert not final_pose([sight(0.3, 0.3, hdg=10.0)] * 10)["confirmed"]


class _Poses:
    def at(self, t):
        return (0.0, 0.0, 0.0)


def test_localiser_places_target_from_drone_detections(tmp_path):
    """Drone mode (no tag rotation): XIAO 0.3 m behind tag 2, facing +x."""
    arena_file = tmp_path / "arena.yaml"
    arena_file.write_text("tag_size_m: 0.126\nknown_tags:\n  2: [0.0, 0.0, 0]\n"
                          "targets: [10]\ntarget_truth:\n  10: [0.30, -0.30]\n")
    arena = tm.Arena(arena_file)
    corr = tm.Correction()
    corr.assume_start(0.0, 0.0, 0.0)
    R_bc = tm.r_body_cam(45.0)
    loc = tm.TagLocaliser("127.0.0.1", 1, arena, _Poses(), corr, None, R_bc, tmp_path / "s.csv")
    p_bc = loc.p_bc
    drone = np.array([-0.30, 0.0, -0.50])            # arena, z down -> 0.5 m up

    def det(tid, where):
        t_cam = R_bc.T @ (np.array([*where, 0.0]) - drone - p_bc)
        return types.SimpleNamespace(id=tid, t=tuple(t_cam), R=None,
                                     range_m=float(np.linalg.norm(t_cam)), decision_margin=80.0)

    buf = io.StringIO()
    writer = csv.writer(buf)
    for i in range(8):
        loc._process(float(i), [det(2, (0.0, 0.0)), det(10, (0.30, -0.30))], writer)
    (tid, f, method, _), = loc.final_poses()
    assert (tid, method, f["n"]) == (10, "tag", 8)
    assert f["mean"] == pytest.approx([0.30, -0.30], abs=1e-6)
    assert len(buf.getvalue().splitlines()[0].split(",")) == 15
    tm.print_results(loc, arena)                     # must not raise
