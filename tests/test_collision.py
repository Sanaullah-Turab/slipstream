from __future__ import annotations

import math
import numpy as np
import pytest

from src.env.collision import (
    obb_vertices,
    obb_overlap,
    resolve_collision,
    classify_contact,
)

HALF_LEN = 20.0
HALF_W = 9.0


# ---------------------------------------------------------------------------
# OBB geometry
# ---------------------------------------------------------------------------

def test_obb_vertices_axis_aligned():
    pos = np.array([0.0, 0.0])
    verts = obb_vertices(pos, 0.0, HALF_LEN, HALF_W)
    xs = sorted(verts[:, 0])
    ys = sorted(verts[:, 1])
    assert xs[0] == pytest.approx(-HALF_LEN)
    assert xs[-1] == pytest.approx(HALF_LEN)
    assert ys[0] == pytest.approx(-HALF_W)
    assert ys[-1] == pytest.approx(HALF_W)


def test_obb_vertices_rotated_90():
    pos = np.array([0.0, 0.0])
    verts = obb_vertices(pos, math.pi / 2, HALF_LEN, HALF_W)
    xs = sorted(verts[:, 0])
    ys = sorted(verts[:, 1])
    assert xs[0] == pytest.approx(-HALF_W, abs=1e-6)
    assert xs[-1] == pytest.approx(HALF_W, abs=1e-6)
    assert ys[0] == pytest.approx(-HALF_LEN, abs=1e-6)
    assert ys[-1] == pytest.approx(HALF_LEN, abs=1e-6)


# ---------------------------------------------------------------------------
# SAT overlap detection
# ---------------------------------------------------------------------------

def test_no_overlap_far_apart():
    pos_a = np.array([0.0, 0.0])
    pos_b = np.array([200.0, 0.0])
    overlapping, _, _ = obb_overlap(pos_a, 0.0, pos_b, 0.0, HALF_LEN, HALF_W)
    assert not overlapping


def test_overlap_identical_positions():
    pos = np.array([0.0, 0.0])
    overlapping, _, pen = obb_overlap(pos, 0.0, pos, 0.0, HALF_LEN, HALF_W)
    assert overlapping
    assert pen > 0.0


def test_overlap_side_by_side_touching():
    pos_a = np.array([0.0, 0.0])
    pos_b = np.array([0.0, 2.0 * HALF_W - 0.5])
    overlapping, normal, _ = obb_overlap(pos_a, 0.0, pos_b, 0.0, HALF_LEN, HALF_W)
    assert overlapping
    # Normal should point roughly along y-axis (from b toward a)
    assert abs(normal[1]) > 0.5


def test_no_overlap_side_by_side_separated():
    pos_a = np.array([0.0, 0.0])
    pos_b = np.array([0.0, 2.0 * HALF_W + 1.0])
    overlapping, _, _ = obb_overlap(pos_a, 0.0, pos_b, 0.0, HALF_LEN, HALF_W)
    assert not overlapping


def test_overlap_nose_to_tail():
    pos_a = np.array([0.0, 0.0])
    pos_b = np.array([2.0 * HALF_LEN - 1.0, 0.0])
    overlapping, normal, _ = obb_overlap(pos_a, 0.0, pos_b, 0.0, HALF_LEN, HALF_W)
    assert overlapping
    # Normal points from b to a, so should be roughly -x
    assert normal[0] < 0.0


def test_normal_direction_convention():
    """Normal must always point from b toward a."""
    pos_a = np.array([0.0, 0.0])
    pos_b = np.array([0.0, HALF_W])
    overlapping, normal, _ = obb_overlap(pos_a, 0.0, pos_b, 0.0, HALF_LEN, HALF_W)
    if overlapping:
        d = pos_a - pos_b
        assert np.dot(d, normal) >= 0.0


# ---------------------------------------------------------------------------
# Impulse-based collision resolution
# ---------------------------------------------------------------------------

def test_resolve_head_on():
    # normal points from b (at origin) toward a (to the right, pos=[1,0])
    # For them to be approaching, a moves left (-x) and b moves right (+x)
    vel_a = np.array([-10.0, 0.0])
    vel_b = np.array([10.0, 0.0])
    normal = np.array([1.0, 0.0])
    va2, vb2 = resolve_collision(np.zeros(2), vel_a, np.zeros(2), vel_b, normal, restitution=0.3)
    # After collision: a should bounce right (positive x), b should bounce left (negative x)
    assert va2[0] > vel_a[0]
    assert vb2[0] < vel_b[0]


def test_resolve_no_impulse_separating():
    vel_a = np.array([10.0, 0.0])
    vel_b = np.array([5.0, 0.0])
    normal = np.array([1.0, 0.0])
    va2, vb2 = resolve_collision(np.zeros(2), vel_a, np.zeros(2), vel_b, normal, restitution=0.3)
    # a faster than b along normal -> they are separating -> no impulse
    np.testing.assert_array_equal(va2, vel_a)
    np.testing.assert_array_equal(vb2, vel_b)


def test_resolve_momentum_conserved():
    vel_a = np.array([8.0, 2.0])
    vel_b = np.array([-3.0, 1.0])
    normal = np.array([1.0, 0.0])
    va2, vb2 = resolve_collision(np.zeros(2), vel_a, np.zeros(2), vel_b, normal)
    np.testing.assert_allclose(va2 + vb2, vel_a + vel_b, atol=1e-10)


def test_resolve_low_restitution_dissipates_energy():
    vel_a = np.array([-10.0, 0.0])
    vel_b = np.array([10.0, 0.0])
    normal = np.array([1.0, 0.0])
    va2, vb2 = resolve_collision(np.zeros(2), vel_a, np.zeros(2), vel_b, normal, restitution=0.2)
    ke_before = np.dot(vel_a, vel_a) + np.dot(vel_b, vel_b)
    ke_after = np.dot(va2, va2) + np.dot(vb2, vb2)
    assert ke_after < ke_before


@pytest.mark.parametrize("restitution", [0.2, 0.25, 0.3, 0.35, 0.4])
def test_resolve_restitution_range_energy_dissipation(restitution):
    vel_a = np.array([-15.0, 2.0])
    vel_b = np.array([12.0, -1.0])
    normal = np.array([1.0, 0.0])
    va2, vb2 = resolve_collision(np.zeros(2), vel_a, np.zeros(2), vel_b, normal, restitution=restitution)
    ke_before = np.dot(vel_a, vel_a) + np.dot(vel_b, vel_b)
    ke_after = np.dot(va2, va2) + np.dot(vb2, vb2)
    assert ke_after < ke_before
    rel_approach = abs(float(np.dot(vel_a - vel_b, normal)))
    rel_separate = abs(float(np.dot(va2 - vb2, normal)))
    assert rel_separate == pytest.approx(restitution * rel_approach, rel=1e-5)


def test_no_tunneling_closing_head_on_max_speed():
    from src.env.car import MAX_SPEED, DT
    step_dist = MAX_SPEED * DT
    for gap in np.linspace(0.01, step_dist * 2.0 - 0.01, 50):
        pos_a_0 = np.array([0.0, 0.0])
        pos_b_0 = np.array([2.0 * HALF_LEN + gap, 0.0])
        init_overlap, _, _ = obb_overlap(pos_a_0, 0.0, pos_b_0, 0.0, HALF_LEN, HALF_W)
        assert not init_overlap

        pos_a_1 = pos_a_0 + np.array([step_dist, 0.0])
        pos_b_1 = pos_b_0 - np.array([step_dist, 0.0])
        next_overlap, _, pen = obb_overlap(pos_a_1, 0.0, pos_b_1, 0.0, HALF_LEN, HALF_W)
        assert next_overlap
        assert pen > 0.0



# ---------------------------------------------------------------------------
# Fault classification
# ---------------------------------------------------------------------------

def _make_vel(heading: float, speed: float) -> np.ndarray:
    return np.array([math.cos(heading) * speed, math.sin(heading) * speed])


def test_classify_rear_end_follower_fault():
    heading_leader = 0.0
    # Normal points from follower (behind) toward leader (ahead), so along +x
    normal = np.array([1.0, 0.0])
    # rear_dir = -fwd = [-1, 0]; cos(angle between normal and rear_dir) = -1 -> angle = pi, not a rear-end
    # Actually: the leader's rear faces -x. The follower hits the leader from behind,
    # so the contact normal pointing FROM follower TO leader is +x, i.e. pointing away from the rear.
    # Let's set normal to -x (points from leader's rear toward the follower) to test properly.
    normal = np.array([-1.0, 0.0])
    result = classify_contact(
        normal=normal,
        heading_leader=heading_leader,
        vel_a=_make_vel(0.0, 5.0),
        vel_b=_make_vel(0.0, 10.0),
        leader_id=0,
        lateral_history_leader=[0.0] * 15,
        follower_lateral=0.0,
    )
    assert result == "follower_fault"


def test_classify_leader_swerve_fault():
    heading_leader = 0.0
    # Side contact, not a rear-end
    normal = np.array([0.0, 1.0])
    # Leader started at 0.0 and swerved to 0.2 within the lookback window
    history = [0.0] * 15 + [0.2] * 5
    result = classify_contact(
        normal=normal,
        heading_leader=heading_leader,
        vel_a=_make_vel(0.0, 8.0),
        vel_b=_make_vel(0.0, 8.0),
        leader_id=0,
        lateral_history_leader=history,
        follower_lateral=0.5,
        lookback=10,
        lateral_move_threshold=0.1,
    )
    assert result == "leader_fault"


def test_classify_neutral_side_contact_no_history():
    heading_leader = 0.0
    normal = np.array([0.0, 1.0])
    history = [0.0] * 15
    result = classify_contact(
        normal=normal,
        heading_leader=heading_leader,
        vel_a=_make_vel(0.0, 8.0),
        vel_b=_make_vel(0.0, 8.0),
        leader_id=0,
        lateral_history_leader=history,
        follower_lateral=0.5,
    )
    assert result == "neutral"


def test_classify_boundary_angle_is_neutral():
    heading_leader = 0.0
    # Exactly at 45-degree boundary from rear axis -> should not be follower_fault
    angle = math.radians(46.0)
    normal = np.array([math.cos(math.pi + angle), math.sin(math.pi + angle)])
    normal = normal / np.linalg.norm(normal)
    history = [0.0] * 15
    result = classify_contact(
        normal=normal,
        heading_leader=heading_leader,
        vel_a=_make_vel(0.0, 8.0),
        vel_b=_make_vel(0.0, 8.0),
        leader_id=0,
        lateral_history_leader=history,
        follower_lateral=0.0,
    )
    assert result == "neutral"
