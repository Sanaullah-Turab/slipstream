from __future__ import annotations

import math
import numpy as np
import pytest

from src.env.collision import (
    obb_vertices,
    obb_overlap,
    resolve_collision,
    classify_contact,
    resolve_penetration,
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


def test_env_rear_end_classification_symmetry():
    from src.env.multi_racing_env import MultiRacingEnv, AGENTS

    env_a = MultiRacingEnv(enable_draft=False, enable_position_reward=False)
    env_a.reset(seed=42)
    center = env_a.track.centerline[10].copy()
    tang = env_a.track.tangents[10].copy()
    heading = float(np.arctan2(tang[1], tang[0]))
    fwd = np.array([math.cos(heading), math.sin(heading)])

    env_a._state[AGENTS[0]]["pos"][:] = center
    env_a._state[AGENTS[0]]["heading"] = heading
    env_a._state[AGENTS[0]]["speed"] = 100.0
    env_a._state[AGENTS[0]]["cumulative_distance"] = 500.0

    env_a._state[AGENTS[1]]["pos"][:] = center - fwd * 20.0
    env_a._state[AGENTS[1]]["heading"] = heading
    env_a._state[AGENTS[1]]["speed"] = 140.0
    env_a._state[AGENTS[1]]["cumulative_distance"] = 480.0

    actions = {a: np.array([0.0, 1.0], dtype=np.float32) for a in AGENTS}
    _, _, _, _, infos_a = env_a.step(actions)
    fault_a = infos_a[AGENTS[1]]["fault_log"]["follower"]

    env_b = MultiRacingEnv(enable_draft=False, enable_position_reward=False)
    env_b.reset(seed=42)
    env_b._state[AGENTS[1]]["pos"][:] = center
    env_b._state[AGENTS[1]]["heading"] = heading
    env_b._state[AGENTS[1]]["speed"] = 100.0
    env_b._state[AGENTS[1]]["cumulative_distance"] = 500.0

    env_b._state[AGENTS[0]]["pos"][:] = center - fwd * 20.0
    env_b._state[AGENTS[0]]["heading"] = heading
    env_b._state[AGENTS[0]]["speed"] = 140.0
    env_b._state[AGENTS[0]]["cumulative_distance"] = 480.0

    _, _, _, _, infos_b = env_b.step(actions)
    fault_b = infos_b[AGENTS[0]]["fault_log"]["follower"]

    assert fault_a == 1
    assert fault_b == 1


def test_resolve_penetration_separates_along_normal():
    pos_a = np.array([10.0, 0.0])
    pos_b = np.array([0.0, 0.0])
    normal = np.array([1.0, 0.0])
    new_a, new_b = resolve_penetration(pos_a, pos_b, normal, 4.0, epsilon=0.02)
    assert new_a[0] == pytest.approx(11.99)
    assert new_b[0] == pytest.approx(-1.99)
    assert np.linalg.norm(new_a - new_b) == pytest.approx(13.98)


def test_resolve_penetration_no_shift_below_epsilon():
    pos_a = np.array([5.0, 2.0])
    pos_b = np.array([1.0, 1.0])
    normal = np.array([1.0, 0.0])
    new_a, new_b = resolve_penetration(pos_a, pos_b, normal, 0.01, epsilon=0.02)
    np.testing.assert_array_equal(new_a, pos_a)
    np.testing.assert_array_equal(new_b, pos_b)


def test_resolve_penetration_center_of_mass_conserved():
    pos_a = np.array([12.3, -4.5])
    pos_b = np.array([7.1, 8.9])
    normal = np.array([0.6, 0.8])
    new_a, new_b = resolve_penetration(pos_a, pos_b, normal, 6.0, epsilon=0.02)
    np.testing.assert_allclose(0.5 * (new_a + new_b), 0.5 * (pos_a + pos_b))


def test_resolve_penetration_reduces_obb_overlap():
    pos_a = np.array([21.0, 0.0])
    pos_b = np.array([0.0, 0.0])
    ov, n, pen = obb_overlap(pos_a, 0.0, pos_b, 0.0, HALF_LEN, HALF_W)
    assert ov
    assert pen == pytest.approx(19.0)
    p0, p1 = resolve_penetration(pos_a, pos_b, n, pen, epsilon=0.02)
    ov2, _, pen2 = obb_overlap(p0, 0.0, p1, 0.0, HALF_LEN, HALF_W)
    assert ov2
    assert pen2 == pytest.approx(0.02, abs=1e-5)


def test_env_collision_positional_separation():
    from src.env.multi_racing_env import MultiRacingEnv, AGENTS, CAR_HALF_LEN, CAR_HALF_WIDTH

    env = MultiRacingEnv(enable_draft=False, enable_position_reward=False)
    env.reset(seed=42)
    center = env.track.centerline[10].copy()
    tang = env.track.tangents[10].copy()
    heading = float(np.arctan2(tang[1], tang[0]))
    fwd = np.array([math.cos(heading), math.sin(heading)])

    env._state[AGENTS[0]]["pos"][:] = center
    env._state[AGENTS[0]]["heading"] = heading
    env._state[AGENTS[0]]["speed"] = 80.0

    env._state[AGENTS[1]]["pos"][:] = center - fwd * 15.0
    env._state[AGENTS[1]]["heading"] = heading
    env._state[AGENTS[1]]["speed"] = 150.0

    actions = {a: np.array([0.0, 1.0], dtype=np.float32) for a in AGENTS}
    env.step(actions)

    p0 = env._state[AGENTS[0]]["pos"]
    h0 = env._state[AGENTS[0]]["heading"]
    p1 = env._state[AGENTS[1]]["pos"]
    h1 = env._state[AGENTS[1]]["heading"]

    ov, _, pen = obb_overlap(p0, h0, p1, h1, CAR_HALF_LEN, CAR_HALF_WIDTH)
    if ov:
        assert pen <= 0.05


def test_env_collision_shove_off_track_respawns():
    from src.env.multi_racing_env import MultiRacingEnv, AGENTS

    env = MultiRacingEnv(enable_draft=False, enable_position_reward=False)
    env.reset(seed=42)
    center = env.track.centerline[10].copy()
    tang = env.track.tangents[10].copy()
    norm = env.track.normals[10].copy()
    heading = float(np.arctan2(tang[1], tang[0]))

    half_w = env.track.half_width
    env._state[AGENTS[0]]["pos"][:] = center + norm * (half_w - 0.5)
    env._state[AGENTS[0]]["heading"] = heading
    env._state[AGENTS[0]]["speed"] = 0.0

    env._state[AGENTS[1]]["pos"][:] = center + norm * (half_w - 5.0)
    env._state[AGENTS[1]]["heading"] = heading
    env._state[AGENTS[1]]["speed"] = 0.0

    actions = {a: np.array([0.0, 0.0], dtype=np.float32) for a in AGENTS}
    _, _, _, _, infos = env.step(actions)

    assert infos[AGENTS[0]]["respawn"] or infos[AGENTS[0]]["respawns"] >= 1


