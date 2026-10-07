import math
import numpy as np
import pytest

from src.env.car import (
    CarState,
    MAX_SPEED,
    step_physics,
    DRAFT_CONE_LENGTH,
    DRAFT_CONE_HALF_ANGLE,
    DRAFT_DRAG_REDUCTION,
    DRAFT_SPEED_BOOST,
    DRAFT_MIN_GAP,
    DRAFT_PEAK_GAP,
)
from src.env.rewards import (
    DEFAULT_POSITION_K,
    DEFAULT_POSITION_G0,
    DEFAULT_CONTACT_STEP_PENALTY,
    DEFAULT_CONTACT_PENALTY,
    compute_positional_reward,
)
from src.env.multi_racing_env import (
    MultiRacingEnv,
    AGENTS,
    compute_draft_intensity,
    DRAFT_CONE_LENGTH as ENV_DRAFT_CONE_LEN,
    DRAFT_CONE_HALF_ANGLE as ENV_DRAFT_CONE_ANGLE,
    DRAFT_DRAG_REDUCTION as ENV_DRAFT_DRAG_RED,
    DRAFT_SPEED_BOOST as ENV_DRAFT_SPEED_BOOST,
    POSITION_K as ENV_POS_K,
    POSITION_G0 as ENV_POS_G0,
)


def test_module_level_constants_exposed():
    assert DRAFT_CONE_LENGTH > 0.0
    assert DRAFT_CONE_HALF_ANGLE > 0.0
    assert DRAFT_DRAG_REDUCTION > 0.0
    assert DRAFT_SPEED_BOOST > 0.0
    assert DEFAULT_POSITION_K > 0.0
    assert DEFAULT_POSITION_G0 > 0.0

    assert ENV_DRAFT_CONE_LEN == DRAFT_CONE_LENGTH
    assert ENV_DRAFT_CONE_ANGLE == DRAFT_CONE_HALF_ANGLE
    assert ENV_DRAFT_DRAG_RED == DRAFT_DRAG_REDUCTION
    assert ENV_DRAFT_SPEED_BOOST == DRAFT_SPEED_BOOST
    assert ENV_POS_K == DEFAULT_POSITION_K
    assert ENV_POS_G0 == DEFAULT_POSITION_G0


def test_draft_intensity_zero_at_contact_and_below_minimum_gap():
    leader_pos = np.array([100.0, 100.0])
    leader_heading = 0.0

    touching_pos = np.array([100.0 - (DRAFT_MIN_GAP - 5.0), 100.0])
    intensity_below_gap = compute_draft_intensity(touching_pos, leader_pos, leader_heading, in_contact=False)
    assert intensity_below_gap == 0.0

    sweet_spot_pos = np.array([100.0 - DRAFT_PEAK_GAP, 100.0])
    intensity_at_contact = compute_draft_intensity(sweet_spot_pos, leader_pos, leader_heading, in_contact=True)
    assert intensity_at_contact == 0.0


def test_draft_intensity_positive_inside_cone_and_zero_outside():
    leader_pos = np.array([200.0, 200.0])
    leader_heading = 0.0

    pos_in_cone = np.array([200.0 - DRAFT_PEAK_GAP, 200.0])
    intensity_in = compute_draft_intensity(pos_in_cone, leader_pos, leader_heading, in_contact=False)
    assert 0.0 < intensity_in <= 1.0

    pos_ahead = np.array([250.0, 200.0])
    intensity_ahead = compute_draft_intensity(pos_ahead, leader_pos, leader_heading, in_contact=False)
    assert intensity_ahead == 0.0

    pos_too_far = np.array([200.0 - (DRAFT_CONE_LENGTH + 20.0), 200.0])
    intensity_far = compute_draft_intensity(pos_too_far, leader_pos, leader_heading, in_contact=False)
    assert intensity_far == 0.0

    pos_wide_lateral = np.array([200.0 - DRAFT_PEAK_GAP, 200.0 + 50.0])
    intensity_wide = compute_draft_intensity(pos_wide_lateral, leader_pos, leader_heading, in_contact=False)
    assert intensity_wide == 0.0


def test_speed_ceiling_exceeds_150_only_while_drafting():
    state_no_draft = CarState(x=0.0, y=0.0, heading=0.0, speed=150.0)
    next_state_no_draft, _ = step_physics(state_no_draft, throttle=1.0, steer=0.0, draft_intensity=0.0)
    assert next_state_no_draft.speed <= MAX_SPEED

    state_draft = CarState(x=0.0, y=0.0, heading=0.0, speed=150.0)
    next_state_draft, _ = step_physics(state_draft, throttle=1.0, steer=0.0, draft_intensity=1.0)
    assert next_state_draft.speed > 150.0
    assert next_state_draft.speed <= MAX_SPEED + DRAFT_SPEED_BOOST


def test_positional_reward_is_zero_sum():
    for diff in [-100.0, -35.2, 0.0, 12.4, 50.0, 250.0]:
        dist_a = 500.0 + diff
        dist_b = 500.0
        r_a = compute_positional_reward(dist_a, dist_b)
        r_b = compute_positional_reward(dist_b, dist_a)
        assert np.isclose(r_a + r_b, 0.0, atol=1e-8)


def test_per_step_contact_penalty_applies_every_step_in_contact():
    env = MultiRacingEnv(enable_draft=False, enable_position_reward=False)
    env.reset(seed=42)
    pos = env.track.centerline[0].copy()
    for a in AGENTS:
        env._state[a]["pos"][:] = pos
        env._state[a]["speed"] = 0.0

    actions = {a: np.zeros(2, dtype=np.float32) for a in AGENTS}
    _, rewards1, _, _, _ = env.step(actions)
    expected_step1 = 0.2 + DEFAULT_CONTACT_PENALTY + DEFAULT_CONTACT_STEP_PENALTY
    assert np.isclose(rewards1[AGENTS[0]], expected_step1)

    _, rewards2, _, _, _ = env.step(actions)
    expected_step2 = 0.2 + DEFAULT_CONTACT_STEP_PENALTY
    assert np.isclose(rewards2[AGENTS[0]], expected_step2)



def test_observation_dims_17_and_18_populated():
    env = MultiRacingEnv(enable_draft=True)
    env.reset(seed=42)
    leader = AGENTS[0]
    follower = AGENTS[1]

    leader_pos = env.track.centerline[20].copy()
    leader_heading = float(np.arctan2(env.track.tangents[20, 1], env.track.tangents[20, 0]))
    env._state[leader]["pos"][:] = leader_pos
    env._state[leader]["heading"] = leader_heading

    fwd = np.array([math.cos(leader_heading), math.sin(leader_heading)])
    follower_pos = leader_pos - fwd * DRAFT_PEAK_GAP
    env._state[follower]["pos"][:] = follower_pos
    env._state[follower]["heading"] = leader_heading

    obs_follower = env._build_obs(follower)
    obs_leader = env._build_obs(leader)

    assert obs_follower.shape == (19,)
    assert obs_follower[17] > 0.0
    assert obs_follower[18] == 0.0

    assert obs_leader.shape == (19,)
    assert obs_leader[17] == 0.0
    assert obs_leader[18] > 0.0


def test_position_swap_counter_in_info():
    env = MultiRacingEnv()
    env.reset(seed=42)
    actions = {a: np.zeros(2, dtype=np.float32) for a in AGENTS}

    _, _, _, _, infos1 = env.step(actions)
    initial_swaps = infos1[AGENTS[0]]["position_swaps"]
    assert initial_swaps == 0

    env._state[AGENTS[0]]["cumulative_distance"] = 1000.0
    env._state[AGENTS[1]]["cumulative_distance"] = 0.0
    _, _, _, _, infos2 = env.step(actions)

    env._state[AGENTS[1]]["cumulative_distance"] = 2000.0
    _, _, _, _, infos3 = env.step(actions)
    assert infos3[AGENTS[0]]["position_swaps"] >= 1
