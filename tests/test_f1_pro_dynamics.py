from __future__ import annotations

import math
import numpy as np
import pytest

from src.env.car import (
    CarState,
    CarParams,
    DEFAULT_PARAMS,
    SHANGHAI_PARAMS,
    compute_tyre_grip,
    step_physics,
)
from src.env.multi_racing_env import MultiRacingEnv, AGENTS


def test_compute_tyre_grip_temperatures():
    grip_cold = compute_tyre_grip(50.0)
    grip_warm = compute_tyre_grip(80.0)
    grip_opt = compute_tyre_grip(100.0)
    grip_hot = compute_tyre_grip(120.0)
    grip_severe = compute_tyre_grip(135.0)

    assert grip_cold < grip_warm < grip_opt
    assert grip_opt == pytest.approx(1.05)
    assert grip_hot < grip_opt
    assert grip_severe < grip_hot


def test_trail_braking_combined_slip_reduces_lateral_steer():
    state = CarState(x=0.0, y=0.0, heading=0.0, speed=120.0, tyre_temp=100.0)
    _, hr_neutral = step_physics(state, throttle=0.0, steer=0.5, params=SHANGHAI_PARAMS)
    _, hr_heavy_brake = step_physics(state, throttle=-1.0, steer=0.5, params=SHANGHAI_PARAMS)

    assert abs(hr_heavy_brake) < abs(hr_neutral)


def test_kerb_surface_grip_and_rolling_resistance():
    state = CarState(x=0.0, y=0.0, heading=0.0, speed=100.0, tyre_temp=100.0)
    new_tarmac, _ = step_physics(state, throttle=0.0, steer=0.0, surface_grip=1.0, rolling_factor=1.0)
    new_kerb, _ = step_physics(state, throttle=0.0, steer=0.0, surface_grip=0.92, rolling_factor=1.8)
    new_offtrack, _ = step_physics(state, throttle=0.0, steer=0.0, surface_grip=0.35, rolling_factor=8.0)

    assert new_tarmac.speed > new_kerb.speed > new_offtrack.speed


def test_tyre_heating_under_heavy_cornering():
    state = CarState(x=0.0, y=0.0, heading=0.0, speed=150.0, tyre_temp=80.0)
    st = state
    for _ in range(50):
        st, _ = step_physics(st, throttle=0.8, steer=0.4, params=SHANGHAI_PARAMS)
    assert st.tyre_temp > 85.0


def test_single_defensive_move_vs_weaving_penalty():
    env = MultiRacingEnv(enable_draft=True, enable_position_reward=True)
    env.reset(seed=42)
    s0 = env._state[AGENTS[0]]
    s1 = env._state[AGENTS[1]]
    env._current_leader = AGENTS[0]
    s0["pos"][:] = np.array([500.0, 500.0])
    s1["pos"][:] = np.array([480.0, 500.0])
    s0["speed"] = 180.0
    s1["speed"] = 190.0
    s0["gap_to_leader_seconds"] = 0.0
    s1["gap_to_leader_seconds"] = 0.2
    s0["defensive_moves"] = 0
    s0["last_defensive_lat"] = 0.0
    s0["lateral"] = 3.0

    r1 = env._compute_tactical_rewards()
    assert s0["defensive_moves"] == 1
    assert r1[AGENTS[0]] >= 0.0

    s0["lateral"] = -3.0
    r2 = env._compute_tactical_rewards()
    assert s0["defensive_moves"] >= 2
    assert r2[AGENTS[0]] < 0.0


def test_moving_under_braking_penalized():
    env = MultiRacingEnv(enable_draft=True, enable_position_reward=True)
    env.reset(seed=42)
    s0 = env._state[AGENTS[0]]
    s1 = env._state[AGENTS[1]]
    env._current_leader = AGENTS[0]
    s0["pos"][:] = np.array([500.0, 500.0])
    s1["pos"][:] = np.array([485.0, 500.0])
    s0["speed"] = 180.0
    s1["speed"] = 190.0
    s0["gap_to_leader_seconds"] = 0.0
    s1["gap_to_leader_seconds"] = 0.2
    s0["throttle"] = -0.8
    s0["steer"] = 0.5
    s0["prev_steer"] = 0.0

    rewards = env._compute_tactical_rewards()
    assert rewards[AGENTS[0]] <= -2.0
