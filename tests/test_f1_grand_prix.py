from __future__ import annotations

import numpy as np
import pytest

from src.env.multi_racing_env import MultiRacingEnv, AGENTS


def test_f1_starting_grid_spawn():
    env = MultiRacingEnv()
    obs, infos = env.reset(seed=42)
    s0 = env._state[AGENTS[0]]
    s1 = env._state[AGENTS[1]]
    assert s0["progress"] > 0.90 or s0["progress"] < 0.10
    assert s1["progress"] > 0.90 or s1["progress"] < 0.10
    assert s0["laps"] == 0
    assert s1["laps"] == 0
    assert s0["lap_start_time"] is None
    assert s1["lap_start_time"] is None


def test_f1_finish_line_lap_start_and_timing():
    env = MultiRacingEnv()
    env.reset(seed=42)
    s0 = env._state[AGENTS[0]]
    s0["pos"][:] = env.track.centerline[-2]
    s0["speed"] = 120.0
    actions = {a: np.array([0.0, 1.0], dtype=np.float32) for a in AGENTS}
    env.step(actions)
    assert s0["laps"] >= 1
    assert s0["lap_start_time"] is not None


def test_f1_gap_in_seconds_computed():
    env = MultiRacingEnv()
    env.reset(seed=42)
    actions = {a: np.array([0.0, 1.0], dtype=np.float32) for a in AGENTS}
    _, _, _, _, infos = env.step(actions)
    for a in AGENTS:
        assert "gap_to_leader_seconds" in infos[a]
        assert infos[a]["gap_to_leader_seconds"] >= 0.0


def test_f1_render_runs_with_shanghai():
    env = MultiRacingEnv(render_mode="rgb_array")
    env.reset(seed=42)
    actions = {a: np.array([0.0, 1.0], dtype=np.float32) for a in AGENTS}
    env.step(actions)
    frame = env.render()
    assert frame is not None
    assert frame.shape[0] >= 600
    assert frame.shape[1] >= 1000
    env.close()
