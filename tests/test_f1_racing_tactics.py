from __future__ import annotations

import math
import numpy as np
import pytest

from src.env.multi_racing_env import MultiRacingEnv, AGENTS
from src.env.track import Track


def test_track_curvature_computation():
    track = Track(circuit="shanghai")
    assert hasattr(track, "curvatures")
    assert len(track.curvatures) == len(track.centerline)
    assert np.any(track.curvatures > 0.001)
    assert np.any(track.curvatures < -0.001)


def test_leader_defensive_line_reward():
    env = MultiRacingEnv(enable_draft=True, enable_position_reward=True)
    env.reset(seed=42)
    s0 = env._state[AGENTS[0]]
    s1 = env._state[AGENTS[1]]
    s0["speed"] = 80.0
    s1["speed"] = 85.0
    s0["gap_to_leader_seconds"] = 0.0
    s1["gap_to_leader_seconds"] = 0.4
    rewards = env._compute_tactical_rewards()
    assert AGENTS[0] in rewards
    assert AGENTS[1] in rewards


def test_follower_attacking_pullout_reward():
    env = MultiRacingEnv(enable_draft=True, enable_position_reward=True)
    env.reset(seed=42)
    s0 = env._state[AGENTS[0]]
    s1 = env._state[AGENTS[1]]
    s0["pos"][:] = np.array([200.0, 500.0])
    s1["pos"][:] = np.array([185.0, 515.0])
    s0["speed"] = 80.0
    s1["speed"] = 90.0
    s1["gap_to_leader_seconds"] = 0.3
    rewards = env._compute_tactical_rewards()
    assert isinstance(rewards[AGENTS[1]], float)


def test_clean_overtake_reward_applied():
    env = MultiRacingEnv(enable_draft=True, enable_position_reward=True)
    env.reset(seed=42)
    car_len = 26.0
    env._current_leader = AGENTS[0]
    env._state[AGENTS[0]]["start_offset"] = 100.0
    env._state[AGENTS[1]]["start_offset"] = 100.0 + car_len + 5.0
    actions = {a: np.array([0.0, 0.5], dtype=np.float32) for a in AGENTS}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert env._current_leader == AGENTS[1]
    assert env._position_swaps >= 1
