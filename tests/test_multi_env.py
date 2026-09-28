from __future__ import annotations

import numpy as np
import pytest

from src.env.multi_racing_env import MultiRacingEnv, AGENTS
from src.env.vec_multi import TwoCarVecEnv
from src.env.car import MAX_SPEED, DT
from src.env.racing_env import MAX_STEPS


def make_env(seed: int = 0) -> MultiRacingEnv:
    env = MultiRacingEnv()
    env.reset(seed=seed)
    return env


def zero_actions(env: MultiRacingEnv) -> dict:
    return {a: np.zeros(2, dtype=np.float32) for a in env.agents}


def _force_off_track(env: MultiRacingEnv, agent: str) -> None:
    env._state[agent]["pos"][:] = np.array([0.0, 0.0])


def test_reset_returns_two_agents():
    env = MultiRacingEnv()
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == set(AGENTS)
    assert set(info.keys()) == set(AGENTS)


def test_obs_shape():
    env = MultiRacingEnv()
    obs, _ = env.reset(seed=0)
    for a in AGENTS:
        assert obs[a].shape == (15,)


def test_step_all_keys():
    env = make_env()
    obs, rew, terms, truncs, infos = env.step(zero_actions(env))
    for a in AGENTS:
        assert a in obs and a in rew and a in terms and a in truncs and a in infos


def test_info_keys():
    env = make_env()
    _, _, _, _, infos = env.step(zero_actions(env))
    expected = {"laps", "progress", "speed", "respawns", "collision_count", "cumulative_distance"}
    assert expected.issubset(set(infos[AGENTS[0]].keys()))


def test_no_spurious_lap():
    env = make_env(seed=7)
    _, _, _, _, infos = env.step(zero_actions(env))
    for a in AGENTS:
        assert infos[a]["laps"] == 0


def test_spawn_randomized():
    env = MultiRacingEnv()
    env.reset(seed=0)
    pos_seed0 = {a: env._state[a]["pos"].copy() for a in AGENTS}
    env.reset(seed=1)
    pos_seed1 = {a: env._state[a]["pos"].copy() for a in AGENTS}
    different = any(not np.allclose(pos_seed0[a], pos_seed1[a]) for a in AGENTS)
    assert different


def test_seed_determinism():
    env1, env2 = MultiRacingEnv(), MultiRacingEnv()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    for a in AGENTS:
        np.testing.assert_array_equal(obs1[a], obs2[a])
    acts = {a: np.array([0.5, 0.5], dtype=np.float32) for a in AGENTS}
    for _ in range(10):
        r1 = env1.step(acts)
        r2 = env2.step(acts)
    for a in AGENTS:
        np.testing.assert_array_almost_equal(r1[0][a], r2[0][a])


def test_respawn_on_crash():
    env = make_env(seed=0)
    _force_off_track(env, AGENTS[0])
    _, _, _, _, infos = env.step(zero_actions(env))
    assert infos[AGENTS[0]]["respawns"] == 1
    assert env._state[AGENTS[0]]["speed"] == 0.0
    assert env.track.get_track_state(env._state[AGENTS[0]]["pos"]).on_track


def test_cumulative_distance_survives_respawn():
    env = make_env(seed=0)
    acts = {a: np.array([0.0, 1.0], dtype=np.float32) for a in AGENTS}
    for _ in range(5):
        env.step(acts)

    dist_before = env._state[AGENTS[0]]["cumulative_distance"]
    _force_off_track(env, AGENTS[0])
    _, _, _, _, infos = env.step(zero_actions(env))
    dist_after = infos[AGENTS[0]]["cumulative_distance"]
    delta = dist_after - dist_before

    assert delta >= 0.0, "cumulative_distance dropped across respawn"
    max_step_dist = MAX_SPEED * DT * 2
    assert delta < max_step_dist, (
        f"cumulative_distance jumped by {delta:.2f} on respawn step (expected < {max_step_dist:.2f})"
    )


def test_collision_flag():
    env = MultiRacingEnv()
    env.reset(seed=0)
    pos = env.track.centerline[0].copy()
    for a in AGENTS:
        env._state[a]["pos"][:] = pos

    _, _, _, _, infos = env.step(zero_actions(env))
    assert infos[AGENTS[0]]["collision_count"] == 1
    assert infos[AGENTS[1]]["collision_count"] == 1

    _, _, _, _, infos2 = env.step(zero_actions(env))
    assert infos2[AGENTS[0]]["collision_count"] == 1


def test_truncation():
    env = MultiRacingEnv()
    env.reset(seed=0)
    acts = zero_actions(env)
    truncs: dict = {}
    terms: dict = {}
    for _ in range(MAX_STEPS):
        obs, rew, terms, truncs, infos = env.step(acts)
    for a in AGENTS:
        assert truncs[a] is True
        assert terms[a] is False
    assert env.agents == []


def test_vec_env_shape():
    env = TwoCarVecEnv()
    obs = np.asarray(env.reset())
    assert obs.shape == (2, 15)
    actions = np.stack([env.action_space.sample() for _ in range(2)])
    obs_step, rewards, dones, infos = env.step(actions)
    obs_step = np.asarray(obs_step)
    assert obs_step.shape == (2, 15)
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
    env.close()


def test_vec_env_truncation_terminal_obs():
    vec = TwoCarVecEnv()
    vec.reset()
    actions = np.zeros((2, 2), dtype=np.float32)
    infos: list = []
    for _ in range(MAX_STEPS):
        obs, rew, dones, infos = vec.step(actions)
    for info in infos:
        assert "terminal_observation" in info
        assert info["TimeLimit.truncated"] is True
        assert info["terminal_observation"].shape == (15,)
    vec.close()


def test_parallel_api():
    from pettingzoo.test import parallel_api_test
    env = MultiRacingEnv()
    parallel_api_test(env, num_cycles=10)
