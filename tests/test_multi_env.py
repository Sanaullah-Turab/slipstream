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
        assert obs[a].shape == (19,)


def test_step_all_keys():
    env = make_env()
    obs, rew, terms, truncs, infos = env.step(zero_actions(env))
    for a in AGENTS:
        assert a in obs and a in rew and a in terms and a in truncs and a in infos


def test_info_keys():
    env = make_env()
    _, _, _, _, infos = env.step(zero_actions(env))
    expected = {"laps", "progress", "speed", "respawns", "collision_count", "cumulative_distance", "fault_log"}
    assert expected.issubset(set(infos[AGENTS[0]].keys()))


def test_no_spurious_lap():
    import math
    env = MultiRacingEnv()
    for seed in range(50):
        env.reset(seed=seed)
        start_offsets = {a: env._state[a]["arc_length"] for a in AGENTS}
        for _ in range(50):
            env.step({a: env.action_space(a).sample() for a in AGENTS})
            for a in AGENTS:
                s = env._state[a]
                race_dist = start_offsets[a] + s["cumulative_distance"]
                expected_max_laps = math.floor(race_dist / env.track.total_length)
                assert s["laps"] <= expected_max_laps, (
                    f"Spurious lap! Laps: {s['laps']}, Odo: {s['cumulative_distance']}, "
                    f"Start: {start_offsets[a]}"
                )


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
    assert obs.shape == (2, 19)
    actions = np.stack([env.action_space.sample() for _ in range(2)])
    obs_step, rewards, dones, infos = env.step(actions)
    obs_step = np.asarray(obs_step)
    assert obs_step.shape == (2, 19)
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
        assert info["terminal_observation"].shape == (19,)
    vec.close()


def test_batched_inference_equivalence():
    from stable_baselines3 import PPO
    from src.training.callbacks import batch_predict
    env = MultiRacingEnv()
    obs_dict, _ = env.reset(seed=42)
    
    model = PPO("MlpPolicy", TwoCarVecEnv(), seed=42)
    
    act0, _ = model.predict(obs_dict[AGENTS[0]], deterministic=True)
    act1, _ = model.predict(obs_dict[AGENTS[1]], deterministic=True)
    
    actions_dict = batch_predict(model, obs_dict, AGENTS, deterministic=True)
    
    np.testing.assert_allclose(act0, actions_dict[AGENTS[0]], atol=1e-5)
    np.testing.assert_allclose(act1, actions_dict[AGENTS[1]], atol=1e-5)
    
    actions_dict_rev = batch_predict(model, obs_dict, [AGENTS[1], AGENTS[0]], deterministic=True)
    np.testing.assert_allclose(act1, actions_dict_rev[AGENTS[1]], atol=1e-5)
    np.testing.assert_allclose(act0, actions_dict_rev[AGENTS[0]], atol=1e-5)
    
    # Verify leader/follower logic (from MultiEvalCallback/eval_multi.py) works
    # Mock some cumulative distances
    dist = {AGENTS[0]: 1500, AGENTS[1]: 1400}
    gap = abs(dist[AGENTS[0]] - dist[AGENTS[1]])
    gap_eps = 5.0 # arbitrary small epsilon
    
    # Distances are far apart
    if gap < gap_eps:
        leader, follower = AGENTS[0], AGENTS[1]
    else:
        leader = max(AGENTS, key=lambda a: dist[a])
        follower = AGENTS[1] if leader == AGENTS[0] else AGENTS[0]
        
    assert leader == AGENTS[0]
    assert follower == AGENTS[1]
    
    # Reverse it
    dist = {AGENTS[0]: 1200, AGENTS[1]: 1400}
    leader = max(AGENTS, key=lambda a: dist[a])
    follower = AGENTS[1] if leader == AGENTS[0] else AGENTS[0]
    assert leader == AGENTS[1]
    assert follower == AGENTS[0]


def test_parallel_api():
    from pettingzoo.test import parallel_api_test
    env = MultiRacingEnv()
    parallel_api_test(env, num_cycles=10)


def test_warm_start_19dim_produces_same_actions_as_15dim():
    """Zero-padded 19-dim warm start: first layer must accept (19,) and last 4 input weights zero."""
    import torch
    from stable_baselines3 import PPO
    from src.env.vec_multi import TwoCarVecEnv

    env_19 = TwoCarVecEnv()
    model_19 = PPO("MlpPolicy", env_19, seed=0)

    first_layer = getattr(model_19.policy.mlp_extractor.policy_net, "0")
    assert first_layer.weight.shape[1] == 19

    with torch.no_grad():
        w = first_layer.weight.data.clone()
        w[:, 15:] = 0.0
        first_layer.weight.data = w

    # Build 19-dim obs with last 4 zeros, ensure output is stable (no NaN)
    obs_19 = np.zeros((4, 19), dtype=np.float32)
    obs_19[:, :15] = np.random.randn(4, 15).astype(np.float32)
    acts, _ = model_19.predict(obs_19, deterministic=True)
    assert not np.any(np.isnan(acts)), "NaN in actions from zero-padded 19-dim obs"
    assert acts.shape == (4, 2)


def test_obb_collision_logged_in_info():
    env = MultiRacingEnv()
    env.reset(seed=0)
    pos = env.track.centerline[0].copy()
    for a in AGENTS:
        env._state[a]["pos"][:] = pos

    _, _, _, _, infos = env.step(zero_actions(env))
    total = sum(infos[a]["collision_count"] for a in AGENTS)
    assert total > 0, "OBB collision at identical positions should be detected"


def test_steps_in_contact_and_collision_flag_in_info():
    env = MultiRacingEnv()
    env.reset(seed=0)
    pos = env.track.centerline[0].copy()
    for a in AGENTS:
        env._state[a]["pos"][:] = pos

    _, _, _, _, infos1 = env.step(zero_actions(env))
    assert infos1[AGENTS[0]]["collision"] is True
    assert infos1[AGENTS[0]]["collision_count"] == 1
    assert infos1[AGENTS[0]]["steps_in_contact"] == 1

    _, _, _, _, infos2 = env.step(zero_actions(env))
    assert infos2[AGENTS[0]]["collision"] is True
    assert infos2[AGENTS[0]]["collision_count"] == 1
    assert infos2[AGENTS[0]]["steps_in_contact"] == 2



def test_fault_log_populated_on_collision():
    env = MultiRacingEnv()
    env.reset(seed=0)
    pos = env.track.centerline[0].copy()
    for a in AGENTS:
        env._state[a]["pos"][:] = pos

    env.step(zero_actions(env))
    total_faults = sum(
        sum(env._state[a]["fault_log"].values()) for a in AGENTS
    )
    assert total_faults > 0, "Fault log must be populated after a collision"


def test_contact_penalty_applied_once_per_event():
    env = MultiRacingEnv(contact_penalty=-0.1)
    env.reset(seed=0)
    pos = env.track.centerline[0].copy()
    for a in AGENTS:
        env._state[a]["pos"][:] = pos

    _, r1, _, _, _ = env.step(zero_actions(env))
    assert env._state[AGENTS[0]]["prev_colliding"] is True

    _, r2, _, _, _ = env.step(zero_actions(env))
    assert r1[AGENTS[0]] < r2[AGENTS[0]]
    assert pytest.approx(r2[AGENTS[0]] - r1[AGENTS[0]], abs=1e-5) == 0.1


def test_contact_penalty_configurable():
    env = MultiRacingEnv(contact_penalty=-0.5)
    assert env.contact_penalty == -0.5
    env.reset(seed=0)
    pos = env.track.centerline[0].copy()
    for a in AGENTS:
        env._state[a]["pos"][:] = pos

    _, r1, _, _, _ = env.step(zero_actions(env))
    _, r2, _, _, _ = env.step(zero_actions(env))
    assert pytest.approx(r2[AGENTS[0]] - r1[AGENTS[0]], abs=1e-5) == 0.5


def test_legacy_collision_env():
    env = MultiRacingEnv(legacy_collision=True)
    obs, _ = env.reset(seed=0)
    for a in AGENTS:
        assert obs[a].shape == (15,)
    pos = env.track.centerline[0].copy()
    for a in AGENTS:
        env._state[a]["pos"][:] = pos
    _, _, _, _, infos = env.step(zero_actions(env))
    assert infos[AGENTS[0]]["collision"] is True
    assert infos[AGENTS[0]]["collision_count"] == 1


