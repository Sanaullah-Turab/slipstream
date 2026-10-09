from __future__ import annotations

import numpy as np
import pytest

from src.env.car import CAR_HALF_WIDTH, KERB_WIDTH, SHANGHAI_PARAMS
from src.env.racing_env import RacingEnv, _OBS_DIM


def test_racing_env_reset_and_obs_shape():
    env = RacingEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (_OBS_DIM,)
    assert obs.dtype == np.float32
    assert -1.0 <= obs[0] <= 1.0
    assert "laps" not in info or isinstance(info, dict)


def test_racing_env_step_full_throttle():
    env = RacingEnv()
    env.reset(seed=42)
    action = np.array([0.0, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (_OBS_DIM,)
    assert not terminated
    assert not truncated
    assert info["speed"] > 0.0
    assert "tyre_temp" in info
    assert "on_kerb" in info
    assert "cumulative_distance" in info


def test_racing_env_kerb_surface_legal():
    env = RacingEnv()
    env.reset(seed=42)
    hw = env.track.half_width
    kw = KERB_WIDTH
    env._lateral = hw + 0.5 * kw
    ts = env.track.get_track_state(env._pos)
    action = np.array([0.5, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert info["on_kerb"] is True


def test_racing_env_off_track_terminated():
    env = RacingEnv()
    env.reset(seed=42)
    hw = env.track.half_width
    kw = KERB_WIDTH
    norm = env.track.normals[0]
    env._pos[:] = env.track.centerline[0] + norm * (hw + kw + 15.0)
    action = np.array([0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    assert reward < 0.0


def test_racing_env_tyre_thermal_dynamics():
    env = RacingEnv()
    env.reset(seed=42)
    initial_temp = env._tyre_temp
    for _ in range(30):
        action = np.array([0.8, 0.3], dtype=np.float32)
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert env._tyre_temp != initial_temp
