import numpy as np
import pytest

from src.env.multi_racing_env import MultiRacingEnv, AGENTS, CAR_HALF_LEN
from src.env.car import CAR_HALF_WIDTH


def test_rgb_array_render_dimensions_and_channels():
    env = MultiRacingEnv(render_mode="rgb_array")
    env.reset(seed=123)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert frame.shape[0] >= 600
    assert frame.shape[1] >= 1000
    env.close()


def test_render_executes_multiple_steps_without_error():
    env = MultiRacingEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    for _ in range(5):
        actions = {agent: np.array([0.5, 0.0], dtype=np.float32) for agent in AGENTS}
        obs, rewards, terms, truncs, infos = env.step(actions)
        frame = env.render()
        assert frame is not None
        assert frame.shape[2] == 3
    env.close()


def test_render_chassis_matches_physical_obb_constants():
    assert CAR_HALF_LEN == 20.0
    assert CAR_HALF_WIDTH == 9.0
    total_len = 2.0 * CAR_HALF_LEN
    total_width = 2.0 * CAR_HALF_WIDTH
    assert total_len == 40.0
    assert total_width == 18.0


def test_watch_multi_agent_parser():
    from scripts.watch_multi_agent import build_parser
    parser = build_parser()
    args = parser.parse_args([])
    assert "slipstream-p4-1b-7b1372f/final" in args.checkpoint
    assert args.stochastic is False

