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
    from src.env.track import TRACK_WIDTH
    assert CAR_HALF_LEN == 13.0
    assert CAR_HALF_WIDTH == 5.5
    total_len = 2.0 * CAR_HALF_LEN
    total_width = 2.0 * CAR_HALF_WIDTH
    assert total_len == 26.0
    assert total_width == 11.0
    assert TRACK_WIDTH == 70.0


def test_watch_multi_agent_parser():
    from scripts.watch_multi_agent import build_parser
    parser = build_parser()
    args = parser.parse_args([])
    assert "slipstream-p4-1b-7b1372f/final" in args.checkpoint
    assert args.stochastic is False


def test_render_caches_track_surface_and_fonts():
    env = MultiRacingEnv(render_mode="rgb_array")
    assert env._track_surface is None
    assert env._fonts is None
    env.reset(seed=42)
    env.render()
    assert env._track_surface is not None
    assert env._fonts is not None
    cached_surface = env._track_surface
    cached_fonts = env._fonts
    env.render()
    assert env._track_surface is cached_surface
    assert env._fonts is cached_fonts
    env.close()
    assert env._track_surface is None
    assert env._fonts is None

