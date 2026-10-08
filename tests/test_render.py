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
    assert frame.shape[0] == 900
    assert frame.shape[1] == 1600
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
    from src.env.track import TRACK_WIDTH, SHANGHAI_TRACK_WIDTH
    assert CAR_HALF_LEN == pytest.approx(0.39 * SHANGHAI_TRACK_WIDTH / 2.0)
    assert CAR_HALF_WIDTH == pytest.approx(0.14 * SHANGHAI_TRACK_WIDTH / 2.0)
    total_len = 2.0 * CAR_HALF_LEN
    total_width = 2.0 * CAR_HALF_WIDTH
    assert total_len == pytest.approx(0.39 * SHANGHAI_TRACK_WIDTH)
    assert total_width == pytest.approx(0.14 * SHANGHAI_TRACK_WIDTH)
    assert TRACK_WIDTH == 70.0


def test_watch_multi_agent_parser():
    from scripts.watch_multi_agent import build_parser
    parser = build_parser()
    args = parser.parse_args([])
    assert "slipstream-p4-1b-7b1372f/final" in args.checkpoint
    assert args.stochastic is False
    assert args.size is None
    args_1080 = parser.parse_args(["--size", "1920x1080"])
    assert args_1080.size == "1920x1080"


def test_render_layout_fit_and_custom_window_size():
    env = MultiRacingEnv(render_mode="rgb_array", window_size=(1920, 1080))
    env.reset(seed=123)
    frame = env.render()
    assert frame.shape[0] == 1080
    assert frame.shape[1] == 1920
    assert env._scale is not None
    assert env.track_view_w == 1600
    assert env.hud_w == 320
    env.close()


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

