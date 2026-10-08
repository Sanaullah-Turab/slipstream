from __future__ import annotations

import math
import numpy as np
import pytest


def test_camera_world_to_screen_round_trip():
    from src.viewer.camera import FollowCamera
    cam = FollowCamera()
    cam.pos = np.array([500.0, -300.0])
    view_size = (1280, 590)
    for zoom in (2.4, 3.2, 4.0):
        for pt in [
            np.array([500.0, -300.0]),
            np.array([0.0, 0.0]),
            np.array([1234.5, -987.6]),
            np.array([-543.2, 876.1]),
        ]:
            screen_pt = cam.world_to_screen(pt, view_size, zoom)
            round_trip = cam.screen_to_world(screen_pt, view_size, zoom)
            assert np.allclose(pt, round_trip, atol=1e-6)


def test_tile_index_covers_every_centerline_sample():
    from src.env.track import Track
    from src.viewer.tile_cache import TileCache, ZOOM_LEVELS
    track = Track()
    cache = TileCache(track)
    n_samples = len(track.centerline)
    all_expected = set(range(n_samples))
    for z in ZOOM_LEVELS:
        covered = set()
        for idx_array in cache.tile_indices[z].values():
            covered.update(idx_array.tolist())
        assert covered == all_expected


def test_camera_hysteresis_does_not_flip_between_thresholds():
    from src.viewer.camera import FollowCamera
    cam = FollowCamera()
    view_size = (1280, 590)
    zoom = 3.2
    half_ext = min((view_size[0] * 0.5) / zoom, (view_size[1] * 0.5) / zoom)
    vel = np.zeros(2)

    pos_ham = np.array([0.0, 0.0])
    pos_ver = np.array([0.50 * half_ext, 0.0])
    cam.update(pos_ham, vel, pos_ver, vel, False, view_size, zoom, 0.05)
    assert cam.framing_target == "midpoint"

    pos_ver = np.array([0.62 * half_ext, 0.0])
    cam.update(pos_ham, vel, pos_ver, vel, False, view_size, zoom, 0.05)
    assert cam.framing_target == "midpoint"

    pos_ver = np.array([0.75 * half_ext, 0.0])
    cam.update(pos_ham, vel, pos_ver, vel, False, view_size, zoom, 0.05)
    assert cam.framing_target == "leader"

    pos_ver = np.array([0.62 * half_ext, 0.0])
    cam.update(pos_ham, vel, pos_ver, vel, False, view_size, zoom, 0.05)
    assert cam.framing_target == "leader"

    pos_ver = np.array([0.50 * half_ext, 0.0])
    cam.update(pos_ham, vel, pos_ver, vel, False, view_size, zoom, 0.05)
    assert cam.framing_target == "midpoint"


def test_interpolation_alpha_and_shortest_angle_wrap():
    from src.viewer.interpolator import StateInterpolator, interpolate_heading
    h1 = math.pi - 0.1
    h2 = -math.pi + 0.1
    h_mid = interpolate_heading(h1, h2, 0.5)
    diff_from_pi = abs(abs(h_mid) - math.pi)
    assert diff_from_pi < 1e-4

    h3 = 0.2
    h4 = 0.8
    assert math.isclose(interpolate_heading(h3, h4, 0.0), 0.2, abs_tol=1e-5)
    assert math.isclose(interpolate_heading(h3, h4, 0.5), 0.5, abs_tol=1e-5)
    assert math.isclose(interpolate_heading(h3, h4, 1.0), 0.8, abs_tol=1e-5)

    interp = StateInterpolator(step_subdivisions=3)
    st = {
        "agent_0": {"pos": np.array([0.0, 0.0]), "heading": 0.0, "speed": 10.0},
        "agent_1": {"pos": np.array([10.0, 20.0]), "heading": math.pi, "speed": 20.0},
    }
    interp.reset(st)
    st_next = {
        "agent_0": {"pos": np.array([30.0, 0.0]), "heading": 0.0, "speed": 10.0},
        "agent_1": {"pos": np.array([40.0, 50.0]), "heading": math.pi, "speed": 20.0},
    }
    interp.on_sim_step(st_next)
    assert np.allclose(interp.get_interpolated_state("agent_0")["pos"], [0.0, 0.0])
    interp.advance_frame()
    assert np.allclose(interp.get_interpolated_state("agent_0")["pos"], [10.0, 0.0])
    interp.advance_frame()
    assert np.allclose(interp.get_interpolated_state("agent_0")["pos"], [20.0, 0.0])
    assert np.allclose(interp.get_interpolated_state("agent_0", alpha=1.0)["pos"], [30.0, 0.0])


def test_layout_rects_fit_inside_window_720p_and_1080p():
    from src.env.track import Track
    from src.viewer.broadcast_hud import BroadcastHUD
    track = Track()
    for w, h in ((1280, 720), (1920, 1080)):
        hud = BroadcastHUD(track, (w, h))
        assert hud.bottom_y >= 0
        assert hud.bottom_y + hud.bottom_h <= h
        assert hud.static_bottom_bar.get_width() == w
        assert hud.static_bottom_bar.get_height() == hud.bottom_h

        col_w = w // 3
        card_m = int(round(8 * hud.font_scale))
        for col_idx in range(3):
            cx = col_idx * col_w + card_m
            cw = col_w - 2 * card_m if col_idx < 2 else (w - 2 * col_w) - 2 * card_m
            cy = hud.bottom_y + card_m
            ch = hud.bottom_h - 2 * card_m
            assert cx >= 0 and cx + cw <= w
            assert cy >= hud.bottom_y and cy + ch <= h

        assert hud.minimap_x >= 0 and hud.minimap_x + hud.minimap_w <= w
        assert hud.minimap_y >= 0 and hud.minimap_y + hud.minimap_h <= hud.view_h

        assert hud.help_x >= 0 and hud.help_x + hud.help_w <= w
        assert hud.help_y >= 0 and hud.help_y + hud.help_h <= hud.view_h
