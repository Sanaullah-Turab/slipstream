from __future__ import annotations

import math
import numpy as np
import pytest

from src.env.track import Track, SHANGHAI_CIRCUIT_LENGTH_MIN


def test_shanghai_track_total_length():
    track = Track(circuit="shanghai")
    assert track.total_length >= SHANGHAI_CIRCUIT_LENGTH_MIN
    assert len(track.centerline) >= 1500


def test_shanghai_track_start_finish_line_and_direction():
    track = Track(circuit="shanghai")
    t0 = track.tangents[0]
    assert t0[0] > 0.0
    assert t0[1] < 0.0


def test_shanghai_track_starting_grid_positions():
    track = Track(circuit="shanghai")
    grid = track.get_starting_grid()
    assert "agent_0" in grid and "agent_1" in grid
    pos0, heading0 = grid["agent_0"]
    pos1, heading1 = grid["agent_1"]
    ts0 = track.get_track_state(pos0)
    ts1 = track.get_track_state(pos1)
    assert ts0.on_track and ts1.on_track
    assert ts0.arc_length > ts1.arc_length
    assert np.linalg.norm(pos0 - pos1) > 15.0


def test_shanghai_track_ray_casting():
    track = Track(circuit="shanghai")
    rays = track.ray_distances(track.centerline[0], float(np.arctan2(track.tangents[0, 1], track.tangents[0, 0])))
    assert len(rays) == 5
    assert all(r > 0.0 for r in rays)
    assert all(np.isfinite(r) for r in rays)


def test_shanghai_track_clearance_and_minimum_turn_radius():
    track = Track(circuit="shanghai")
    w = track.half_width * 2.0
    min_dist_sq = 1e12
    chunk_size = 500
    n_pts = len(track.centerline)
    cl = track.centerline
    arc = track.arc_lengths
    l_circ = track.total_length
    for i in range(0, n_pts, chunk_size):
        chunk_pts = cl[i : i + chunk_size]
        chunk_arc = arc[i : i + chunk_size]
        d_arc = np.abs(chunk_arc[:, None] - arc[None, :])
        d_arc = np.minimum(d_arc, l_circ - d_arc)
        mask = d_arc > 2.0 * w
        diff = chunk_pts[:, None, :] - cl[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        dist_sq[~mask] = 1e12
        cur_min = float(np.min(dist_sq))
        if cur_min < min_dist_sq:
            min_dist_sq = cur_min
    min_dist = float(np.sqrt(min_dist_sq))
    r_min = float(1.0 / (np.max(np.abs(track.curvatures)) + 1e-12))
    assert min_dist >= 1.4 * w
    assert r_min >= 0.8 * w


def test_shanghai_track_straight_segments_zero_curvature():
    track = Track(circuit="shanghai")
    zero_curv_count = int(np.sum(np.abs(track.curvatures) < 1e-9))
    assert zero_curv_count >= 1000


def test_shanghai_track_loop_closure_smoothness():
    track = Track(circuit="shanghai")
    p_start = track.centerline[0]
    p_end = track.centerline[-1]
    assert np.linalg.norm(p_start - p_end) <= 2.05
    t_start = track.tangents[0]
    t_end = track.tangents[-1]
    assert float(np.dot(t_start, t_end)) > 0.999
    assert abs(track.curvatures[0]) < 1e-6
    assert abs(track.curvatures[-1]) < 1e-6
