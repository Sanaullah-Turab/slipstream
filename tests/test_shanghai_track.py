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
