"""Unit tests for src/env/rewards.py. No RacingEnv is instantiated."""

import numpy as np
import pytest

from src.env.track import Track
from src.env.rewards import (
    AgentState,
    MAX_SPEED,
    WALL_ZONE,
    _heading_reward,
    _lap_bonus,
    _lateral_penalty,
    _off_track_penalty,
    _progress_reward,
    _speed_reward,
    compute_reward,
)
from src.env.car import CAR_HALF_WIDTH


@pytest.fixture(scope="module")
def track() -> Track:
    return Track()


def _state(**overrides) -> AgentState:
    defaults = dict(
        pos=np.array([200.0, 530.0]),
        heading=0.0,
        speed=75.0,
        progress=0.5,
        lateral=0.0,
        track_heading=0.0,
        on_track=True,
        laps=0,
        arc_length=0.0,
    )
    defaults.update(overrides)
    return AgentState(**defaults)


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

class TestProgressReward:
    def test_forward_positive(self, track):
        assert _progress_reward(_state(arc_length=107.5), _state(arc_length=100.0), track) > 0.0

    def test_stationary_zero(self, track):
        assert _progress_reward(_state(arc_length=100.0), _state(arc_length=100.0), track) == 0.0

    def test_reverse_clamped_to_zero(self, track):
        assert _progress_reward(_state(arc_length=92.5), _state(arc_length=100.0), track) == 0.0

    def test_lap_wraparound_positive(self, track):
        prev = _state(arc_length=track.total_length * 0.99)
        curr = _state(arc_length=track.total_length * 0.01)
        assert _progress_reward(curr, prev, track) > 0.0

    def test_proportional_scaling(self, track):
        base = _state(arc_length=0.0)
        r1 = _progress_reward(_state(arc_length=10.0), base, track)
        r2 = _progress_reward(_state(arc_length=20.0), base, track)
        assert r2 == pytest.approx(2 * r1, rel=1e-6)


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

class TestSpeedReward:
    @pytest.mark.parametrize("speed,expected", [
        (0.0,              0.0),
        (MAX_SPEED * 0.25, 0.3 * 0.25),
        (MAX_SPEED * 0.5,  0.3 * 0.5),
        (MAX_SPEED * 0.75, 0.3 * 0.75),
        (MAX_SPEED,        0.3),
    ])
    def test_speed_scaling(self, speed, expected):
        assert _speed_reward(_state(speed=speed)) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Heading
# ---------------------------------------------------------------------------

class TestHeadingReward:
    @pytest.mark.parametrize("angle_deg,expected", [
        (0.0,   0.2),
        (45.0,  0.2 * np.cos(np.deg2rad(45))),
        (90.0,  0.0),
        (135.0, 0.2 * np.cos(np.deg2rad(135))),
        (180.0, -0.2),
    ])
    def test_heading_values(self, angle_deg, expected):
        s = _state(heading=np.deg2rad(angle_deg), track_heading=0.0)
        assert _heading_reward(s) == pytest.approx(expected, abs=1e-6)

    def test_monotone_decrease(self):
        rewards = [
            _heading_reward(_state(heading=np.deg2rad(a), track_heading=0.0))
            for a in [0, 45, 90, 135, 180]
        ]
        assert rewards == sorted(rewards, reverse=True)


# ---------------------------------------------------------------------------
# Off-track penalty
# ---------------------------------------------------------------------------

class TestOffTrackPenalty:
    def test_penalty_value(self):
        assert _off_track_penalty() == -1.0

    def test_compute_reward_off_track_returns_minus_one(self, track):
        curr = _state(arc_length=10.0, on_track=False)
        assert compute_reward(curr, _state(arc_length=0.0), track) == -1.0

    def test_compute_reward_on_track_not_short_circuited(self, track):
        curr = _state(arc_length=10.0, on_track=True)
        assert compute_reward(curr, _state(arc_length=0.0), track) != -1.0


# ---------------------------------------------------------------------------
# Lateral penalty
# ---------------------------------------------------------------------------

class TestLateralPenalty:
    def test_center_zero(self, track):
        assert _lateral_penalty(_state(lateral=0.0), track) == 0.0

    def test_max_magnitude_at_boundary(self, track):
        assert _lateral_penalty(_state(lateral=track.half_width), track) == pytest.approx(-0.15, rel=1e-6)

    def test_symmetric(self, track):
        usable = track.half_width - CAR_HALF_WIDTH
        lat = 0.9 * usable
        assert _lateral_penalty(_state(lateral=-lat), track) == pytest.approx(
            _lateral_penalty(_state(lateral=lat), track)
        )

    def test_defensive_clamp(self, track):
        assert _lateral_penalty(_state(lateral=track.half_width * 2), track) == pytest.approx(-0.15, rel=1e-6)

    def test_always_non_positive(self, track):
        for lat in [0.0, 10.0, -10.0, track.half_width]:
            assert _lateral_penalty(_state(lateral=lat), track) <= 0.0

    def test_lateral_inner_zone_zero(self, track):
        usable = track.half_width - CAR_HALF_WIDTH
        assert _lateral_penalty(_state(lateral=0.5 * usable), track) == 0.0

    def test_lateral_wall_onset_zero(self, track):
        usable = track.half_width - CAR_HALF_WIDTH
        assert _lateral_penalty(_state(lateral=WALL_ZONE * usable), track) == pytest.approx(0.0, abs=1e-9)

    def test_lateral_monotone_near_wall(self, track):
        usable = track.half_width - CAR_HALF_WIDTH
        ratios = [0.82, 0.87, 0.92, 0.97, 1.0]
        penalties = [_lateral_penalty(_state(lateral=r * usable), track) for r in ratios]
        for i in range(len(penalties) - 1):
            assert penalties[i] > penalties[i + 1]


# ---------------------------------------------------------------------------
# Lap bonus
# ---------------------------------------------------------------------------

class TestLapBonus:
    def test_bonus_on_increment(self):
        assert _lap_bonus(_state(laps=1), _state(laps=0)) == 20.0

    def test_no_bonus_same_laps(self):
        assert _lap_bonus(_state(laps=0), _state(laps=0)) == 0.0

    def test_bonus_at_higher_lap_count(self):
        assert _lap_bonus(_state(laps=4), _state(laps=3)) == 20.0


# ---------------------------------------------------------------------------
# compute_reward integration
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_returns_float(self, track):
        assert isinstance(compute_reward(_state(arc_length=10.0), _state(arc_length=0.0), track), float)

    def test_off_track_short_circuit(self, track):
        curr = _state(arc_length=10.0, on_track=False, laps=0)
        assert compute_reward(curr, _state(arc_length=0.0, laps=0), track) == -1.0

    def test_lap_bonus_suppressed_by_offtrack(self, track):
        prev = _state(arc_length=track.total_length * 0.99, laps=0)
        curr = _state(arc_length=track.total_length * 0.01, on_track=False, laps=1)
        assert compute_reward(curr, prev, track) == -1.0

    def test_sum_of_components(self, track):
        prev = _state(arc_length=0.0, laps=0)
        curr = _state(arc_length=10.0, on_track=True, speed=75.0,
                      heading=0.0, track_heading=0.0, lateral=0.0, laps=0)
        expected = (
            _progress_reward(curr, prev, track)
            + _speed_reward(curr)
            + _heading_reward(curr)
            + _lateral_penalty(curr, track)
            + _lap_bonus(curr, prev)
        )
        assert compute_reward(curr, prev, track) == pytest.approx(expected, rel=1e-6)
