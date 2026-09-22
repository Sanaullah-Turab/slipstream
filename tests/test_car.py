"""Unit tests for src/env/car.py physics model."""

import pytest

from src.env.car import DEFAULT_PARAMS, CarState, step_physics


@pytest.fixture
def still() -> CarState:
    return CarState(x=0.0, y=0.0, heading=0.0, speed=0.0)


@pytest.fixture
def rolling() -> CarState:
    return CarState(x=0.0, y=0.0, heading=0.0, speed=50.0)


def test_zero_throttle_decelerates(rolling):
    new, _ = step_physics(rolling, throttle=0.0, steer=0.0)
    assert new.speed < rolling.speed


def test_full_throttle_accelerates(still):
    new, _ = step_physics(still, throttle=1.0, steer=0.0)
    assert new.speed > 0.0


def test_zero_steer_drives_straight(rolling):
    new, _ = step_physics(rolling, throttle=0.5, steer=0.0)
    assert new.heading == pytest.approx(rolling.heading, abs=1e-9)
    assert new.y == pytest.approx(0.0, abs=1e-6)


def test_max_speed_not_exceeded():
    state = CarState(x=0.0, y=0.0, heading=0.0, speed=DEFAULT_PARAMS.max_speed)
    new, _ = step_physics(state, throttle=1.0, steer=0.0)
    assert new.speed <= DEFAULT_PARAMS.max_speed


def test_steer_damp_reduces_normalized_rate():
    slow = CarState(x=0.0, y=0.0, heading=0.0, speed=30.0)
    fast = CarState(x=0.0, y=0.0, heading=0.0, speed=140.0)
    _, hr_slow = step_physics(slow, throttle=0.0, steer=1.0)
    _, hr_fast = step_physics(fast, throttle=0.0, steer=1.0)
    assert (hr_slow / slow.speed) > (hr_fast / fast.speed)
