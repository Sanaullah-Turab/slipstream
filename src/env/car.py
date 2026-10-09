from __future__ import annotations

import math
from typing import NamedTuple

from .track import SHANGHAI_TRACK_WIDTH

DT = 0.05
CAR_WIDTH = 0.14 * SHANGHAI_TRACK_WIDTH
CAR_LENGTH = 0.39 * SHANGHAI_TRACK_WIDTH
CAR_HALF_WIDTH = CAR_WIDTH / 2.0
CAR_HALF_LEN = CAR_LENGTH / 2.0
WHEELBASE = 0.24 * SHANGHAI_TRACK_WIDTH
GRID_SLOT_SPACING = 0.57 * SHANGHAI_TRACK_WIDTH
GRID_COL_OFFSET = 0.22 * SHANGHAI_TRACK_WIDTH
KERB_WIDTH = 0.07 * SHANGHAI_TRACK_WIDTH

DRAFT_CONE_LENGTH = 3.41 * SHANGHAI_TRACK_WIDTH
DRAFT_CONE_HALF_ANGLE = 0.26
DRAFT_MIN_GAP = 0.68 * SHANGHAI_TRACK_WIDTH
DRAFT_PEAK_GAP = 1.02 * SHANGHAI_TRACK_WIDTH
DRAFT_DRAG_REDUCTION = 0.8
DRAFT_SPEED_BOOST = 35.0


class CarState(NamedTuple):
    x: float
    y: float
    heading: float
    speed: float
    tyre_temp: float = 95.0


class CarParams(NamedTuple):
    max_accel: float = 90.0
    max_speed: float = 150.0
    drag: float = 0.003
    rolling: float = 0.1
    wheelbase: float = WHEELBASE
    max_steer: float = 0.5
    steer_damp: float = 0.7
    corner_drag: float = 0.35


DEFAULT_PARAMS = CarParams()
MAX_SPEED = DEFAULT_PARAMS.max_speed

SHANGHAI_PARAMS = CarParams(
    max_accel=115.0,
    max_speed=210.0,
    drag=0.0016,
    rolling=0.1,
    wheelbase=WHEELBASE,
    max_steer=0.5,
    steer_damp=0.72,
    corner_drag=16.0,
)


def compute_tyre_grip(tyre_temp: float) -> float:
    if tyre_temp < 75.0:
        return 0.85 + 0.05 * max(0.0, (tyre_temp - 50.0) / 25.0)
    elif tyre_temp < 90.0:
        return 0.90 + 0.15 * ((tyre_temp - 75.0) / 15.0)
    elif tyre_temp <= 110.0:
        return 1.05
    elif tyre_temp <= 125.0:
        return 1.05 - 0.20 * ((tyre_temp - 110.0) / 15.0)
    else:
        return max(0.70, 0.85 - 0.15 * min(1.0, (tyre_temp - 125.0) / 25.0))


def step_physics(
    state: CarState,
    throttle: float,
    steer: float,
    params: CarParams = DEFAULT_PARAMS,
    dt: float = DT,
    draft_intensity: float = 0.0,
    surface_grip: float = 1.0,
    rolling_factor: float = 1.0,
) -> tuple[CarState, float]:
    v = state.speed
    lr = params.wheelbase / 2.0
    curr_temp = getattr(state, "tyre_temp", 95.0)
    tyre_grip = compute_tyre_grip(curr_temp)
    total_grip = surface_grip * tyre_grip

    eff_drag = params.drag * (1.0 - DRAFT_DRAG_REDUCTION * draft_intensity)
    eff_max_speed = params.max_speed + DRAFT_SPEED_BOOST * draft_intensity

    long_accel_req = throttle * params.max_accel
    max_long_grip = total_grip * 32.0
    long_ratio = min(0.92, abs(long_accel_req) / max(1e-5, max_long_grip))
    lat_grip_scale = math.sqrt(max(0.05, 1.0 - (long_ratio ** 2)))

    steer_eff = steer * params.max_steer * (1.0 - params.steer_damp * v / params.max_speed) * lat_grip_scale
    beta = math.atan(0.5 * math.tan(steer_eff))
    lat_scrub = getattr(params, "corner_drag", 0.35) * (math.sin(beta) ** 2) * v * (1.0 + 0.02 * v) / max(0.4, total_grip)

    rolling_drag = params.rolling * v * rolling_factor
    accel = long_accel_req * min(1.0, total_grip) - eff_drag * v * v - rolling_drag - lat_scrub
    if v > eff_max_speed:
        v_new = max(0.0, min(v, v + accel * dt))
    else:
        v_new = max(0.0, min(v + accel * dt, eff_max_speed))

    lat_slip_v = math.sin(beta) * v
    q_roll = 3.5 * (v / max(1.0, params.max_speed)) * (0.3 + 0.7 * abs(throttle))
    q_scrub = 0.8 * (lat_slip_v ** 2)
    q_brake = 5.0 * max(0.0, -throttle) * (v / max(1.0, params.max_speed))
    q_cool = (0.012 + 0.00015 * v) * (curr_temp - 30.0)
    new_temp = max(30.0, min(140.0, curr_temp + (q_roll + q_scrub + q_brake - q_cool) * dt))

    heading_rate = (v_new / lr) * math.sin(beta)

    theta = state.heading
    return (
        CarState(
            x=state.x + v_new * math.cos(theta + beta) * dt,
            y=state.y + v_new * math.sin(theta + beta) * dt,
            heading=theta + heading_rate * dt,
            speed=v_new,
            tyre_temp=new_temp,
        ),
        heading_rate,
    )
