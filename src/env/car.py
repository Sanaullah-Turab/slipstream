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


class CarParams(NamedTuple):
    max_accel: float = 80.0
    max_speed: float = 150.0
    drag: float = 0.003
    rolling: float = 0.1
    wheelbase: float = WHEELBASE
    max_steer: float = 0.5
    steer_damp: float = 0.7


DEFAULT_PARAMS = CarParams()
MAX_SPEED = DEFAULT_PARAMS.max_speed


def step_physics(
    state: CarState,
    throttle: float,
    steer: float,
    params: CarParams = DEFAULT_PARAMS,
    dt: float = DT,
    draft_intensity: float = 0.0,
) -> tuple[CarState, float]:
    v = state.speed
    lr = params.wheelbase / 2.0

    eff_drag = params.drag * (1.0 - DRAFT_DRAG_REDUCTION * draft_intensity)
    eff_max_speed = params.max_speed + DRAFT_SPEED_BOOST * draft_intensity

    accel = throttle * params.max_accel - eff_drag * v * v - params.rolling * v
    if v > eff_max_speed:
        v_new = max(0.0, min(v, v + accel * dt))
    else:
        v_new = max(0.0, min(v + accel * dt, eff_max_speed))

    steer_eff = steer * params.max_steer * (1.0 - params.steer_damp * v / params.max_speed)
    beta = math.atan(0.5 * math.tan(steer_eff))
    heading_rate = (v_new / lr) * math.sin(beta)

    theta = state.heading
    return (
        CarState(
            x=state.x + v_new * math.cos(theta + beta) * dt,
            y=state.y + v_new * math.sin(theta + beta) * dt,
            heading=theta + heading_rate * dt,
            speed=v_new,
        ),
        heading_rate,
    )
