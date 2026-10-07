from __future__ import annotations

import math
from typing import NamedTuple

DT = 0.05
CAR_HALF_WIDTH = 5.5


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
    wheelbase: float = 20.0
    max_steer: float = 0.5
    steer_damp: float = 0.7


DEFAULT_PARAMS = CarParams()
MAX_SPEED = DEFAULT_PARAMS.max_speed

DRAFT_CONE_LENGTH = 150.0
DRAFT_CONE_HALF_ANGLE = 0.26
DRAFT_MIN_GAP = 30.0
DRAFT_PEAK_GAP = 45.0
DRAFT_DRAG_REDUCTION = 0.4
DRAFT_SPEED_BOOST = 15.0


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

