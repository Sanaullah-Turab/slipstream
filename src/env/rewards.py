from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import Track
from .car import MAX_SPEED, CAR_HALF_WIDTH

WALL_ZONE = 0.8


@dataclass
class AgentState:
    pos: np.ndarray
    heading: float
    speed: float
    progress: float
    lateral: float
    track_heading: float
    on_track: bool
    laps: int
    arc_length: float


def _progress_reward(curr: AgentState, prev: AgentState, track: Track) -> float:
    delta = curr.arc_length - prev.arc_length
    if delta < -track.total_length / 2:
        delta += track.total_length
    elif delta > track.total_length / 2:
        delta -= track.total_length
    return max(0.0, delta / track.total_length)


def _speed_reward(curr: AgentState) -> float:
    return 0.3 * min(curr.speed, MAX_SPEED) / MAX_SPEED


def _heading_reward(curr: AgentState) -> float:
    return 0.2 * float(np.cos(curr.heading - curr.track_heading))


def _off_track_penalty() -> float:
    return -1.0


def _lateral_penalty(curr: AgentState, track: Track) -> float:
    usable = track.half_width - CAR_HALF_WIDTH
    ratio = min(abs(curr.lateral) / usable, 1.0)
    wall_ratio = max(0.0, (ratio - WALL_ZONE) / (1.0 - WALL_ZONE))
    return -0.15 * wall_ratio ** 2


def _lap_bonus(curr: AgentState, prev: AgentState) -> float:
    return 20.0 if curr.laps > prev.laps else 0.0


def compute_reward(curr: AgentState, prev: AgentState, track: Track) -> float:
    if not curr.on_track:
        return _off_track_penalty()
    return (
        _progress_reward(curr, prev, track)
        + _speed_reward(curr)
        + _heading_reward(curr)
        + _lateral_penalty(curr, track)
        + _lap_bonus(curr, prev)
    )


DEFAULT_CONTACT_PENALTY = -0.1
DEFAULT_CONTACT_STEP_PENALTY = -0.02
DEFAULT_POSITION_K = 0.03
DEFAULT_POSITION_G0 = 50.0


def contact_penalty(penalty: float = DEFAULT_CONTACT_PENALTY) -> float:
    return penalty


def contact_step_penalty(penalty: float = DEFAULT_CONTACT_STEP_PENALTY) -> float:
    return penalty


def compute_positional_reward(
    dist_ego: float,
    dist_opp: float,
    k: float = DEFAULT_POSITION_K,
    g0: float = DEFAULT_POSITION_G0,
) -> float:
    diff = dist_ego - dist_opp
    return float(k * np.clip(diff / g0, -1.0, 1.0))



