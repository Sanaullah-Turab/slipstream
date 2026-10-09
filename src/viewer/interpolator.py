from __future__ import annotations

import math
import numpy as np


def shortest_angle_diff(angle_to: float, angle_from: float) -> float:
    return (angle_to - angle_from + math.pi) % (2.0 * math.pi) - math.pi


def interpolate_heading(heading_from: float, heading_to: float, alpha: float) -> float:
    d_theta = shortest_angle_diff(heading_to, heading_from)
    return heading_from + alpha * d_theta


class StateInterpolator:
    def __init__(self, step_subdivisions: int = 3):
        self.step_subdivisions = step_subdivisions
        self.frame_in_step = 0
        self.prev_states: dict[str, dict] = {}
        self.curr_states: dict[str, dict] = {}

    def reset(self, states: dict[str, dict]) -> None:
        self.frame_in_step = 0
        self.prev_states = {
            agent: {
                "pos": s["pos"].copy(),
                "heading": float(s["heading"]),
                "speed": float(s["speed"]),
            }
            for agent, s in states.items()
        }
        self.curr_states = {
            agent: {
                "pos": s["pos"].copy(),
                "heading": float(s["heading"]),
                "speed": float(s["speed"]),
            }
            for agent, s in states.items()
        }

    def on_sim_step(self, states: dict[str, dict]) -> None:
        self.frame_in_step = 0
        self.current_alpha = None
        self.prev_states = {
            agent: {
                "pos": self.curr_states[agent]["pos"].copy(),
                "heading": self.curr_states[agent]["heading"],
                "speed": self.curr_states[agent]["speed"],
            }
            for agent in states
        }
        self.curr_states = {
            agent: {
                "pos": states[agent]["pos"].copy(),
                "heading": float(states[agent]["heading"]),
                "speed": float(states[agent]["speed"]),
            }
            for agent in states
        }

    def advance_frame(self) -> float:
        alpha = min(1.0, max(0.0, self.frame_in_step / float(self.step_subdivisions)))
        self.frame_in_step = min(self.step_subdivisions - 1, self.frame_in_step + 1)
        return alpha

    def get_interpolated_state(self, agent: str, alpha: float | None = None) -> dict:
        if alpha is None:
            alpha = getattr(self, "current_alpha", None)
        if alpha is None:
            if self.step_subdivisions <= 1:
                return self.curr_states[agent].copy()
            alpha = min(1.0, max(0.0, self.frame_in_step / float(self.step_subdivisions)))

        p0 = self.prev_states[agent]["pos"]
        p1 = self.curr_states[agent]["pos"]
        pos = p0 * (1.0 - alpha) + p1 * alpha

        h0 = self.prev_states[agent]["heading"]
        h1 = self.curr_states[agent]["heading"]
        heading = interpolate_heading(h0, h1, alpha)

        s0 = self.prev_states[agent]["speed"]
        s1 = self.curr_states[agent]["speed"]
        speed = s0 * (1.0 - alpha) + s1 * alpha

        return {
            "pos": pos,
            "heading": heading,
            "speed": speed,
        }
