from __future__ import annotations

from typing import Optional

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .track import Track
from .car import CarState, DEFAULT_PARAMS, DT, MAX_SPEED, step_physics
from .racing_env import MAX_STEPS, MAX_HEADING_RATE, MAX_RAY_DIST

AGENTS = ["agent_0", "agent_1"]
SPAWN_OFFSET_IDX = 10
MAX_OPP_DIST = 300.0

class MultiRacingEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "multi_racing_v1"}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.track = Track()
        self.render_mode = render_mode
        self.possible_agents = AGENTS[:]

        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_spaces = {a: obs_space for a in AGENTS}
        self.action_spaces = {a: act_space for a in AGENTS}

        self._state: dict = {}
        self._step_count = 0
        self._screen = None
        self._clock = None

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        self.agents = AGENTS[:]
        self._step_count = 0

        slots = [0, SPAWN_OFFSET_IDX]
        if self.np_random.integers(0, 2):
            slots = slots[::-1]

        self._state = {}
        for agent, slot in zip(AGENTS, slots):
            lat_offset = float(self.np_random.uniform(-5.0, 5.0))
            pos = self.track.centerline[slot] + self.track.normals[slot] * lat_offset
            heading = float(
                np.arctan2(self.track.tangents[slot, 1], self.track.tangents[slot, 0])
            )
            ts = self.track.get_track_state(pos)
            self._state[agent] = {
                "pos": pos.copy(),
                "heading": heading,
                "speed": 0.0,
                "heading_rate": 0.0,
                "progress": ts.progress,
                "lateral": ts.lateral,
                "track_heading": ts.track_heading,
                "arc_length": ts.arc_length,
                "laps": 0,
                "cumulative_distance": 0.0,
                "respawns": 0,
                "prev_colliding": False,
                "collision_count": 0,
            }

        obs = {a: self._build_obs(a) for a in self.agents}
        info = {a: self._build_info(a) for a in self.agents}
        return obs, info

    def _build_obs(self, agent: str) -> np.ndarray:
        s = self._state[agent]
        opp = self._state[AGENTS[1] if agent == AGENTS[0] else AGENTS[0]]

        heading_err = s["heading"] - s["track_heading"]
        lateral_norm = float(np.clip(s["lateral"] / self.track.half_width, -1.0, 1.0))
        rays = self.track.ray_distances(s["pos"], s["heading"], MAX_RAY_DIST) / MAX_RAY_DIST

        opp_dist = float(np.clip(np.linalg.norm(opp["pos"] - s["pos"]) / MAX_OPP_DIST, 0.0, 1.0))

        arc_gap = opp["arc_length"] - s["arc_length"]
        half_len = self.track.total_length / 2.0
        if arc_gap > half_len:
            arc_gap -= self.track.total_length
        elif arc_gap < -half_len:
            arc_gap += self.track.total_length
        opp_progress_gap = float(np.clip(arc_gap / MAX_OPP_DIST, -1.0, 1.0))

        opp_rel_speed = float(np.clip((opp["speed"] - s["speed"]) / MAX_SPEED, -1.0, 1.0))
        opp_lateral_gap = float(
            np.clip((opp["lateral"] - s["lateral"]) / (2.0 * self.track.half_width), -1.0, 1.0)
        )

        return np.array(
            [
                s["speed"] / MAX_SPEED,
                np.sin(heading_err),
                np.cos(heading_err),
                lateral_norm,
                s["progress"],
                float(np.clip(s["heading_rate"] / MAX_HEADING_RATE, -1.0, 1.0)),
                *np.clip(rays, 0.0, 1.0),
                opp_dist,
                opp_progress_gap,
                opp_rel_speed,
                opp_lateral_gap,
            ],
            dtype=np.float32,
        )

    def _build_info(self, agent: str) -> dict:
        s = self._state[agent]
        return {
            "laps": s["laps"],
            "progress": s["progress"],
            "speed": s["speed"],
            "respawns": s["respawns"],
            "collision_count": s["collision_count"],
            "cumulative_distance": s["cumulative_distance"],
        }

    def step(self, actions: dict[str, np.ndarray]):
        from .rewards import compute_reward, AgentState

        rewards = {}
        terminations = {a: False for a in self.agents}
        truncations = {}
        infos = {}

        for agent in self.agents:
            s = self._state[agent]
            action = actions[agent]
            steer = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip(action[1], -1.0, 1.0))

            prev_state = AgentState(
                pos=s["pos"].copy(),
                heading=s["heading"],
                speed=s["speed"],
                progress=s["progress"],
                lateral=s["lateral"],
                track_heading=s["track_heading"],
                on_track=True,
                laps=s["laps"],
                arc_length=s["arc_length"],
            )

            car = CarState(x=s["pos"][0], y=s["pos"][1], heading=s["heading"], speed=s["speed"])
            car, heading_rate = step_physics(car, throttle, steer, DEFAULT_PARAMS, DT)
            s["pos"][:] = car.x, car.y
            s["heading"] = car.heading
            s["speed"] = car.speed
            s["heading_rate"] = heading_rate

            ts = self.track.get_track_state(s["pos"])
            s["progress"] = ts.progress
            s["lateral"] = ts.lateral
            s["track_heading"] = ts.track_heading

            if s["progress"] - prev_state.progress < -0.5:
                s["laps"] += 1

            s["arc_length"] = ts.arc_length

            arc_delta = s["arc_length"] - prev_state.arc_length
            if arc_delta < -self.track.total_length / 2.0:
                arc_delta += self.track.total_length
            s["cumulative_distance"] += max(0.0, arc_delta)

            curr_state = AgentState(
                pos=s["pos"].copy(),
                heading=s["heading"],
                speed=s["speed"],
                progress=s["progress"],
                lateral=s["lateral"],
                track_heading=s["track_heading"],
                on_track=ts.on_track,
                laps=s["laps"],
                arc_length=s["arc_length"],
            )

            reward = compute_reward(curr_state, prev_state, self.track)

            if not ts.on_track:
                reward = -5.0
                s["respawns"] += 1
                self._respawn(agent)

            rewards[agent] = reward

        self._step_count += 1

        # Collision detection (rising edges only)
        s0, s1 = self._state[AGENTS[0]], self._state[AGENTS[1]]
        from .car import CAR_HALF_WIDTH
        colliding = float(np.linalg.norm(s0["pos"] - s1["pos"])) < 2.0 * CAR_HALF_WIDTH
        for s in (s0, s1):
            if colliding and not s["prev_colliding"]:
                s["collision_count"] += 1
            s["prev_colliding"] = colliding

        truncated = self._step_count >= MAX_STEPS

        for agent in self.agents:
            truncations[agent] = truncated
            infos[agent] = self._build_info(agent)

        if truncated:
            self.agents = []

        obs_agents = AGENTS if truncated else self.agents
        obs = {a: self._build_obs(a) for a in obs_agents}

        return obs, rewards, terminations, truncations, infos

    def _respawn(self, agent: str) -> None:
        s = self._state[agent]
        opp_pos = self._state[AGENTS[1] if agent == AGENTS[0] else AGENTS[0]]["pos"]

        idx = self.track.nearest_idx(s["pos"])
        spawn_pos = self.track.centerline[idx].copy()

        from .car import CAR_HALF_WIDTH
        if np.linalg.norm(spawn_pos - opp_pos) < 2.0 * CAR_HALF_WIDTH:
            spawn_pos = spawn_pos + self.track.normals[idx] * 2.0 * CAR_HALF_WIDTH

        heading = float(np.arctan2(self.track.tangents[idx, 1], self.track.tangents[idx, 0]))
        ts = self.track.get_track_state(spawn_pos)

        s["pos"] = spawn_pos
        s["heading"] = heading
        s["speed"] = 0.0
        s["heading_rate"] = 0.0
        s["progress"] = ts.progress
        s["lateral"] = ts.lateral
        s["track_heading"] = ts.track_heading
        s["arc_length"] = ts.arc_length

    def render(self):
        return None

    def close(self) -> None:
        pass
