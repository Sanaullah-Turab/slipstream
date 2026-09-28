from __future__ import annotations

from typing import Optional

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .track import Track

AGENTS = ["agent_0", "agent_1"]
SPAWN_OFFSET_IDX = 10

class MultiRacingEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "multi_racing_v1"}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.track = Track()
        self.render_mode = render_mode
        self.possible_agents = AGENTS[:]

        # Placeholder 15-dim observation for now
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
            }

        obs = {a: np.zeros(15, dtype=np.float32) for a in self.agents}
        info = {a: {} for a in self.agents}
        return obs, info

    def step(self, actions: dict[str, np.ndarray]):
        # Stub step function for Commit 2
        self._step_count += 1
        
        obs = {a: np.zeros(15, dtype=np.float32) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self) -> None:
        pass
