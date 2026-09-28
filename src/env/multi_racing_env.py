from __future__ import annotations

from typing import Optional

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .track import Track
from .car import CarState, DEFAULT_PARAMS, DT, MAX_SPEED, CAR_HALF_WIDTH, step_physics
from .racing_env import MAX_STEPS, MAX_HEADING_RATE, MAX_RAY_DIST
from .rewards import compute_reward, AgentState

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

        self.observation_spaces = {a: spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32) for a in AGENTS}
        self.action_spaces = {
            a: spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            for a in AGENTS
        }

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
                "laps": -1 if ts.progress > 0.5 else 0,
                "cumulative_distance": 0.0,
                "respawns": 0,
                "prev_colliding": False,
                "collision_count": 0,
                "start_offset": ts.arc_length,
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
            "start_offset": s["start_offset"],
            "lateral_ratio": abs(s["lateral"]) / self.track.half_width,
        }

    def step(self, actions: dict[str, np.ndarray]):
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

            if ts.on_track:
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

        if self.render_mode == "human":
            self._render_frame()

        obs_agents = AGENTS if truncated else self.agents
        obs = {a: self._build_obs(a) for a in obs_agents}

        return obs, rewards, terminations, truncations, infos

    def _respawn(self, agent: str) -> None:
        s = self._state[agent]
        opp_pos = self._state[AGENTS[1] if agent == AGENTS[0] else AGENTS[0]]["pos"]

        idx = self.track.nearest_idx(s["pos"])
        spawn_pos = self.track.centerline[idx].copy()

        if np.linalg.norm(spawn_pos - opp_pos) < 2.0 * CAR_HALF_WIDTH:
            spawn_pos = spawn_pos + self.track.normals[idx] * 2.0 * CAR_HALF_WIDTH

        heading = float(np.arctan2(self.track.tangents[idx, 1], self.track.tangents[idx, 0]))
        ts = self.track.get_track_state(spawn_pos)

        s["pos"][:] = spawn_pos
        s["heading"] = heading
        s["speed"] = 0.0
        s["heading_rate"] = 0.0
        s["progress"] = ts.progress
        s["lateral"] = ts.lateral
        s["track_heading"] = ts.track_heading
        s["arc_length"] = ts.arc_length

    def render(self):
        frame = self._render_frame()
        if self.render_mode == "rgb_array":
            return frame

    def _render_frame(self):
        import pygame
        from .track import RAY_ANGLES

        W, H = 800, 650

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Slipstream - Multi Agent")
            else:
                self._screen = pygame.Surface((W, H))
            self._clock = pygame.time.Clock()

        surf = self._screen
        surf.fill((28, 38, 22))

        outer = [(int(x), int(y)) for x, y in self.track.outer]
        inner = [(int(x), int(y)) for x, y in self.track.inner]
        pygame.draw.polygon(surf, (52, 52, 52), outer)
        pygame.draw.polygon(surf, (28, 38, 22), inner)
        pygame.draw.lines(surf, (210, 210, 210), True, outer, 2)
        pygame.draw.lines(surf, (210, 210, 210), True, inner, 2)

        cl = self.track.centerline
        for i in range(0, len(cl) - 8, 16):
            pygame.draw.line(
                surf, (85, 85, 85),
                (int(cl[i, 0]), int(cl[i, 1])),
                (int(cl[i + 8, 0]), int(cl[i + 8, 1])),
                1,
            )

        colors = {"agent_0": (0, 200, 255), "agent_1": (255, 100, 50)}
        ray_colors = {"agent_0": (255, 140, 0), "agent_1": (180, 255, 80)}

        for agent in AGENTS:
            s = self._state[agent]
            pos, heading = s["pos"], s["heading"]
            color = colors[agent]

            rays = self.track.ray_distances(pos, heading, MAX_RAY_DIST)
            for a, dist in zip(RAY_ANGLES, rays):
                angle = heading + a
                end = pos + dist * np.array([np.cos(angle), np.sin(angle)])
                pygame.draw.line(
                    surf, ray_colors[agent],
                    (int(pos[0]), int(pos[1])),
                    (int(end[0]), int(end[1])),
                    1,
                )

            sz = 9
            fwd = np.array([np.cos(heading), np.sin(heading)])
            left = np.array([-fwd[1], fwd[0]])
            tip = pos + fwd * sz
            bl = pos - fwd * sz * 0.6 + left * sz * 0.55
            br = pos - fwd * sz * 0.6 - left * sz * 0.55
            pygame.draw.polygon(surf, color, [
                (int(tip[0]), int(tip[1])),
                (int(bl[0]), int(bl[1])),
                (int(br[0]), int(br[1])),
            ])

        font = pygame.font.SysFont("monospace", 13)
        hud_x = {AGENTS[0]: 8, AGENTS[1]: W // 2 + 4}
        for agent in AGENTS:
            s = self._state[agent]
            x = hud_x[agent]
            label_color = colors[agent]
            surf.blit(font.render(agent, True, label_color), (x, 570))
            for i, text in enumerate([
                f"Speed:      {s['speed']:6.1f}",
                f"Laps:       {s['laps']}",
                f"Progress:   {s['progress']:.3f}",
                f"Respawns:   {s['respawns']}",
                f"Collisions: {s['collision_count']}",
            ]):
                surf.blit(font.render(text, True, (200, 200, 200)), (x, 585 + i * 14))

        surf.blit(font.render(f"Step: {self._step_count}", True, (160, 160, 160)), (8, 555))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            assert self._clock is not None
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    def close(self) -> None:
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
