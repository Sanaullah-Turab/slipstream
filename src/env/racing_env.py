from __future__ import annotations

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .track import Track, RAY_ANGLES
from .rewards import compute_reward, AgentState
from .car import CarState, DEFAULT_PARAMS, SHANGHAI_PARAMS, DT, MAX_SPEED, CAR_HALF_WIDTH, KERB_WIDTH, step_physics

MAX_STEPS = 2000
MAX_HEADING_RATE = 2.0

MAX_RAY_DIST = 250.0
_OBS_DIM = 11


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.track = Track()
        self.render_mode = render_mode
        self.car_params = SHANGHAI_PARAMS

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._pos = np.zeros(2)
        self._heading = 0.0
        self._speed = 0.0
        self._step_count = 0
        self._laps = 0
        self._progress = 0.0
        self._lateral = 0.0
        self._track_heading = 0.0
        self._arc_length = 0.0
        self._heading_rate = 0.0
        self._tyre_temp = 95.0
        self._prev_steer = 0.0

        self._screen = None
        self._clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = self.track.centerline[0].copy()
        self._heading = float(
            np.arctan2(self.track.tangents[0, 1], self.track.tangents[0, 0])
        )
        self._speed = 0.0
        self._step_count = 0
        self._laps = 0
        self._heading_rate = 0.0
        self._tyre_temp = 95.0
        self._prev_steer = 0.0
        ts = self.track.get_track_state(self._pos)
        self._progress = ts.progress
        self._lateral = ts.lateral
        self._track_heading = ts.track_heading
        self._arc_length = ts.arc_length
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))

        prev_state = AgentState(
            pos=self._pos.copy(),
            heading=self._heading,
            speed=self._speed,
            progress=self._progress,
            lateral=self._lateral,
            track_heading=self._track_heading,
            on_track=True,
            laps=self._laps,
            arc_length=self._arc_length,
        )

        lat_abs = abs(self._lateral)
        hw = self.track.half_width
        kw = KERB_WIDTH
        if lat_abs <= hw - CAR_HALF_WIDTH:
            surf_grip = 1.0
            roll_fac = 1.0
            on_kerb = False
        elif lat_abs <= hw + kw:
            surf_grip = 0.92
            roll_fac = 1.8
            on_kerb = True
        else:
            surf_grip = 0.35
            roll_fac = 8.0
            on_kerb = False

        car = CarState(x=self._pos[0], y=self._pos[1], heading=self._heading, speed=self._speed, tyre_temp=getattr(self, "_tyre_temp", 95.0))
        car, self._heading_rate = step_physics(
            car, throttle, steer, getattr(self, "car_params", SHANGHAI_PARAMS), DT,
            surface_grip=surf_grip, rolling_factor=roll_fac
        )
        self._pos[:] = car.x, car.y
        self._heading = car.heading
        self._speed = car.speed
        self._tyre_temp = car.tyre_temp
        self._step_count += 1

        ts = self.track.get_track_state(self._pos)
        self._progress = ts.progress
        self._lateral = ts.lateral
        self._track_heading = ts.track_heading

        if self._progress - prev_state.progress < -0.5:
            self._laps += 1

        self._arc_length = ts.arc_length

        legal_surface = abs(ts.lateral) <= hw + kw
        curr_state = AgentState(
            pos=self._pos.copy(),
            heading=self._heading,
            speed=self._speed,
            progress=self._progress,
            lateral=self._lateral,
            track_heading=self._track_heading,
            on_track=legal_surface,
            laps=self._laps,
            arc_length=self._arc_length,
        )

        reward = compute_reward(curr_state, prev_state, self.track)

        c_val = getattr(ts, "curvature", 0.0)
        max_v = getattr(self, "car_params", SHANGHAI_PARAMS).max_speed
        spd_ratio = self._speed / max(1.0, max_v)
        fwd_align = max(0.0, math.cos(self._heading - self._track_heading))

        if abs(c_val) > 0.001:
            in_dir = 1.0 if c_val > 0.0 else -1.0
            inside_offset = self._lateral * in_dir
            if inside_offset > 0.5:
                reward += 0.15 * spd_ratio * min(inside_offset / 10.0, 1.0)
            if on_kerb and abs(c_val) > 0.005:
                reward += 0.25 * spd_ratio
        else:
            reward += 0.12 * spd_ratio * fwd_align * max(0.0, 1.0 - abs(steer) * 2.0)

        reward -= 0.05 * ((steer - getattr(self, "_prev_steer", steer)) ** 2)
        self._prev_steer = steer

        terminated = not legal_surface
        if terminated:
            reward -= 5.0
        truncated = self._step_count >= MAX_STEPS

        if self.render_mode == "human":
            self._render_frame()

        return (
            self._build_obs(),
            reward,
            terminated,
            truncated,
            {"laps": self._laps, "progress": self._progress, "speed": self._speed,
             "cumulative_distance": self._laps * self.track.total_length + self._progress * self.track.total_length,
             "lateral_ratio": abs(self._lateral) / self.track.half_width,
             "tyre_temp": self._tyre_temp,
             "on_kerb": on_kerb},
        )

    def _build_obs(self) -> np.ndarray:
        heading_err = self._heading - self._track_heading
        lateral_norm = float(np.clip(self._lateral / self.track.half_width, -1.0, 1.0))
        rays = (
            self.track.ray_distances(self._pos, self._heading, MAX_RAY_DIST)
            / MAX_RAY_DIST
        )
        max_v = getattr(self, "car_params", SHANGHAI_PARAMS).max_speed
        return np.array(
            [
                self._speed / max_v,
                np.sin(heading_err),
                np.cos(heading_err),
                lateral_norm,
                self._progress,
                float(np.clip(self._heading_rate / MAX_HEADING_RATE, -1.0, 1.0)),
                *np.clip(rays, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def render(self):
        frame = self._render_frame()
        if self.render_mode == "rgb_array":
            return frame

    def _render_frame(self):
        import pygame

        W, H = 800, 600

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Slipstream")
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

        rays = self.track.ray_distances(self._pos, self._heading, MAX_RAY_DIST)
        for a, dist in zip(RAY_ANGLES, rays):
            angle = self._heading + a
            end = self._pos + dist * np.array([np.cos(angle), np.sin(angle)])
            pygame.draw.line(
                surf, (255, 140, 0),
                (int(self._pos[0]), int(self._pos[1])),
                (int(end[0]), int(end[1])),
                1,
            )

        sz = 9
        fwd = np.array([np.cos(self._heading), np.sin(self._heading)])
        left = np.array([-fwd[1], fwd[0]])
        tip = self._pos + fwd * sz
        bl = self._pos - fwd * sz * 0.6 + left * sz * 0.55
        br = self._pos - fwd * sz * 0.6 - left * sz * 0.55
        pygame.draw.polygon(surf, (0, 200, 255), [
            (int(tip[0]), int(tip[1])),
            (int(bl[0]), int(bl[1])),
            (int(br[0]), int(br[1])),
        ])

        font = pygame.font.SysFont("monospace", 14)
        for i, text in enumerate([
            f"Speed:    {self._speed:6.1f}",
            f"HeadRate: {self._heading_rate:+.2f}",
            f"Progress: {self._progress:.3f}",
            f"Laps:     {self._laps}",
            f"Step:     {self._step_count}",
        ]):
            surf.blit(font.render(text, True, (200, 200, 200)), (10, 10 + i * 18))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            assert self._clock is not None
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
        )

    def close(self) -> None:
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
