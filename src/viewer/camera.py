from __future__ import annotations

import math
import numpy as np


class FollowCamera:
    def __init__(self, k: float = 8.0, lookahead_time: float = 0.18):
        self.pos = np.zeros(2, dtype=np.float64)
        self.smoothed_vel = np.zeros(2, dtype=np.float64)
        self.smoothed_target_pos = np.zeros(2, dtype=np.float64)
        self.mode = "auto"
        self.framing_target = "midpoint"
        self.k = k
        self.lookahead_time = lookahead_time

    def snap_to(self, target_pos: np.ndarray, target_vel: np.ndarray | None = None) -> None:
        if target_vel is None:
            target_vel = np.zeros(2, dtype=np.float64)
        self.smoothed_vel = target_vel.astype(np.float64)
        self.smoothed_target_pos = target_pos.astype(np.float64).copy()
        self.pos = target_pos.astype(np.float64) + self.lookahead_time * self.smoothed_vel

    def world_to_screen(
        self,
        world_pt: np.ndarray,
        view_size: tuple[int, int],
        zoom: float,
    ) -> np.ndarray:
        vc = np.array([view_size[0] * 0.5, view_size[1] * 0.5], dtype=np.float64)
        return (world_pt - self.pos) * zoom + vc

    def screen_to_world(
        self,
        screen_pt: np.ndarray,
        view_size: tuple[int, int],
        zoom: float,
    ) -> np.ndarray:
        vc = np.array([view_size[0] * 0.5, view_size[1] * 0.5], dtype=np.float64)
        return (screen_pt - vc) / zoom + self.pos

    def update(
        self,
        pos_ham: np.ndarray,
        vel_ham: np.ndarray,
        pos_ver: np.ndarray,
        vel_ver: np.ndarray,
        leader_is_ham: bool,
        view_size: tuple[int, int],
        zoom: float,
        dt: float,
    ) -> None:
        vw, vh = view_size
        half_ext = min((vw * 0.5) / zoom, (vh * 0.5) / zoom)
        separation = float(np.linalg.norm(pos_ham - pos_ver))

        if self.framing_target == "midpoint":
            if separation > 0.70 * half_ext:
                self.framing_target = "leader"
        else:
            if separation < 0.55 * half_ext:
                self.framing_target = "midpoint"

        if self.mode == "ham":
            t_pos = pos_ham.astype(np.float64)
            t_vel = vel_ham.astype(np.float64)
        elif self.mode == "ver":
            t_pos = pos_ver.astype(np.float64)
            t_vel = vel_ver.astype(np.float64)
        else:
            if self.framing_target == "midpoint":
                t_pos = (pos_ham + pos_ver) * 0.5
                t_vel = (vel_ham + vel_ver) * 0.5
            else:
                t_pos = pos_ham.copy() if leader_is_ham else pos_ver.copy()
                t_vel = vel_ham.copy() if leader_is_ham else vel_ver.copy()

        if not hasattr(self, "smoothed_vel") or np.all(self.smoothed_vel == 0):
            self.smoothed_vel = t_vel.astype(np.float64)
        else:
            v_alpha = 1.0 - math.exp(-dt * 2.5)
            self.smoothed_vel = self.smoothed_vel + v_alpha * (t_vel.astype(np.float64) - self.smoothed_vel)

        if not hasattr(self, "smoothed_target_pos") or np.all(self.smoothed_target_pos == 0):
            self.smoothed_target_pos = t_pos.astype(np.float64).copy()
        else:
            t_alpha = 1.0 - math.exp(-dt * 4.0)
            self.smoothed_target_pos = self.smoothed_target_pos + t_alpha * (t_pos.astype(np.float64) - self.smoothed_target_pos)

        target = self.smoothed_target_pos + self.lookahead_time * self.smoothed_vel
        alpha = 1.0 - math.exp(-dt * self.k)
        self.pos = self.pos + alpha * (target - self.pos)
