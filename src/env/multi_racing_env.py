from __future__ import annotations

from typing import Optional

import math
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .track import Track, SHANGHAI_TRACK_WIDTH
from .car import (
    CarState,
    DEFAULT_PARAMS,
    DT,
    MAX_SPEED,
    CAR_HALF_WIDTH,
    CAR_HALF_LEN,
    CAR_LENGTH,
    CAR_WIDTH,
    GRID_SLOT_SPACING,
    GRID_COL_OFFSET,
    KERB_WIDTH,
    WHEELBASE,
    step_physics,
    DRAFT_CONE_LENGTH,
    DRAFT_CONE_HALF_ANGLE,
    DRAFT_MIN_GAP,
    DRAFT_PEAK_GAP,
    DRAFT_DRAG_REDUCTION,
    DRAFT_SPEED_BOOST,
)
from .racing_env import MAX_STEPS, MAX_HEADING_RATE, MAX_RAY_DIST
from .rewards import (
    compute_reward,
    AgentState,
    DEFAULT_CONTACT_PENALTY,
    DEFAULT_CONTACT_STEP_PENALTY,
    DEFAULT_POSITION_K,
    DEFAULT_POSITION_G0,
    compute_positional_reward,
)
from .collision import obb_overlap, resolve_collision, classify_contact, resolve_penetration

AGENTS = ["agent_0", "agent_1"]
DEFAULT_SPAWN_OFFSET_IDX = max(1, int(round(GRID_SLOT_SPACING / 2.0)))
SPAWN_OFFSET_IDX = DEFAULT_SPAWN_OFFSET_IDX
MAX_OPP_DIST = 300.0
LATERAL_HISTORY_LEN = 20

POSITION_K = DEFAULT_POSITION_K
POSITION_G0 = DEFAULT_POSITION_G0


def compute_draft_intensity(
    follower_pos: np.ndarray,
    leader_pos: np.ndarray,
    leader_heading: float,
    in_contact: bool = False,
    cone_length: float = DRAFT_CONE_LENGTH,
    cone_half_angle: float = DRAFT_CONE_HALF_ANGLE,
    min_gap: float = DRAFT_MIN_GAP,
    peak_gap: float = DRAFT_PEAK_GAP,
) -> float:
    if in_contact:
        return 0.0

    fwd = np.array([math.cos(leader_heading), math.sin(leader_heading)])
    lat_vec = np.array([-math.sin(leader_heading), math.cos(leader_heading)])
    delta = follower_pos - leader_pos

    d_long = -float(np.dot(delta, fwd))
    d_lat = abs(float(np.dot(delta, lat_vec)))

    max_lat = 0.5 * SHANGHAI_TRACK_WIDTH
    core_lat = (14.0 / 44.0) * SHANGHAI_TRACK_WIDTH
    if d_long <= 0.0 or d_long > cone_length or d_lat >= max_lat:
        return 0.0

    car_width = 2.0 * CAR_HALF_WIDTH
    if d_lat < car_width and d_long <= min_gap:
        return 0.0

    if d_lat <= core_lat:
        fade_lat = 1.0 - 0.2 * (d_lat / core_lat)
    else:
        fade_lat = 0.8 * (max_lat - d_lat) / (max_lat - core_lat)

    if d_lat < car_width:
        if d_long <= peak_gap:
            fade_long = (d_long - min_gap) / (peak_gap - min_gap)
        else:
            fade_long = 1.0 - (d_long - peak_gap) / (cone_length - peak_gap)
    else:
        if d_long <= peak_gap:
            fade_long = 1.0
        else:
            fade_long = 1.0 - (d_long - peak_gap) / (cone_length - peak_gap)

    return float(np.clip(fade_long * fade_lat, 0.0, 1.0))


def aggregate_fault_counts(ep_infos: dict, agents: list[str] = AGENTS) -> dict[str, int]:
    return {
        "follower": sum(ep_infos[a].get("fault_log", {}).get("follower", 0) for a in agents),
        "leader": sum(ep_infos[a].get("fault_log", {}).get("leader", 0) for a in agents),
        "neutral": max((ep_infos[a].get("fault_log", {}).get("neutral", 0) for a in agents), default=0),
    }


class MultiRacingEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "multi_racing_v1"}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        enable_draft: bool = True,
        enable_position_reward: bool = True,
        contact_penalty: float = DEFAULT_CONTACT_PENALTY,
        contact_step_penalty: float = DEFAULT_CONTACT_STEP_PENALTY,
        position_k: float = DEFAULT_POSITION_K,
        position_g0: float = DEFAULT_POSITION_G0,
        legacy_collision: bool = False,
        spawn_offset_idx: int = DEFAULT_SPAWN_OFFSET_IDX,
        window_size: tuple[int, int] = (1600, 900),
    ) -> None:
        super().__init__()
        self.track = Track()
        self.render_mode = render_mode
        self.enable_draft = enable_draft
        self.enable_position_reward = enable_position_reward
        self.contact_penalty = contact_penalty
        self.contact_step_penalty = contact_step_penalty
        self.position_k = position_k
        self.position_g0 = position_g0
        self.legacy_collision = legacy_collision
        self.spawn_offset_idx = spawn_offset_idx
        self.window_size = window_size
        self.window_w, self.window_h = window_size
        self.hud_w = 320
        self.track_view_w = self.window_w - self.hud_w
        self.track_view_h = self.window_h
        self._scale: float | None = None
        self._offset: np.ndarray | None = None
        self.possible_agents = AGENTS[:]

        obs_dim = 15 if legacy_collision else 19
        self.observation_spaces = {a: spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32) for a in AGENTS}
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
        self._current_leader = AGENTS[0]
        self._position_swaps = 0
        self._last_respawn_step = {a: -9999 for a in AGENTS}
        self._screen = None
        self._clock = None
        self._track_surface = None
        self._pristine_track_surface = None
        self._fonts = None
        self._car_sprites_base = None
        self._car_rot_cache = None

    def _compute_transform(self) -> None:
        min_x = float(np.min(self.track.outer[:, 0])) - 25.0
        max_x = float(np.max(self.track.outer[:, 0])) + 25.0
        min_y = float(np.min(self.track.outer[:, 1])) - 25.0
        max_y = float(np.max(self.track.outer[:, 1])) + 25.0
        margin = 40.0
        avail_w = self.track_view_w - 2.0 * margin
        avail_h = self.track_view_h - 2.0 * margin
        scale = min(avail_w / (max_x - min_x), avail_h / (max_y - min_y))
        off_x = margin + (avail_w - (max_x - min_x) * scale) / 2.0 - min_x * scale
        off_y = margin + (avail_h - (max_y - min_y) * scale) / 2.0 - min_y * scale
        self._scale = scale
        self._offset = np.array([off_x, off_y], dtype=np.float64)

    def _world_to_screen(self, pt: np.ndarray) -> np.ndarray:
        if self._scale is None:
            self._compute_transform()
        return pt * self._scale + self._offset

    def _world_to_screen_2x(self, pt: np.ndarray) -> np.ndarray:
        if self._scale is None:
            self._compute_transform()
        return pt * (self._scale * 2.0) + (self._offset * 2.0)

    def _build_car_sprite(self, agent: str):
        import pygame
        cfg = {
            "agent_0": {
                "primary": (11, 24, 60),
                "accent": (220, 20, 40),
                "detail": (255, 215, 0),
                "helmet": (255, 140, 20),
            },
            "agent_1": {
                "primary": (220, 20, 35),
                "accent": (20, 20, 24),
                "detail": (255, 220, 0),
                "helmet": (255, 225, 30),
            },
        }[agent]

        scale = self._scale
        c_len_s = CAR_LENGTH * scale
        c_wid_s = CAR_WIDTH * scale
        wb_s = WHEELBASE * scale

        l_4x = max(16, int(round(c_len_s * 4.0)))
        w_4x = max(8, int(round(c_wid_s * 4.0)))
        pad = 8
        cw_4x = l_4x + pad
        ch_4x = w_4x + pad
        cx, cy = cw_4x / 2.0, ch_4x / 2.0
        hl = l_4x / 2.0
        hw = w_4x / 2.0
        wb = wb_s * 4.0

        surf_4x = pygame.Surface((cw_4x, ch_4x), pygame.SRCALPHA)

        wl_4x = max(4, int(l_4x * 0.28))
        ww_4x = max(2, int(w_4x * 0.26))
        for wx in (cx + wb / 2.0, cx - wb / 2.0):
            for wy in (cy - hw + ww_4x / 2.0, cy + hw - ww_4x / 2.0):
                w_r = pygame.Rect(int(wx - wl_4x / 2.0), int(wy - ww_4x / 2.0), wl_4x, ww_4x)
                pygame.draw.rect(surf_4x, (15, 15, 18), w_r, border_radius=1)

        fw_rect = pygame.Rect(int(cx + hl * 0.75), int(cy - hw), max(2, int(hl * 0.25)), int(hw * 2))
        pygame.draw.rect(surf_4x, cfg["accent"], fw_rect)
        rw_rect = pygame.Rect(int(cx - hl), int(cy - hw * 0.9), max(2, int(hl * 0.25)), int(hw * 1.8))
        pygame.draw.rect(surf_4x, cfg["accent"], rw_rect)

        chassis_pts = [
            (cx + hl, cy),
            (cx + hl * 0.6, cy - hw * 0.4),
            (cx + hl * 0.1, cy - hw * 0.75),
            (cx - hl * 0.6, cy - hw * 0.75),
            (cx - hl * 0.9, cy - hw * 0.45),
            (cx - hl * 0.9, cy + hw * 0.45),
            (cx - hl * 0.6, cy + hw * 0.75),
            (cx + hl * 0.1, cy + hw * 0.75),
            (cx + hl * 0.6, cy + hw * 0.4),
        ]
        pygame.draw.polygon(surf_4x, cfg["primary"], chassis_pts)
        pygame.draw.polygon(surf_4x, (12, 14, 18), chassis_pts, 1)

        pygame.draw.circle(surf_4x, cfg["detail"], (int(cx + hl * 0.9), int(cy)), max(1, int(hw * 0.2)))
        pygame.draw.ellipse(surf_4x, (12, 14, 18), pygame.Rect(int(cx - hl * 0.3), int(cy - hw * 0.35), int(hl * 0.6), int(hw * 0.7)))
        pygame.draw.circle(surf_4x, (30, 35, 42), (int(cx - hl * 0.05), int(cy)), max(2, int(hw * 0.4)), 1)
        pygame.draw.circle(surf_4x, cfg["helmet"], (int(cx - hl * 0.1), int(cy)), max(1, int(hw * 0.3)))

        cw_1x = max(4, int(round(cw_4x / 4.0)))
        ch_1x = max(2, int(round(ch_4x / 4.0)))
        return pygame.transform.smoothscale(surf_4x, (cw_1x, ch_1x))

    def _get_car_sprite(self, agent: str, heading: float):
        import pygame
        if self._car_sprites_base is None:
            self._car_sprites_base = {
                "agent_0": self._build_car_sprite("agent_0"),
                "agent_1": self._build_car_sprite("agent_1"),
            }
            self._car_rot_cache = {"agent_0": {}, "agent_1": {}}

        deg = int(round(math.degrees(heading) / 3.0)) * 3 % 360
        cache = self._car_rot_cache[agent]
        if deg not in cache:
            cache[deg] = pygame.transform.rotate(self._car_sprites_base[agent], -deg)
        return cache[deg]

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

        if self._track_surface is not None and getattr(self, "_pristine_track_surface", None) is not None:
            self._track_surface.blit(self._pristine_track_surface, (0, 0))

        if hasattr(self.track, "get_starting_grid") and getattr(self.track, "circuit", "") == "shanghai":
            grid = self.track.get_starting_grid()
            keys = ["agent_0", "agent_1"]
            if self.np_random.integers(0, 2):
                keys = keys[::-1]
            self._state = {}
            for agent, slot_key in zip(AGENTS, keys):
                pos, heading = grid[slot_key]
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
                    "lap_start_time": None,
                    "last_lap_time": None,
                    "best_lap_time": None,
                    "gap_to_leader_seconds": 0.0,
                    "cumulative_distance": 0.0,
                    "respawn": False,
                    "respawns": 0,
                    "prev_colliding": False,
                    "collision_count": 0,
                    "steps_in_contact": 0,
                    "start_offset": ts.arc_length,
                    "lateral_history": [ts.lateral] * LATERAL_HISTORY_LEN,
                    "fault_log": {"follower": 0, "leader": 0, "neutral": 0},
                    "draft_intensity": 0.0,
                }
        else:
            slots = [0, self.spawn_offset_idx]
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
                    "lap_start_time": None,
                    "last_lap_time": None,
                    "best_lap_time": None,
                    "gap_to_leader_seconds": 0.0,
                    "cumulative_distance": 0.0,
                    "respawn": False,
                    "respawns": 0,
                    "prev_colliding": False,
                    "collision_count": 0,
                    "steps_in_contact": 0,
                    "start_offset": ts.arc_length,
                    "lateral_history": [ts.lateral] * LATERAL_HISTORY_LEN,
                    "fault_log": {"follower": 0, "leader": 0, "neutral": 0},
                    "draft_intensity": 0.0,
                }

        self._position_swaps = 0
        self._last_respawn_step = {a: -9999 for a in AGENTS}
        race_pos_0 = self._state[AGENTS[0]]["start_offset"]
        race_pos_1 = self._state[AGENTS[1]]["start_offset"]
        self._current_leader = AGENTS[0] if race_pos_0 >= race_pos_1 else AGENTS[1]

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

        # Ego-frame opponent velocity components (dims 15-16)
        opp_vel_world = opp["speed"] * np.array([math.cos(opp["heading"]), math.sin(opp["heading"])])
        c, ss = math.cos(s["heading"]), math.sin(s["heading"])
        opp_vel_ego_fwd = float(np.clip((c * opp_vel_world[0] + ss * opp_vel_world[1]) / MAX_SPEED, -1.0, 1.0))
        opp_vel_ego_lat = float(np.clip((-ss * opp_vel_world[0] + c * opp_vel_world[1]) / MAX_SPEED, -1.0, 1.0))

        # Relative heading (dim 17)
        rel_heading = opp["heading"] - s["heading"]
        opp_rel_heading = float(np.clip(math.sin(rel_heading), -1.0, 1.0))

        if self.enable_draft:
            draft_ego = compute_draft_intensity(
                s["pos"], opp["pos"], opp["heading"], in_contact=s["prev_colliding"]
            )
            draft_opp = compute_draft_intensity(
                opp["pos"], s["pos"], s["heading"], in_contact=s["prev_colliding"]
            )
        else:
            draft_ego = 0.0
            draft_opp = 0.0

        base_obs = [
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
        ]
        if self.legacy_collision:
            return np.array(base_obs, dtype=np.float32)

        return np.array(
            base_obs + [
                opp_vel_ego_lat,
                opp_rel_heading,
                draft_ego,
                draft_opp,
            ],
            dtype=np.float32,
        )

    def _build_info(self, agent: str) -> dict:
        s = self._state[agent]
        return {
            "laps": s["laps"],
            "progress": s["progress"],
            "speed": s["speed"],
            "respawn": s["respawn"],
            "respawns": s["respawns"],
            "collision": s["prev_colliding"],
            "collision_count": s["collision_count"],
            "steps_in_contact": s["steps_in_contact"],
            "cumulative_distance": s["cumulative_distance"],
            "start_offset": s["start_offset"],
            "lateral_ratio": abs(s["lateral"]) / self.track.half_width,
            "fault_log": s["fault_log"].copy(),
            "position_swaps": self._position_swaps,
            "draft_intensity": s.get("draft_intensity", 0.0),
            "is_drafting": s.get("draft_intensity", 0.0) > 0.0,
            "lap_start_time": s.get("lap_start_time", None),
            "last_lap_time": s.get("last_lap_time", None),
            "best_lap_time": s.get("best_lap_time", None),
            "gap_to_leader_seconds": s.get("gap_to_leader_seconds", 0.0),
        }

    def _compute_tactical_rewards(self) -> dict[str, float]:
        rewards = {AGENTS[0]: 0.0, AGENTS[1]: 0.0}
        s0, s1 = self._state[AGENTS[0]], self._state[AGENTS[1]]
        race_pos_0 = s0["cumulative_distance"] + s0["start_offset"]
        race_pos_1 = s1["cumulative_distance"] + s1["start_offset"]

        if race_pos_0 >= race_pos_1:
            lead_ag, foll_ag = AGENTS[0], AGENTS[1]
        else:
            lead_ag, foll_ag = AGENTS[1], AGENTS[0]

        lead_s, foll_s = self._state[lead_ag], self._state[foll_ag]
        lead_ts = self.track.get_track_state(lead_s["pos"])
        gap_sec = foll_s.get("gap_to_leader_seconds", 0.0)

        lead_curv = getattr(lead_ts, "curvature", 0.0)
        if gap_sec < 1.5 and abs(lead_curv) > 0.001:
            inside_dir = 1.0 if lead_curv > 0.0 else -1.0
            if lead_s["lateral"] * inside_dir > 2.0:
                rewards[lead_ag] += 0.08 * min(abs(lead_s["lateral"]) / (self.track.half_width * 0.5), 1.0)

        fwd_lead = np.array([math.cos(lead_s["heading"]), math.sin(lead_s["heading"])])
        lat_lead = np.array([-fwd_lead[1], fwd_lead[0]])
        delta = foll_s["pos"] - lead_s["pos"]
        d_long = -float(np.dot(delta, fwd_lead))
        d_lat = abs(float(np.dot(delta, lat_lead)))

        if 0.0 < d_long < 45.0 and gap_sec < 1.0 and 8.0 < d_lat <= 22.0:
            rewards[foll_ag] += 0.10 * min((d_lat - 8.0) / 6.0, 1.0)

        return rewards

    def step(self, actions: dict[str, np.ndarray]):
        rewards = {}
        terminations = {a: False for a in self.agents}
        truncations = {}
        infos = {}

        draft_ints = {}
        for agent in self.agents:
            opp_agent = AGENTS[1] if agent == AGENTS[0] else AGENTS[0]
            s = self._state[agent]
            opp = self._state[opp_agent]
            if self.enable_draft:
                draft_ints[agent] = compute_draft_intensity(
                    s["pos"], opp["pos"], opp["heading"], in_contact=s["prev_colliding"]
                )
            else:
                draft_ints[agent] = 0.0

        for agent in self.agents:
            s = self._state[agent]
            ts_start = self.track.get_track_state(s["pos"])
            s["progress"] = ts_start.progress
            s["lateral"] = ts_start.lateral
            s["track_heading"] = ts_start.track_heading
            s["arc_length"] = ts_start.arc_length
            if abs(s["lateral"]) < 1e-4 and s["speed"] == 0.0:
                s["heading"] = ts_start.track_heading
            s["respawn"] = False
            action = actions[agent]
            steer = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip(action[1], -1.0, 1.0))
            draft_int = draft_ints[agent]
            s["draft_intensity"] = draft_int

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
            car, heading_rate = step_physics(
                car, throttle, steer, DEFAULT_PARAMS, DT, draft_intensity=draft_int
            )
            s["pos"][:] = car.x, car.y
            s["heading"] = car.heading
            s["speed"] = car.speed
            s["heading_rate"] = heading_rate

            ts = self.track.get_track_state(s["pos"])
            s["progress"] = ts.progress
            s["lateral"] = ts.lateral
            s["track_heading"] = ts.track_heading

            if s["progress"] - prev_state.progress < -0.5:
                curr_time = self._step_count * DT
                if s["lap_start_time"] is None:
                    s["laps"] = 1
                    s["lap_start_time"] = curr_time
                else:
                    lap_time = curr_time - s["lap_start_time"]
                    s["last_lap_time"] = lap_time
                    if s["best_lap_time"] is None or lap_time < s["best_lap_time"]:
                        s["best_lap_time"] = lap_time
                    s["lap_start_time"] = curr_time
                    s["laps"] += 1

            s["arc_length"] = ts.arc_length

            if ts.on_track:
                arc_delta = s["arc_length"] - prev_state.arc_length
                if arc_delta < -self.track.total_length / 2.0:
                    arc_delta += self.track.total_length
                elif arc_delta > self.track.total_length / 2.0:
                    arc_delta -= self.track.total_length
                s["cumulative_distance"] += max(0.0, arc_delta)

            s["lateral_history"].append(ts.lateral)
            if len(s["lateral_history"]) > LATERAL_HISTORY_LEN:
                s["lateral_history"].pop(0)

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
                s["respawn"] = True
                s["respawns"] += 1
                self._respawn(agent)

            rewards[agent] = reward

        self._step_count += 1

        # --- OBB collision detection and response ---
        if self.legacy_collision:
            s0, s1 = self._state[AGENTS[0]], self._state[AGENTS[1]]
            colliding = float(np.linalg.norm(s0["pos"] - s1["pos"])) < 2.0 * CAR_HALF_WIDTH
            if colliding:
                s0["steps_in_contact"] += 1
                s1["steps_in_contact"] += 1
                if not s0["prev_colliding"]:
                    s0["collision_count"] += 1
                    s1["collision_count"] += 1
            for s in (s0, s1):
                s["prev_colliding"] = colliding
        else:
            s0, s1 = self._state[AGENTS[0]], self._state[AGENTS[1]]
            overlapping, normal, pen = obb_overlap(
                s0["pos"], s0["heading"],
                s1["pos"], s1["heading"],
                CAR_HALF_LEN, CAR_HALF_WIDTH,
            )

            if overlapping:
                s0["steps_in_contact"] += 1
                s1["steps_in_contact"] += 1
                rewards[AGENTS[0]] += self.contact_step_penalty
                rewards[AGENTS[1]] += self.contact_step_penalty
                if not s0["prev_colliding"]:
                    s0["collision_count"] += 1
                    s1["collision_count"] += 1
                    rewards[AGENTS[0]] += self.contact_penalty
                    rewards[AGENTS[1]] += self.contact_penalty

                    # Determine leader by cumulative distance
                    if s0["cumulative_distance"] >= s1["cumulative_distance"]:
                        leader_agent, follower_agent = AGENTS[0], AGENTS[1]
                        leader_s, follower_s = s0, s1
                    else:
                        leader_agent, follower_agent = AGENTS[1], AGENTS[0]
                        leader_s, follower_s = s1, s0

                    contact_normal = -normal if leader_agent == AGENTS[0] else normal
                    fault = classify_contact(
                        normal=contact_normal,
                        heading_leader=leader_s["heading"],
                        vel_a=s0["speed"] * np.array([math.cos(s0["heading"]), math.sin(s0["heading"])]),
                        vel_b=s1["speed"] * np.array([math.cos(s1["heading"]), math.sin(s1["heading"])]),
                        leader_id=0 if leader_agent == AGENTS[0] else 1,
                        lateral_history_leader=leader_s["lateral_history"],
                        follower_lateral=follower_s["lateral"],
                    )
                    if fault == "follower_fault":
                        follower_s["fault_log"]["follower"] += 1
                    elif fault == "leader_fault":
                        leader_s["fault_log"]["leader"] += 1
                    else:
                        leader_s["fault_log"]["neutral"] += 1
                        follower_s["fault_log"]["neutral"] += 1

                # Momentum transfer via impulse resolution
                vel0 = s0["speed"] * np.array([math.cos(s0["heading"]), math.sin(s0["heading"])])
                vel1 = s1["speed"] * np.array([math.cos(s1["heading"]), math.sin(s1["heading"])])
                vel0_new, vel1_new = resolve_collision(s0["pos"], vel0, s1["pos"], vel1, normal)
                s0["speed"] = float(np.clip(np.linalg.norm(vel0_new), 0.0, MAX_SPEED))
                s1["speed"] = float(np.clip(np.linalg.norm(vel1_new), 0.0, MAX_SPEED))

                pos0_new, pos1_new = resolve_penetration(s0["pos"], s1["pos"], normal, pen)
                s0["pos"][:] = pos0_new
                s1["pos"][:] = pos1_new

                ov2, normal2, pen2 = obb_overlap(
                    s0["pos"], s0["heading"],
                    s1["pos"], s1["heading"],
                    CAR_HALF_LEN, CAR_HALF_WIDTH,
                )
                if ov2 and pen2 > 0.05:
                    p0_sub, p1_sub = resolve_penetration(s0["pos"], s1["pos"], normal2, pen2)
                    s0["pos"][:] = p0_sub
                    s1["pos"][:] = p1_sub

                for a in AGENTS:
                    s = self._state[a]
                    if not s["respawn"]:
                        ts = self.track.get_track_state(s["pos"])
                        s["progress"] = ts.progress
                        s["lateral"] = ts.lateral
                        s["track_heading"] = ts.track_heading
                        s["arc_length"] = ts.arc_length
                        if len(s["lateral_history"]) > 0:
                            s["lateral_history"][-1] = ts.lateral
                        if not ts.on_track:
                            rewards[a] = -5.0
                            s["respawn"] = True
                            s["respawns"] += 1
                            self._respawn(a)

            for s in (s0, s1):
                s["prev_colliding"] = overlapping

        race_pos_0 = self._state[AGENTS[0]]["cumulative_distance"] + self._state[AGENTS[0]]["start_offset"]
        race_pos_1 = self._state[AGENTS[1]]["cumulative_distance"] + self._state[AGENTS[1]]["start_offset"]
        car_len = 2.0 * CAR_HALF_LEN
        recent_respawn = any(self._step_count - self._last_respawn_step[a] <= 10 for a in AGENTS)

        if self._current_leader == AGENTS[0]:
            if race_pos_1 - race_pos_0 >= car_len:
                self._current_leader = AGENTS[1]
                if not recent_respawn:
                    self._position_swaps += 1
                    if not self._state[AGENTS[1]]["prev_colliding"]:
                        rewards[AGENTS[1]] += 1.5
        else:
            if race_pos_0 - race_pos_1 >= car_len:
                self._current_leader = AGENTS[0]
                if not recent_respawn:
                    self._position_swaps += 1
                    if not self._state[AGENTS[0]]["prev_colliding"]:
                        rewards[AGENTS[0]] += 1.5

        if self.enable_position_reward:
            r_pos_0 = compute_positional_reward(race_pos_0, race_pos_1, self.position_k, self.position_g0)
            r_pos_1 = compute_positional_reward(race_pos_1, race_pos_0, self.position_k, self.position_g0)
            rewards[AGENTS[0]] += r_pos_0
            rewards[AGENTS[1]] += r_pos_1

        gap_dist = abs(race_pos_0 - race_pos_1)
        if race_pos_0 >= race_pos_1:
            leader_ag, follower_ag = AGENTS[0], AGENTS[1]
        else:
            leader_ag, follower_ag = AGENTS[1], AGENTS[0]
        follower_spd = self._state[follower_ag]["speed"]
        gap_sec = float(gap_dist / max(follower_spd, 15.0)) if gap_dist > 0.05 else 0.0
        self._state[leader_ag]["gap_to_leader_seconds"] = 0.0
        self._state[follower_ag]["gap_to_leader_seconds"] = gap_sec

        tac_rewards = self._compute_tactical_rewards()
        for a in AGENTS:
            rewards[a] += tac_rewards[a]

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
        self._last_respawn_step[agent] = self._step_count
        s = self._state[agent]
        opp = self._state[AGENTS[1] if agent == AGENTS[0] else AGENTS[0]]
        opp_pos = opp["pos"]
        opp_heading = opp["heading"]

        idx = self.track.nearest_idx(s["pos"])
        spawn_pos = self.track.centerline[idx].copy()
        heading = float(np.arctan2(self.track.tangents[idx, 1], self.track.tangents[idx, 0]))

        if self.legacy_collision:
            if np.linalg.norm(spawn_pos - opp_pos) < 2.0 * CAR_HALF_WIDTH:
                spawn_pos = spawn_pos + self.track.normals[idx] * 2.0 * CAR_HALF_WIDTH
        else:
            ov, _, _ = obb_overlap(
                spawn_pos, heading, opp_pos, opp_heading, CAR_HALF_LEN, CAR_HALF_WIDTH
            )
            while ov:
                idx = (idx - self.spawn_offset_idx) % len(self.track.centerline)
                spawn_pos = self.track.centerline[idx].copy()
                heading = float(np.arctan2(self.track.tangents[idx, 1], self.track.tangents[idx, 0]))
                ov, _, _ = obb_overlap(
                    spawn_pos, heading, opp_pos, opp_heading, CAR_HALF_LEN, CAR_HALF_WIDTH
                )

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

    def _bake_track_surface(self, W: int, H: int, sidebar_x: int) -> None:
        import pygame

        if self._scale is None:
            self._compute_transform()

        surf_2x = pygame.Surface((W * 2, H * 2))
        surf_2x.fill((18, 30, 20))

        w2 = sidebar_x * 2
        h2 = H * 2
        step = 60
        for d in range(-h2, w2 + h2, step * 2):
            pts = [(d, 0), (d + step, 0), (d + step - h2, h2), (d - h2, h2)]
            pygame.draw.polygon(surf_2x, (22, 35, 24), pts)

        N = len(self.track.centerline)
        outer = self.track.outer
        inner = self.track.inner
        curv = self.track.curvatures
        norm = self.track.normals
        tang = self.track.tangents
        w_track = self.track.half_width * 2.0

        pit_indices = list(range(5850, N)) + list(range(0, 350))
        w_pit = 0.5 * w_track
        for idx in range(len(pit_indices) - 1):
            i = pit_indices[idx]
            nxt = pit_indices[idx + 1]
            p0 = inner[i]
            p1 = inner[nxt]
            pit0 = p0 - norm[i] * w_pit
            pit1 = p1 - norm[nxt] * w_pit
            pygame.draw.polygon(surf_2x, (27, 28, 31), [self._world_to_screen_2x(p) for p in (p0, p1, pit1, pit0)])
            pygame.draw.line(surf_2x, (140, 145, 155), self._world_to_screen_2x(p0), self._world_to_screen_2x(p1), 4)

        team_colors = [
            (195, 25, 30), (15, 25, 75), (0, 160, 140), (255, 135, 0), (0, 110, 75),
            (30, 65, 180), (180, 20, 40), (80, 140, 200), (220, 220, 225), (70, 75, 85)
        ]
        for b_i, i in enumerate(pit_indices[20:-20:25]):
            center = inner[i] - norm[i] * (w_pit + 4.0)
            f_box = tang[i] * 10.0
            s_box = norm[i] * 4.0
            pts = [center + f_box + s_box, center + f_box - s_box, center - f_box - s_box, center - f_box + s_box]
            col = team_colors[b_i % len(team_colors)]
            pygame.draw.polygon(surf_2x, col, [self._world_to_screen_2x(p) for p in pts])

        for i in range(N):
            nxt = (i + 1) % N
            r_i = 1.0 / (abs(curv[i]) + 1e-9)
            r_nxt = 1.0 / (abs(curv[nxt]) + 1e-9)
            t_i = min(1.0, max(0.0, (6.0 * w_track / r_i - 1.0) / 0.3)) if r_i < 6.0 * w_track else 0.0
            t_nxt = min(1.0, max(0.0, (6.0 * w_track / r_nxt - 1.0) / 0.3)) if r_nxt < 6.0 * w_track else 0.0
            if t_i > 0 or t_nxt > 0:
                if curv[i] >= 0:
                    p0, p1 = inner[i], inner[nxt]
                    ro0 = p0 - norm[i] * (0.35 * w_track * t_i)
                    ro1 = p1 - norm[nxt] * (0.35 * w_track * t_nxt)
                    gr0 = ro0 - norm[i] * (0.25 * w_track * t_i)
                    gr1 = ro1 - norm[nxt] * (0.25 * w_track * t_nxt)
                else:
                    p0, p1 = outer[i], outer[nxt]
                    ro0 = p0 + norm[i] * (0.35 * w_track * t_i)
                    ro1 = p1 + norm[nxt] * (0.35 * w_track * t_nxt)
                    gr0 = ro0 + norm[i] * (0.25 * w_track * t_i)
                    gr1 = ro1 + norm[nxt] * (0.25 * w_track * t_nxt)
                pygame.draw.polygon(surf_2x, (194, 168, 126), [self._world_to_screen_2x(p) for p in (ro0, ro1, gr1, gr0)])
                pygame.draw.polygon(surf_2x, (52, 54, 58), [self._world_to_screen_2x(p) for p in (p0, p1, ro1, ro0)])

        for i in range(N):
            nxt = (i + 1) % N
            noise = ((i * 73 + 19) % 7) - 3
            col = (32 + noise, 33 + noise, 36 + noise)
            poly = [self._world_to_screen_2x(outer[i]), self._world_to_screen_2x(outer[nxt]), self._world_to_screen_2x(inner[nxt]), self._world_to_screen_2x(inner[i])]
            pygame.draw.polygon(surf_2x, col, poly)

        for idx in range(3980, 5220, 50):
            c = self.track.centerline[idx]
            t_vec = tang[idx]
            n_vec = norm[idx]
            tip = c + t_vec * 8.0
            pl = c - t_vec * 4.0 - n_vec * 6.0
            pr = c - t_vec * 4.0 + n_vec * 6.0
            pygame.draw.lines(surf_2x, (44, 46, 52), False, [self._world_to_screen_2x(pl), self._world_to_screen_2x(tip), self._world_to_screen_2x(pr)], 4)

        for idx in range(50, 320, 50):
            c = self.track.centerline[idx]
            t_vec = tang[idx]
            n_vec = norm[idx]
            tip = c + t_vec * 8.0
            pl = c - t_vec * 4.0 - n_vec * 6.0
            pr = c - t_vec * 4.0 + n_vec * 6.0
            pygame.draw.lines(surf_2x, (44, 46, 52), False, [self._world_to_screen_2x(pl), self._world_to_screen_2x(tip), self._world_to_screen_2x(pr)], 4)

        p_out = outer[0]
        p_in = inner[0]
        tang0 = tang[0]
        for col in range(14):
            for row in range(2):
                is_white = (col + row) % 2 == 0
                c = (245, 248, 255) if is_white else (25, 27, 30)
                t0 = col / 14.0
                t1 = (col + 1) / 14.0
                p0a = p_in + (p_out - p_in) * t0 + tang0 * (row * 3.0)
                p1a = p_in + (p_out - p_in) * t1 + tang0 * (row * 3.0)
                p1b = p_in + (p_out - p_in) * t1 + tang0 * ((row + 1) * 3.0)
                p0b = p_in + (p_out - p_in) * t0 + tang0 * ((row + 1) * 3.0)
                pygame.draw.polygon(surf_2x, c, [self._world_to_screen_2x(p) for p in (p0a, p1a, p1b, p0b)])

        box_hl = (CAR_LENGTH * 1.15) / 2.0
        box_hw = (CAR_WIDTH * 1.5) / 2.0
        for k in range(20):
            s_slot = (self.track.total_length - 1.0 * w_track - k * 0.57 * w_track) % self.track.total_length
            idx_slot = int(np.clip(s_slot / 2.0, 0, len(self.track.centerline) - 1))
            t_slot = tang[idx_slot]
            n_slot = norm[idx_slot]
            lat_sign = 1.0 if (k % 2 == 0) else -1.0
            lat_off = lat_sign * (0.22 * w_track)
            pos_slot = self.track.centerline[idx_slot] + n_slot * lat_off
            fwd = t_slot * box_hl
            side = n_slot * box_hw
            box_pts = [pos_slot + fwd + side, pos_slot + fwd - side, pos_slot - fwd - side, pos_slot - fwd + side]
            poly = [self._world_to_screen_2x(p) for p in box_pts]
            pygame.draw.polygon(surf_2x, (230, 235, 245), poly, 2)
            pygame.draw.line(surf_2x, (245, 248, 255), poly[0], poly[1], 5)

        for i in range(N):
            nxt = (i + 1) % N
            r_i = 1.0 / (abs(curv[i]) + 1e-9)
            r_nxt = 1.0 / (abs(curv[nxt]) + 1e-9)
            t_i = min(1.0, max(0.0, (4.0 * w_track / r_i - 1.0) / 0.3)) if r_i < 4.0 * w_track else 0.0
            t_nxt = min(1.0, max(0.0, (4.0 * w_track / r_nxt - 1.0) / 0.3)) if r_nxt < 4.0 * w_track else 0.0
            if t_i > 0 or t_nxt > 0:
                if curv[i] >= 0:
                    p0, p1 = outer[i], outer[nxt]
                    k0 = p0 + norm[i] * (0.07 * w_track * t_i)
                    k1 = p1 + norm[nxt] * (0.07 * w_track * t_nxt)
                else:
                    p0, p1 = inner[i], inner[nxt]
                    k0 = p0 - norm[i] * (0.07 * w_track * t_i)
                    k1 = p1 - norm[nxt] * (0.07 * w_track * t_nxt)
                is_red = (i // 4) % 2 == 0
                col = (215, 30, 30) if is_red else (245, 245, 248)
                pygame.draw.polygon(surf_2x, col, [self._world_to_screen_2x(p) for p in (p0, p1, k1, k0)])

        outer_2x = [self._world_to_screen_2x(p) for p in outer]
        inner_2x = [self._world_to_screen_2x(p) for p in inner]
        pygame.draw.lines(surf_2x, (240, 242, 248), True, outer_2x, 6)
        pygame.draw.lines(surf_2x, (240, 242, 248), True, inner_2x, 6)

        corner_labels = [
            (1, (790, 62)), (2, (835, 160)), (3, (770, 188)), (4, (795, 283)),
            (5, (1090, 192)), (6, (1328, 283)), (7, (935, 402)), (8, (922, 628)),
            (9, (807, 693)), (10, (790, 797)), (11, (1283, 797)), (12, (1288, 672)),
            (13, (1403, 793)), (14, (97, 884)), (15, (183, 808)), (16, (490, 835)),
        ]
        ref_scale = self.track.centerline[0, 0] / 599.05
        font_cnum = pygame.font.Font(None, 24)
        for num, (rx, ry) in corner_labels:
            w_pt = np.array([rx * ref_scale, ry * ref_scale])
            s_pt = self._world_to_screen_2x(w_pt)
            txt = font_cnum.render(str(num), True, (215, 225, 240))
            rect = txt.get_rect(center=(int(s_pt[0]), int(s_pt[1])))
            surf_2x.blit(txt, rect)

        scaled = pygame.transform.smoothscale(surf_2x, (W, H))
        try:
            final_surf = scaled.convert()
        except Exception:
            final_surf = scaled
        self._pristine_track_surface = final_surf.copy()
        self._track_surface = final_surf.copy()

    def _render_frame(self):
        import pygame
        from .track import RAY_ANGLES

        W, H = self.window_w, self.window_h
        SIDEBAR_X = self.track_view_w
        SIDEBAR_W = self.hud_w

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                flags = pygame.DOUBLEBUF | pygame.SCALED
                try:
                    self._screen = pygame.display.set_mode((W, H), flags, vsync=1)
                except Exception:
                    try:
                        self._screen = pygame.display.set_mode((W, H), flags)
                    except Exception:
                        self._screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Slipstream - Formula 1 Shanghai Grand Prix")
            else:
                self._screen = pygame.Surface((W, H))
            self._clock = pygame.time.Clock()

        if self._track_surface is None:
            self._bake_track_surface(W, H, SIDEBAR_X)

        if self._fonts is None:
            font_family = "ubuntu" if "ubuntu" in pygame.font.get_fonts() else "dejavusans"
            mono_family = "ubuntumono" if "ubuntumono" in pygame.font.get_fonts() else "monospace"
            self._fonts = {
                "title": pygame.font.SysFont(font_family, 17, bold=True),
                "sub": pygame.font.SysFont(font_family, 11, bold=True),
                "body": pygame.font.SysFont(font_family, 12, bold=True),
                "stat": pygame.font.SysFont(font_family, 13, bold=True),
                "num": pygame.font.SysFont(mono_family, 12, bold=True),
                "badge": pygame.font.SysFont(font_family, 10, bold=True),
                "f1": pygame.font.SysFont(font_family, 14, bold=True),
            }

        surf = self._screen
        surf.blit(self._track_surface, (0, 0))

        team_configs = {
            "agent_0": {
                "name": "RED BULL RACING",
                "car_num": "#1 VER",
                "primary": (11, 24, 60),
                "accent": (220, 20, 40),
                "detail": (255, 215, 0),
                "ray": (20, 60, 160),
            },
            "agent_1": {
                "name": "SCUDERIA FERRARI",
                "car_num": "#44 HAM",
                "primary": (220, 20, 35),
                "accent": (20, 20, 24),
                "detail": (255, 220, 0),
                "ray": (220, 50, 40),
            },
        }

        scale = self._scale
        for agent in AGENTS:
            s = self._state[agent]
            pos, heading = s["pos"], s["heading"]
            pos_s = self._world_to_screen(pos)
            cfg = team_configs[agent]

            rays = self.track.ray_distances(pos, heading, MAX_RAY_DIST)
            for a, dist in zip(RAY_ANGLES, rays):
                angle = heading + a
                end = pos + dist * np.array([np.cos(angle), np.sin(angle)])
                end_s = self._world_to_screen(end)
                pygame.draw.line(
                    surf, cfg["ray"],
                    (int(pos_s[0]), int(pos_s[1])),
                    (int(end_s[0]), int(end_s[1])),
                    1,
                )
                pygame.draw.circle(surf, cfg["ray"], (int(end_s[0]), int(end_s[1])), 2)

            car_surf = self._get_car_sprite(agent, heading)
            rect = car_surf.get_rect(center=(int(pos_s[0]), int(pos_s[1])))
            surf.blit(car_surf, rect)

            if self._state[AGENTS[0]]["prev_colliding"]:
                c_hlen = CAR_HALF_LEN * scale
                c_hwid = CAR_HALF_WIDTH * scale
                fwd = np.array([np.cos(heading), np.sin(heading)]) * scale
                left = np.array([-fwd[1], fwd[0]])
                fl = pos_s + fwd * c_hlen + left * c_hwid
                fr = pos_s + fwd * c_hlen - left * c_hwid
                rr = pos_s - fwd * c_hlen - left * c_hwid
                rl = pos_s - fwd * c_hlen + left * c_hwid
                pygame.draw.polygon(
                    surf, (255, 220, 40),
                    [(int(p[0]), int(p[1])) for p in (fl, fr, rr, rl)],
                    2,
                )

        pygame.draw.rect(surf, (14, 17, 24), (SIDEBAR_X, 0, SIDEBAR_W, H))
        pygame.draw.line(surf, (36, 44, 60), (SIDEBAR_X, 0), (SIDEBAR_X, H), 2)

        font_title = self._fonts["title"]
        font_sub = self._fonts["sub"]
        font_body = self._fonts["body"]
        font_stat = self._fonts["stat"]
        font_num = self._fonts["num"]
        font_badge = self._fonts["badge"]
        font_f1 = self._fonts["f1"]

        pygame.draw.rect(surf, (220, 20, 35), (SIDEBAR_X + 16, 14, 34, 22), border_radius=4)
        surf.blit(font_f1.render("F1", True, (255, 255, 255)), (SIDEBAR_X + 24, 16))
        surf.blit(font_title.render("GRAND PRIX", True, (255, 255, 255)), (SIDEBAR_X + 58, 14))
        surf.blit(font_sub.render("SHANGHAI INTERNATIONAL CIRCUIT", True, (135, 155, 185)), (SIDEBAR_X + 58, 34))

        pygame.draw.rect(surf, (22, 28, 40), (SIDEBAR_X + 16, 58, 148, 24), border_radius=5)
        surf.blit(font_num.render(f"STEP: {self._step_count:04d}", True, (185, 215, 255)), (SIDEBAR_X + 24, 62))
        pygame.draw.rect(surf, (22, 28, 40), (SIDEBAR_X + 176, 58, 148, 24), border_radius=5)
        surf.blit(font_num.render(f"TIME: {self._step_count * DT:5.1f}s", True, (185, 215, 255)), (SIDEBAR_X + 184, 62))

        score_0 = self._state["agent_0"]["laps"] + self._state["agent_0"]["progress"]
        score_1 = self._state["agent_1"]["laps"] + self._state["agent_1"]["progress"]
        if score_0 >= score_1:
            p1_ag, p2_ag = "agent_0", "agent_1"
        else:
            p1_ag, p2_ag = "agent_1", "agent_0"

        def _fmt(sec: float | None) -> str:
            if sec is None:
                return "--:--.---"
            m = int(sec // 60)
            s_rem = sec % 60
            return f"{m:02d}:{s_rem:06.3f}"

        y_card = 92
        for rank_str, ag in (("P1", p1_ag), ("P2", p2_ag)):
            cfg = team_configs[ag]
            st = self._state[ag]
            card_border = cfg["primary"] if ag == "agent_0" else cfg["primary"]

            pygame.draw.rect(surf, (20, 25, 36), (SIDEBAR_X + 16, y_card, SIDEBAR_W - 32, 146), border_radius=8)
            pygame.draw.rect(surf, card_border, (SIDEBAR_X + 16, y_card, SIDEBAR_W - 32, 146), 2, border_radius=8)

            badge_col = (0, 230, 140) if rank_str == "P1" else (220, 225, 235)
            surf.blit(font_stat.render(rank_str, True, badge_col), (SIDEBAR_X + 28, y_card + 8))
            surf.blit(font_body.render(f"{cfg['car_num']}  {cfg['name']}", True, (255, 255, 255)), (SIDEBAR_X + 58, y_card + 9))

            spd = st["speed"]
            surf.blit(font_stat.render(f"Speed: {spd:5.1f} u/s", True, (240, 245, 255)), (SIDEBAR_X + 28, y_card + 32))
            pygame.draw.rect(surf, (32, 40, 56), (SIDEBAR_X + 28, y_card + 52, 280, 6), border_radius=3)
            bar_w = int(min(max(spd, 0.0) / MAX_SPEED, 1.0) * 280)
            if bar_w > 0:
                pygame.draw.rect(surf, cfg["detail"], (SIDEBAR_X + 28, y_card + 52, bar_w, 6), border_radius=3)

            lap_str = f"Lap: {st['laps']}   Progress: {st['progress'] * 100:4.1f}%"
            surf.blit(font_num.render(lap_str, True, (210, 225, 245)), (SIDEBAR_X + 28, y_card + 66))

            last_t = _fmt(st["last_lap_time"])
            best_t = _fmt(st["best_lap_time"])
            surf.blit(font_num.render(f"Last: {last_t}   Best: {best_t}", True, (180, 205, 235)), (SIDEBAR_X + 28, y_card + 88))

            if rank_str == "P1":
                surf.blit(font_badge.render("LEADER", True, (0, 230, 140)), (SIDEBAR_X + 28, y_card + 114))
            else:
                gap_s = st.get("gap_to_leader_seconds", 0.0)
                surf.blit(font_num.render(f"INTERVAL: +{gap_s:5.3f}s", True, (255, 215, 100)), (SIDEBAR_X + 28, y_card + 114))
                if gap_s < 1.0 and gap_s > 0.0:
                    pygame.draw.rect(surf, (15, 60, 30), (SIDEBAR_X + 175, y_card + 110, 115, 20), border_radius=4)
                    surf.blit(font_badge.render("DRS ENABLED (<1s)", True, (80, 245, 140)), (SIDEBAR_X + 183, y_card + 114))
                else:
                    pygame.draw.rect(surf, (35, 42, 52), (SIDEBAR_X + 175, y_card + 110, 115, 20), border_radius=4)
                    surf.blit(font_badge.render("DRS DISABLED", True, (160, 175, 195)), (SIDEBAR_X + 191, y_card + 114))

            y_card += 156

        pygame.draw.rect(surf, (20, 25, 36), (SIDEBAR_X + 16, 404, SIDEBAR_W - 32, 108), border_radius=8)
        pygame.draw.rect(surf, (40, 50, 70), (SIDEBAR_X + 16, 404, SIDEBAR_W - 32, 108), 1, border_radius=8)
        surf.blit(font_body.render("RACE CONTROL & TELEMETRY", True, (180, 205, 235)), (SIDEBAR_X + 28, 414))

        gap_sec_disp = self._state[p2_ag].get("gap_to_leader_seconds", 0.0)
        dist_between = float(np.linalg.norm(self._state[p1_ag]["pos"] - self._state[p2_ag]["pos"]))
        surf.blit(font_num.render(f"Gap (Time):     +{gap_sec_disp:5.3f} s", True, (225, 235, 250)), (SIDEBAR_X + 28, 436))
        surf.blit(font_num.render(f"Gap (Distance):  {dist_between:5.1f} u", True, (225, 235, 250)), (SIDEBAR_X + 28, 456))
        surf.blit(font_num.render(f"Swaps / Moves:   {self._position_swaps}", True, (225, 235, 250)), (SIDEBAR_X + 28, 476))

        pygame.draw.rect(surf, (20, 25, 36), (SIDEBAR_X + 16, 520, SIDEBAR_W - 32, 80), border_radius=8)
        pygame.draw.rect(surf, (40, 50, 70), (SIDEBAR_X + 16, 520, SIDEBAR_W - 32, 80), 1, border_radius=8)
        is_contact = self._state[AGENTS[0]]["prev_colliding"]
        if is_contact:
            pygame.draw.rect(surf, (75, 20, 25), (SIDEBAR_X + 26, 530, SIDEBAR_W - 52, 24), border_radius=5)
            surf.blit(font_badge.render("! STEWARDS: CONTACT DETECTED !", True, (255, 90, 90)), (SIDEBAR_X + 46, 535))
        else:
            pygame.draw.rect(surf, (16, 50, 30), (SIDEBAR_X + 26, 530, SIDEBAR_W - 52, 24), border_radius=5)
            surf.blit(font_badge.render("TRACK CLEAR - GREEN FLAG", True, (80, 235, 140)), (SIDEBAR_X + 66, 535))

        steps_contact = self._state[AGENTS[0]]["steps_in_contact"]
        surf.blit(font_num.render(f"Contact Steps:   {steps_contact} steps", True, (210, 225, 245)), (SIDEBAR_X + 28, 566))

        pygame.draw.rect(surf, (16, 22, 32), (SIDEBAR_X + 16, 608, SIDEBAR_W - 32, 98), border_radius=8)
        surf.blit(font_body.render("CONTROLS & SHORTCUTS:", True, (160, 185, 215)), (SIDEBAR_X + 26, 618))
        surf.blit(font_num.render("ESC / Q : Exit viewer", True, (210, 220, 235)), (SIDEBAR_X + 26, 638))
        surf.blit(font_num.render("R       : Restart race grid", True, (210, 220, 235)), (SIDEBAR_X + 26, 658))
        surf.blit(font_sub.render(f"{W}x{H} DISPLAY | 30 FPS LOCK", True, (130, 150, 175)), (SIDEBAR_X + 26, 678))

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
        self._track_surface = None
        self._pristine_track_surface = None
        self._fonts = None
        self._car_sprites_base = None
        self._car_rot_cache = None

