from __future__ import annotations

from typing import Optional
import collections
import math
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .track import Track, SHANGHAI_TRACK_WIDTH
from .car import (
    CarState,
    DEFAULT_PARAMS,
    SHANGHAI_PARAMS,
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
_STREAK_OFFSETS = ((0.12, -0.35), (0.28, 0.40), (0.45, -0.15), (0.60, 0.30), (0.75, -0.25), (0.88, 0.15))
_TRAIL_SHADES = {
    "agent_0": [
        (int(28 + (40 - 28) * t), int(30 + (130 - 30) * t), int(35 + (240 - 35) * t))
        for t in np.linspace(0.08, 0.85, 20)
    ],
    "agent_1": [
        (int(28 + (220 - 28) * t), int(30 + (25 - 30) * t), int(35 + (40 - 35) * t))
        for t in np.linspace(0.08, 0.85, 20)
    ],
}


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
        fault_penalty: float = 3.5,
        continuous: bool = False,
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
        self.fault_penalty = fault_penalty
        self.continuous = continuous or (render_mode == "human")
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
        self._car_trails = {a: collections.deque(maxlen=20) for a in AGENTS}
        self._prev_rear_axle_s = {a: None for a in AGENTS}
        self._static_hud_surface = None
        self._text_cache = {}
        from src.viewer.interpolator import StateInterpolator
        self._interpolator = StateInterpolator(step_subdivisions=3)
        self.view_w = self.window_w
        self.view_h = self.window_h - int(round(self.window_h * 0.18))
        from src.viewer.camera import FollowCamera
        from src.viewer.tile_cache import TileCache
        self._camera = FollowCamera()
        self._tile_cache = TileCache(self.track)
        self._hud = None
        self._effects = None
        self._camera_overview = False
        self._prev_rear_axle_world = {a: None for a in AGENTS}
        self.car_params = SHANGHAI_PARAMS if getattr(self.track, "circuit", "") == "shanghai" else DEFAULT_PARAMS

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

    def _build_car_sprite(self, agent: str, zoom: float | None = None):
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

        scale = zoom if zoom is not None else (self._scale if self._scale is not None else 1.0)
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

    def _get_car_sprite(self, agent: str, heading: float, zoom: float | None = None):
        import pygame
        z = round(float(zoom if zoom is not None else (self._scale if self._scale is not None else 1.0)), 2)
        if self._car_rot_cache is None:
            self._car_rot_cache = {}
        if z not in self._car_rot_cache:
            self._car_rot_cache[z] = {
                "agent_0": {},
                "agent_1": {},
                "base_0": self._build_car_sprite("agent_0", z),
                "base_1": self._build_car_sprite("agent_1", z),
            }

        deg = int(round(math.degrees(heading) / 3.0)) * 3 % 360
        cache = self._car_rot_cache[z][agent]
        if deg not in cache:
            base_surf = self._car_rot_cache[z]["base_0" if agent == "agent_0" else "base_1"]
            cache[deg] = pygame.transform.rotate(base_surf, -deg)
        return cache[deg]

    def _render_text(self, text: str, font_name: str, color: tuple[int, int, int]):
        key = (text, font_name, color)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 2000:
                self._text_cache.clear()
            surf = self._fonts[font_name].render(text, True, color)
            self._text_cache[key] = surf
        return surf

    def _build_static_hud(self, W: int, H: int, sidebar_w: int):
        import pygame
        surf = pygame.Surface((sidebar_w, H))
        surf.fill((14, 17, 24))
        pygame.draw.line(surf, (36, 44, 60), (0, 0), (0, H), 2)

        pygame.draw.rect(surf, (220, 20, 35), (16, 14, 34, 22), border_radius=4)
        surf.blit(self._fonts["f1"].render("F1", True, (255, 255, 255)), (24, 16))
        surf.blit(self._fonts["title"].render("GRAND PRIX", True, (255, 255, 255)), (58, 14))
        surf.blit(self._fonts["sub"].render("SHANGHAI INTERNATIONAL CIRCUIT", True, (135, 155, 185)), (58, 34))

        for s_idx, (s_label, s_color) in enumerate((("S1", (220, 40, 40)), ("S2", (40, 200, 220)), ("S3", (240, 210, 40))), start=1):
            bx = 224 + (s_idx - 1) * 26
            pygame.draw.rect(surf, s_color, (bx, 17, 24, 16), border_radius=2)
            t_col = (10, 10, 15) if s_idx != 1 else (255, 255, 255)
            surf.blit(self._fonts["badge"].render(s_label, True, t_col), (bx + 4, 18))

        pygame.draw.rect(surf, (20, 25, 36), (16, 404, sidebar_w - 32, 108), border_radius=8)
        pygame.draw.rect(surf, (40, 50, 70), (16, 404, sidebar_w - 32, 108), 1, border_radius=8)
        surf.blit(self._fonts["body"].render("RACE CONTROL & TELEMETRY", True, (180, 205, 235)), (28, 414))

        pygame.draw.rect(surf, (20, 25, 36), (16, 520, sidebar_w - 32, 80), border_radius=8)
        pygame.draw.rect(surf, (40, 50, 70), (16, 520, sidebar_w - 32, 80), 1, border_radius=8)

        pygame.draw.rect(surf, (16, 22, 32), (16, 608, sidebar_w - 32, 98), border_radius=8)
        surf.blit(self._fonts["body"].render("CONTROLS & SHORTCUTS:", True, (160, 185, 215)), (26, 618))
        surf.blit(self._fonts["num"].render("ESC / Q : Exit viewer", True, (210, 220, 235)), (26, 638))
        surf.blit(self._fonts["num"].render("R       : Restart race grid", True, (210, 220, 235)), (26, 658))
        surf.blit(self._fonts["sub"].render(f"{W}x{H} DISPLAY | 30 FPS LOCK", True, (130, 150, 175)), (26, 678))

        try:
            return surf.convert()
        except Exception:
            return surf

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
        self._prev_rear_axle_s = {a: None for a in AGENTS}
        if getattr(self, "_car_trails", None) is not None:
            for a in AGENTS:
                self._car_trails[a].clear()

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
                    "steer": 0.0,
                    "prev_steer": 0.0,
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
                    "steer": 0.0,
                    "prev_steer": 0.0,
                }

        self._position_swaps = 0
        self._last_respawn_step = {a: -9999 for a in AGENTS}
        race_pos_0 = self._state[AGENTS[0]]["start_offset"]
        race_pos_1 = self._state[AGENTS[1]]["start_offset"]
        self._current_leader = AGENTS[0] if race_pos_0 >= race_pos_1 else AGENTS[1]

        if getattr(self, "_interpolator", None) is not None:
            self._interpolator.reset(self._state)

        if getattr(self, "_camera", None) is not None:
            spawn_pos = (self._state[AGENTS[0]]["pos"] + self._state[AGENTS[1]]["pos"]) * 0.5
            self._camera.snap_to(spawn_pos)
            if getattr(self, "_tile_cache", None) is not None:
                self._tile_cache.reset_rubber()
                self._tile_cache.prebake_spawn_ring(spawn_pos, (self.view_w, self.view_h))
        if getattr(self, "_effects", None) is not None:
            self._effects.overtake_frames = 0
            self._effects.contact_frames = 0
        self._prev_rear_axle_world = {a: None for a in AGENTS}

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

        fwd_lead = np.array([math.cos(lead_s["heading"]), math.sin(lead_s["heading"])])
        lat_lead = np.array([-fwd_lead[1], fwd_lead[0]])
        delta = foll_s["pos"] - lead_s["pos"]
        d_long = -float(np.dot(delta, fwd_lead))
        d_lat = abs(float(np.dot(delta, lat_lead)))
        rel_lat_vec = float(np.dot(delta, lat_lead))

        lead_idx = self.track.nearest_idx(lead_s["pos"])
        lookahead_idx = (lead_idx + 80) % len(self.track.centerline)
        curv_ahead = float(self.track.curvatures[lookahead_idx])
        lead_curv = getattr(lead_ts, "curvature", 0.0)

        car_width = 2.0 * CAR_HALF_WIDTH

        if 0.0 < d_long < 40.0 and gap_sec < 1.2:
            if d_lat < car_width and d_long < 22.0:
                rewards[foll_ag] -= 0.12 * max(0.0, 1.0 - d_long / 22.0)
            elif d_lat >= car_width and d_lat <= 20.0:
                rewards[foll_ag] += 0.15 * min((d_lat - car_width) / 5.0, 1.0)

        if 0.0 < d_long < 45.0 and gap_sec < 1.5:
            eff_curv = lead_curv if abs(lead_curv) > 0.001 else curv_ahead
            if abs(eff_curv) > 0.001:
                inside_dir = 1.0 if eff_curv > 0.0 else -1.0
                if lead_s["lateral"] * inside_dir > 1.5:
                    rewards[lead_ag] += 0.10 * min(abs(lead_s["lateral"]) / (self.track.half_width * 0.5), 1.0)
            elif abs(rel_lat_vec) > car_width * 0.8:
                attack_dir = 1.0 if rel_lat_vec > 0.0 else -1.0
                if lead_s["lateral"] * attack_dir > 1.0:
                    rewards[lead_ag] += 0.08 * min(abs(lead_s["lateral"]) / (self.track.half_width * 0.4), 1.0)

        for a in AGENTS:
            st = self._state[a]
            ts = self.track.get_track_state(st["pos"])
            c_val = getattr(ts, "curvature", 0.0)
            if abs(c_val) > 0.001:
                in_dir = 1.0 if c_val > 0.0 else -1.0
                if st["lateral"] * in_dir > 1.0:
                    rewards[a] += 0.04 * min(st["lateral"] * in_dir / 10.0, 1.0)
            else:
                if not (a == foll_ag and 0.0 < d_long < 35.0 and gap_sec < 1.2):
                    rewards[a] += 0.03 * max(0.0, 1.0 - abs(st["lateral"]) / 8.0)

            cur_st = float(st.get("steer", 0.0))
            prv_st = float(st.get("prev_steer", 0.0))
            rewards[a] -= 0.02 * ((cur_st - prv_st) ** 2)

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
            steer_cmd = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip(action[1], -1.0, 1.0))
            draft_int = draft_ints[agent]
            s["draft_intensity"] = draft_int

            prev_steer = float(s.get("steer", 0.0))
            if not self.legacy_collision:
                max_steer_rate = 8.0
                d_steer = float(np.clip(steer_cmd - prev_steer, -max_steer_rate * DT, max_steer_rate * DT))
                steer = float(prev_steer + d_steer)
            else:
                steer = steer_cmd
            s["prev_steer"] = prev_steer
            s["steer"] = steer

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
                car, throttle, steer, getattr(self, "car_params", DEFAULT_PARAMS), DT, draft_intensity=draft_int
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
                        lateral_move_threshold=0.10 * (2.0 * self.track.half_width),
                    )
                    if fault == "follower_fault":
                        follower_s["fault_log"]["follower"] += 1
                        rewards[follower_agent] -= self.fault_penalty
                    elif fault == "leader_fault":
                        leader_s["fault_log"]["leader"] += 1
                        rewards[leader_agent] -= self.fault_penalty
                    else:
                        leader_s["fault_log"]["neutral"] += 1
                        follower_s["fault_log"]["neutral"] += 1

                # Momentum transfer via impulse resolution
                vel0 = s0["speed"] * np.array([math.cos(s0["heading"]), math.sin(s0["heading"])])
                vel1 = s1["speed"] * np.array([math.cos(s1["heading"]), math.sin(s1["heading"])])
                vel0_new, vel1_new = resolve_collision(s0["pos"], vel0, s1["pos"], vel1, normal)
                max_v = getattr(self, "car_params", DEFAULT_PARAMS).max_speed
                s0["speed"] = float(np.clip(np.linalg.norm(vel0_new), 0.0, max_v))
                s1["speed"] = float(np.clip(np.linalg.norm(vel1_new), 0.0, max_v))

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

        if self.enable_position_reward:
            tac_rewards = self._compute_tactical_rewards()
            for a in AGENTS:
                rewards[a] += tac_rewards[a]

        truncated = False if self.continuous else (self._step_count >= MAX_STEPS)

        for agent in self.agents:
            truncations[agent] = truncated
            infos[agent] = self._build_info(agent)

        if getattr(self, "_interpolator", None) is not None:
            self._interpolator.on_sim_step(self._state)

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
        s["steer"] = 0.0
        s["prev_steer"] = 0.0

    def render(self):
        frame = self._render_frame()
        if self.render_mode == "rgb_array":
            return frame

    def _bake_track_surface(self, W: int, H: int, sidebar_x: int) -> None:
        import pygame
        from src.viewer.world import draw_static_world

        if self._scale is None:
            self._compute_transform()

        origin_wx = -self._offset[0] / self._scale
        origin_wy = -self._offset[1] / self._scale
        max_wx = origin_wx + W / self._scale
        max_wy = origin_wy + H / self._scale

        surf_2x = pygame.Surface((W * 2, H * 2))
        draw_static_world(surf_2x, self.track, (origin_wx, origin_wy, max_wx, max_wy), self._scale * 2.0)

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

        if self._hud is None:
            from src.viewer.broadcast_hud import BroadcastHUD
            self._hud = BroadcastHUD(self.track, (W, H))

        if self._track_surface is None:
            self._bake_track_surface(W, H, SIDEBAR_X)

        if self._fonts is None:
            self._fonts = self._hud.fonts

        if self._car_trails is None:
            self._car_trails = {a: collections.deque(maxlen=20) for a in AGENTS}

        interp_states = {}
        for a in AGENTS:
            if getattr(self, "_interpolator", None) is not None and a in getattr(self._interpolator, "curr_states", {}):
                interp_states[a] = self._interpolator.get_interpolated_state(a)
            else:
                interp_states[a] = self._state[a]

        surf = self._screen
        view_w = W
        view_h = self._hud.view_h

        if not self._camera_overview:
            current_zoom = self._tile_cache.zoom
            leader_is_ham = self._current_leader == "agent_1"
            pos_ham = interp_states["agent_1"]["pos"]
            pos_ver = interp_states["agent_0"]["pos"]
            vel_ham = np.array([math.cos(interp_states["agent_1"]["heading"]), math.sin(interp_states["agent_1"]["heading"])]) * interp_states["agent_1"]["speed"]
            vel_ver = np.array([math.cos(interp_states["agent_0"]["heading"]), math.sin(interp_states["agent_0"]["heading"])]) * interp_states["agent_0"]["speed"]
            self._camera.update(pos_ham, vel_ham, pos_ver, vel_ver, leader_is_ham, (view_w, view_h), current_zoom, DT)
            leader_s = self._state[self._current_leader]["arc_length"]
            self._tile_cache.update_frame(self._camera.pos, (view_w, view_h), leader_s)
            self._tile_cache.render_tiles(surf, self._camera.pos, (view_w, view_h))

            def to_screen(pt: np.ndarray) -> np.ndarray:
                return self._camera.world_to_screen(pt, (view_w, view_h), current_zoom)
        else:
            current_zoom = self._scale
            surf.blit(self._track_surface, (0, 0))

            def to_screen(pt: np.ndarray) -> np.ndarray:
                return self._world_to_screen(pt)

        for agent in AGENTS:
            ist = interp_states[agent]
            pos, heading = ist["pos"], ist["heading"]
            fwd = np.array([math.cos(heading), math.sin(heading)])
            rear_axle = pos - fwd * (WHEELBASE * 0.5)
            prev_axle = self._prev_rear_axle_world.get(agent)
            if prev_axle is not None and not np.array_equal(prev_axle, rear_axle):
                self._tile_cache.add_rubber_segment(prev_axle, rear_axle)
            self._prev_rear_axle_world[agent] = rear_axle.copy()

        for agent in AGENTS:
            pos_s = to_screen(interp_states[agent]["pos"])
            pt = (int(round(pos_s[0])), int(round(pos_s[1])))
            trail = self._car_trails[agent]
            trail.append(pt)
            shades = _TRAIL_SHADES[agent]
            n = len(trail)
            if n >= 2:
                for i in range(n - 1):
                    pygame.draw.line(surf, shades[i + (20 - n)], trail[i], trail[i + 1], max(1, int(round(current_zoom * 0.5))))

        draft_0 = self._state["agent_0"].get("draft_intensity", 0.0)
        draft_1 = self._state["agent_1"].get("draft_intensity", 0.0)
        if draft_0 > 0.0 or draft_1 > 0.0:
            lead_ag = "agent_1" if draft_0 > draft_1 else "agent_0"
            lead_s = interp_states[lead_ag]
            l_pos, l_head = lead_s["pos"], lead_s["heading"]
            fwd = np.array([math.cos(l_head), math.sin(l_head)])
            lat_vec = np.array([-fwd[1], fwd[0]])
            rear_c = l_pos - fwd * CAR_HALF_LEN
            cone_len = DRAFT_CONE_LENGTH
            cone_hw = cone_len * math.tan(DRAFT_CONE_HALF_ANGLE)
            cone_end = rear_c - fwd * cone_len
            end_l = cone_end + lat_vec * cone_hw
            end_r = cone_end - lat_vec * cone_hw
            s_rear = to_screen(rear_c)
            s_el = to_screen(end_l)
            s_er = to_screen(end_r)
            pygame.draw.aaline(surf, (60, 180, 200), (int(s_rear[0]), int(s_rear[1])), (int(s_el[0]), int(s_el[1])))
            pygame.draw.aaline(surf, (60, 180, 200), (int(s_rear[0]), int(s_rear[1])), (int(s_er[0]), int(s_er[1])))
            phase = (self._step_count * 0.06) % 1.0
            s_len = 0.08 * cone_len
            for u_base, v_base in _STREAK_OFFSETS:
                u = (u_base + phase) % 1.0
                v = v_base * u
                pt1 = rear_c - fwd * (u * cone_len) + lat_vec * (v * cone_hw)
                pt2 = pt1 - fwd * s_len
                sp1 = to_screen(pt1)
                sp2 = to_screen(pt2)
                pygame.draw.aaline(surf, (60, 180, 200), (int(sp1[0]), int(sp1[1])), (int(sp2[0]), int(sp2[1])))

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

        for agent in AGENTS:
            ist = interp_states[agent]
            pos, heading = ist["pos"], ist["heading"]
            pos_s = to_screen(pos)
            cfg = team_configs[agent]


            car_surf = self._get_car_sprite(agent, heading, current_zoom)
            rect = car_surf.get_rect(center=(int(pos_s[0]), int(pos_s[1])))
            surf.blit(car_surf, rect)

            if self._state[AGENTS[0]]["prev_colliding"]:
                c_hlen = CAR_HALF_LEN * current_zoom
                c_hwid = CAR_HALF_WIDTH * current_zoom
                fwd = np.array([np.cos(heading), np.sin(heading)])
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

        if self._effects is None:
            from src.viewer.effects import ViewerEffects
            self._effects = ViewerEffects((W, H))

        pt_ham_s = to_screen(interp_states["agent_1"]["pos"])
        pt_ver_s = to_screen(interp_states["agent_0"]["pos"])
        is_contact = self._state[AGENTS[0]]["prev_colliding"]
        contact_pt_s = (int(round((pt_ham_s[0] + pt_ver_s[0]) * 0.5)), int(round((pt_ham_s[1] + pt_ver_s[1]) * 0.5)))
        self._effects.on_step(self._current_leader, is_contact, contact_pt_s)

        self._effects.draw_driver_tags(surf, pt_ham_s, pt_ver_s)
        dist_ham = float(np.linalg.norm(interp_states["agent_1"]["pos"] - self._camera.pos))
        dist_ver = float(np.linalg.norm(interp_states["agent_0"]["pos"] - self._camera.pos))
        self._effects.draw_off_screen_arrows(surf, pt_ham_s, dist_ham, (220, 20, 35))
        self._effects.draw_off_screen_arrows(surf, pt_ver_s, dist_ver, (30, 140, 255))
        self._effects.draw_banners_and_effects(surf)
        self._hud.draw_bottom_bar(
            surf,
            self._state["agent_1"],
            self._state["agent_0"],
            self._current_leader,
            self._step_count,
            DT,
            self._position_swaps,
            is_contact,
        )
        self._hud.draw_top_chips(
            surf,
            self._state[self._current_leader]["laps"],
            self._state[self._current_leader]["progress"],
            is_contact,
        )
        self._hud.draw_minimap(
            surf,
            self._state["agent_1"]["pos"],
            self._state["agent_0"]["pos"],
        )

        if self._hud.show_help:
            surf.blit(self._hud.help_overlay, (self._hud.help_x, self._hud.help_y))

        if getattr(self, "_interpolator", None) is not None:
            self._interpolator.advance_frame()

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            assert self._clock is not None
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    def set_camera_mode(self, mode: str) -> None:
        if self._camera is not None:
            self._camera.mode = mode

    def toggle_camera_overview(self) -> bool:
        self._camera_overview = not self._camera_overview
        return self._camera_overview

    def zoom_in(self) -> float:
        if self._tile_cache is not None:
            return self._tile_cache.zoom_in()
        return 3.2

    def zoom_out(self) -> float:
        if self._tile_cache is not None:
            return self._tile_cache.zoom_out()
        return 3.2

    def close(self) -> None:
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
        self._track_surface = None
        self._pristine_track_surface = None
        self._fonts = None
        self._hud = None
        self._effects = None
        self._car_sprites_base = None
        self._car_rot_cache = None
        self._car_trails = None
        self._prev_rear_axle_s = None
        self._prev_rear_axle_world = {a: None for a in AGENTS}
        self._static_hud_surface = None
        self._text_cache.clear()

