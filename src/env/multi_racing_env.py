from __future__ import annotations

from typing import Optional

import math
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .track import Track
from .car import (
    CarState,
    DEFAULT_PARAMS,
    DT,
    MAX_SPEED,
    CAR_HALF_WIDTH,
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
DEFAULT_SPAWN_OFFSET_IDX = 15
SPAWN_OFFSET_IDX = DEFAULT_SPAWN_OFFSET_IDX
MAX_OPP_DIST = 300.0
CAR_HALF_LEN = 13.0
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

    if d_long <= 0.0 or d_long > cone_length or d_lat >= 22.0:
        return 0.0

    car_width = 2.0 * CAR_HALF_WIDTH
    if d_lat < car_width and d_long <= min_gap:
        return 0.0

    if d_lat <= 14.0:
        fade_lat = 1.0 - 0.2 * (d_lat / 14.0)
    else:
        fade_lat = 0.8 * (22.0 - d_lat) / 8.0

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
        self._fonts = None

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
        }

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
                s["laps"] += 1

            s["arc_length"] = ts.arc_length

            if ts.on_track:
                arc_delta = s["arc_length"] - prev_state.arc_length
                if arc_delta < -self.track.total_length / 2.0:
                    arc_delta += self.track.total_length
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
        else:
            if race_pos_0 - race_pos_1 >= car_len:
                self._current_leader = AGENTS[0]
                if not recent_respawn:
                    self._position_swaps += 1

        if self.enable_position_reward:
            r_pos_0 = compute_positional_reward(race_pos_0, race_pos_1, self.position_k, self.position_g0)
            r_pos_1 = compute_positional_reward(race_pos_1, race_pos_0, self.position_k, self.position_g0)
            rewards[AGENTS[0]] += r_pos_0
            rewards[AGENTS[1]] += r_pos_1

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

        surf = pygame.Surface((W, H))
        surf.fill((20, 28, 21))
        for y in range(0, H, 36):
            pygame.draw.rect(surf, (24, 34, 25), (0, y, sidebar_x, 18))

        outer = [(int(x), int(y)) for x, y in self.track.outer]
        inner = [(int(x), int(y)) for x, y in self.track.inner]

        for i in range(len(self.track.outer) - 1):
            is_red = (i // 6) % 2 == 0
            c_base = (220, 42, 42) if is_red else (245, 245, 250)
            c_hi = (255, 75, 75) if is_red else (255, 255, 255)
            p0 = self.track.outer[i]
            p1 = self.track.outer[i + 1]
            n0 = self.track.normals[i]
            n1 = self.track.normals[i + 1]
            poly = [p0, p1, p1 - n1 * 6.0, p0 - n0 * 6.0]
            pygame.draw.polygon(surf, c_base, [(int(x), int(y)) for x, y in poly])
            pygame.draw.line(
                surf, c_hi,
                (int(p0[0] - n0[0] * 5.0), int(p0[1] - n0[1] * 5.0)),
                (int(p1[0] - n1[0] * 5.0), int(p1[1] - n1[1] * 5.0)),
                1,
            )

        pygame.draw.polygon(surf, (32, 35, 40), outer)

        cl = self.track.centerline
        for i in range(len(cl) - 1):
            pygame.draw.line(
                surf, (25, 27, 31),
                (int(cl[i, 0]), int(cl[i, 1])),
                (int(cl[i + 1, 0]), int(cl[i + 1, 1])),
                16,
            )

        pygame.draw.polygon(surf, (20, 28, 21), inner)

        for i in range(len(self.track.inner) - 1):
            is_red = (i // 6) % 2 == 0
            c_base = (220, 42, 42) if is_red else (245, 245, 250)
            c_hi = (255, 75, 75) if is_red else (255, 255, 255)
            p0 = self.track.inner[i]
            p1 = self.track.inner[i + 1]
            n0 = self.track.normals[i]
            n1 = self.track.normals[i + 1]
            poly = [p0, p1, p1 + n1 * 6.0, p0 + n0 * 6.0]
            pygame.draw.polygon(surf, c_base, [(int(x), int(y)) for x, y in poly])
            pygame.draw.line(
                surf, c_hi,
                (int(p0[0] + n0[0] * 5.0), int(p0[1] + n0[1] * 5.0)),
                (int(p1[0] + n1[0] * 5.0), int(p1[1] + n1[1] * 5.0)),
                1,
            )

        pygame.draw.lines(surf, (240, 242, 248), True, outer, 2)
        pygame.draw.lines(surf, (240, 242, 248), True, inner, 2)

        for i in range(0, len(cl) - 8, 16):
            pygame.draw.line(
                surf, (68, 76, 88),
                (int(cl[i, 0]), int(cl[i, 1])),
                (int(cl[i + 8, 0]), int(cl[i + 8, 1])),
                1,
            )

        p_out = np.array(outer[0], dtype=float)
        p_in = np.array(inner[0], dtype=float)
        for k in range(10):
            t0 = k / 10.0
            t1 = (k + 1) / 10.0
            pt0 = p_in + (p_out - p_in) * t0
            pt1 = p_in + (p_out - p_in) * t1
            c_chk = (245, 245, 250) if k % 2 == 0 else (25, 25, 25)
            pygame.draw.line(surf, c_chk, (int(pt0[0]), int(pt0[1])), (int(pt1[0]), int(pt1[1])), 5)

        norm0 = self.track.normals[15]
        tang0 = self.track.tangents[15]
        slot1_pos = self.track.centerline[15] - norm0 * 12.0
        slot2_pos = self.track.centerline[15 - 12] + norm0 * 12.0
        for slot in (slot1_pos, slot2_pos):
            f_vec = tang0 * 10.0
            s_vec = norm0 * 6.0
            box_pts = [slot + f_vec + s_vec, slot + f_vec - s_vec, slot - f_vec - s_vec, slot - f_vec + s_vec]
            pygame.draw.polygon(surf, (215, 218, 225), [(int(p[0]), int(p[1])) for p in box_pts], 1)

        self._track_surface = surf

    def _render_frame(self):
        import pygame
        from .track import RAY_ANGLES

        W, H = 1180, 660
        SIDEBAR_X = 850
        SIDEBAR_W = 330

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Slipstream - Multi-Agent Racing")
            else:
                self._screen = pygame.Surface((W, H))
            self._clock = pygame.time.Clock()

        if self._track_surface is None:
            self._bake_track_surface(W, H, SIDEBAR_X)

        if self._fonts is None:
            font_family = "ubuntu" if "ubuntu" in pygame.font.get_fonts() else "dejavusans"
            mono_family = "ubuntumono" if "ubuntumono" in pygame.font.get_fonts() else "monospace"
            self._fonts = {
                "title": pygame.font.SysFont(font_family, 18, bold=True),
                "sub": pygame.font.SysFont(font_family, 11, bold=True),
                "body": pygame.font.SysFont(font_family, 12, bold=True),
                "stat": pygame.font.SysFont(font_family, 13, bold=True),
                "num": pygame.font.SysFont(mono_family, 13, bold=True),
                "badge": pygame.font.SysFont(font_family, 11, bold=True),
            }

        surf = self._screen
        surf.blit(self._track_surface, (0, 0))

        colors = {"agent_0": (0, 210, 255), "agent_1": (255, 95, 55)}
        ray_colors = {"agent_0": (0, 165, 215), "agent_1": (225, 115, 48)}

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
                pygame.draw.circle(surf, ray_colors[agent], (int(end[0]), int(end[1])), 2)

            fwd = np.array([np.cos(heading), np.sin(heading)])
            left = np.array([-fwd[1], fwd[0]])

            for ax in (6.5, -6.5):
                for lat in (4.2, -4.2):
                    t_center = pos + fwd * ax + left * lat
                    t_fl = t_center + fwd * 2.6 + left * 1.2
                    t_fr = t_center + fwd * 2.6 - left * 1.2
                    t_rr = t_center - fwd * 2.6 - left * 1.2
                    t_rl = t_center - fwd * 2.6 + left * 1.2
                    pygame.draw.polygon(surf, (15, 15, 18), [(int(p[0]), int(p[1])) for p in (t_fl, t_fr, t_rr, t_rl)])
                    pygame.draw.line(
                        surf, (140, 140, 150),
                        (int(t_center[0] - fwd[0]), int(t_center[1] - fwd[1])),
                        (int(t_center[0] + fwd[0]), int(t_center[1] + fwd[1])),
                        1,
                    )

            sp_l = pos + fwd * (CAR_HALF_LEN - 0.2) + left * (CAR_HALF_WIDTH - 0.5)
            sp_r = pos + fwd * (CAR_HALF_LEN - 0.2) - left * (CAR_HALF_WIDTH - 0.5)
            pygame.draw.line(surf, (20, 20, 22), (int(sp_l[0]), int(sp_l[1])), (int(sp_r[0]), int(sp_r[1])), 2)

            nose_tip_l = pos + fwd * CAR_HALF_LEN + left * (CAR_HALF_WIDTH * 0.55)
            nose_tip_r = pos + fwd * CAR_HALF_LEN - left * (CAR_HALF_WIDTH * 0.55)
            front_arch_l = pos + fwd * (CAR_HALF_LEN * 0.65) + left * CAR_HALF_WIDTH
            front_arch_r = pos + fwd * (CAR_HALF_LEN * 0.65) - left * CAR_HALF_WIDTH
            side_pod_l = pos - fwd * (CAR_HALF_LEN * 0.1) + left * (CAR_HALF_WIDTH * 0.95)
            side_pod_r = pos - fwd * (CAR_HALF_LEN * 0.1) - left * (CAR_HALF_WIDTH * 0.95)
            rear_arch_l = pos - fwd * (CAR_HALF_LEN * 0.65) + left * CAR_HALF_WIDTH
            rear_arch_r = pos - fwd * (CAR_HALF_LEN * 0.65) - left * CAR_HALF_WIDTH
            tail_l = pos - fwd * CAR_HALF_LEN + left * (CAR_HALF_WIDTH * 0.85)
            tail_r = pos - fwd * CAR_HALF_LEN - left * (CAR_HALF_WIDTH * 0.85)

            body = [nose_tip_l, front_arch_l, side_pod_l, rear_arch_l, tail_l, tail_r, rear_arch_r, side_pod_r, front_arch_r, nose_tip_r]
            pygame.draw.polygon(surf, color, [(int(p[0]), int(p[1])) for p in body])
            pygame.draw.polygon(surf, (15, 18, 22), [(int(p[0]), int(p[1])) for p in body], 1)

            stripe_f = pos + fwd * (CAR_HALF_LEN * 0.8)
            stripe_r = pos - fwd * (CAR_HALF_LEN * 0.85)
            pygame.draw.line(surf, (20, 24, 30), (int(stripe_f[0]), int(stripe_f[1])), (int(stripe_r[0]), int(stripe_r[1])), 2)

            cp_f = pos + fwd * 3.5
            cp_fl = pos + fwd * 2.8 + left * 2.6
            cp_fr = pos + fwd * 2.8 - left * 2.6
            cp_rl = pos - fwd * 3.8 + left * 2.8
            cp_rr = pos - fwd * 3.8 - left * 2.8
            pygame.draw.polygon(surf, (18, 25, 35), [(int(p[0]), int(p[1])) for p in (cp_f, cp_fl, cp_rl, cp_rr, cp_fr)])
            pygame.draw.polygon(surf, (90, 160, 220), [(int(p[0]), int(p[1])) for p in (cp_f, cp_fl, cp_rl, cp_rr, cp_fr)], 1)

            helmet_pos = pos - fwd * 0.5
            pygame.draw.circle(surf, (240, 240, 80), (int(helmet_pos[0]), int(helmet_pos[1])), 2)

            scoop_pos = pos - fwd * 3.5
            pygame.draw.circle(surf, (12, 14, 18), (int(scoop_pos[0]), int(scoop_pos[1])), 1)

            wing_l = pos - fwd * (CAR_HALF_LEN - 0.4) + left * (CAR_HALF_WIDTH - 0.2)
            wing_r = pos - fwd * (CAR_HALF_LEN - 0.4) - left * (CAR_HALF_WIDTH - 0.2)
            pygame.draw.line(surf, (18, 20, 24), (int(wing_l[0]), int(wing_l[1])), (int(wing_r[0]), int(wing_r[1])), 2)
            for ep in (wing_l, wing_r):
                pygame.draw.line(
                    surf, color,
                    (int(ep[0] - fwd[0] * 1.5), int(ep[1] - fwd[1] * 1.5)),
                    (int(ep[0] + fwd[0] * 1.5), int(ep[1] + fwd[1] * 1.5)),
                    2,
                )

            pygame.draw.circle(surf, (255, 255, 230), (int(nose_tip_l[0]), int(nose_tip_l[1])), 2)
            pygame.draw.circle(surf, (255, 255, 230), (int(nose_tip_r[0]), int(nose_tip_r[1])), 2)
            pygame.draw.circle(surf, (255, 30, 30), (int(tail_l[0]), int(tail_l[1])), 2)
            pygame.draw.circle(surf, (255, 30, 30), (int(tail_r[0]), int(tail_r[1])), 2)

            if self._state[AGENTS[0]]["prev_colliding"]:
                fl = pos + fwd * CAR_HALF_LEN + left * CAR_HALF_WIDTH
                fr = pos + fwd * CAR_HALF_LEN - left * CAR_HALF_WIDTH
                rr = pos - fwd * CAR_HALF_LEN - left * CAR_HALF_WIDTH
                rl = pos - fwd * CAR_HALF_LEN + left * CAR_HALF_WIDTH
                pygame.draw.polygon(
                    surf, (255, 220, 40),
                    [(int(p[0]), int(p[1])) for p in (fl, fr, rr, rl)],
                    2,
                )

        pygame.draw.rect(surf, (14, 18, 24), (SIDEBAR_X, 0, SIDEBAR_W, H))
        pygame.draw.line(surf, (36, 46, 62), (SIDEBAR_X, 0), (SIDEBAR_X, H), 2)

        font_title = self._fonts["title"]
        font_sub = self._fonts["sub"]
        font_body = self._fonts["body"]
        font_stat = self._fonts["stat"]
        font_num = self._fonts["num"]
        font_badge = self._fonts["badge"]

        surf.blit(font_title.render("SLIPSTREAM", True, (255, 255, 255)), (SIDEBAR_X + 18, 16))
        surf.blit(font_sub.render("PHASE 4 GRAND PRIX TELEMETRY", True, (130, 150, 180)), (SIDEBAR_X + 18, 40))

        pygame.draw.rect(surf, (22, 30, 42), (SIDEBAR_X + 18, 62, 140, 24), border_radius=5)
        surf.blit(font_num.render(f"STEP: {self._step_count:04d}", True, (180, 215, 255)), (SIDEBAR_X + 26, 66))
        pygame.draw.rect(surf, (22, 30, 42), (SIDEBAR_X + 168, 62, 140, 24), border_radius=5)
        surf.blit(font_num.render(f"TIME: {self._step_count * 0.05:5.1f}s", True, (180, 215, 255)), (SIDEBAR_X + 176, 66))

        score_0 = self._state["agent_0"]["laps"] + self._state["agent_0"]["progress"]
        score_1 = self._state["agent_1"]["laps"] + self._state["agent_1"]["progress"]
        rank_0 = "P1 LEADER" if score_0 >= score_1 else "P2 CHASER"
        rank_1 = "P1 LEADER" if score_1 > score_0 else "P2 CHASER"
        dist_between = float(np.linalg.norm(self._state["agent_0"]["pos"] - self._state["agent_1"]["pos"]))

        pygame.draw.rect(surf, (20, 26, 36), (SIDEBAR_X + 18, 96, SIDEBAR_W - 36, 142), border_radius=8)
        pygame.draw.rect(surf, (0, 190, 235), (SIDEBAR_X + 18, 96, SIDEBAR_W - 36, 142), 2, border_radius=8)
        pygame.draw.circle(surf, (0, 215, 255), (SIDEBAR_X + 34, 114), 5)
        surf.blit(font_body.render("AGENT 0 (CYAN)", True, (255, 255, 255)), (SIDEBAR_X + 46, 107))
        rank_0_col = (0, 230, 140) if "P1" in rank_0 else (200, 210, 225)
        surf.blit(font_badge.render(rank_0, True, rank_0_col), (SIDEBAR_X + 205, 107))

        speed_0 = self._state["agent_0"]["speed"]
        surf.blit(font_stat.render(f"Speed:  {speed_0:5.1f} u/s", True, (240, 245, 255)), (SIDEBAR_X + 30, 132))
        pygame.draw.rect(surf, (32, 40, 54), (SIDEBAR_X + 30, 152, 234, 7), border_radius=3)
        bar_w_0 = int(min(max(speed_0, 0.0) / MAX_SPEED, 1.0) * 234)
        if bar_w_0 > 0:
            pygame.draw.rect(surf, (0, 210, 255), (SIDEBAR_X + 30, 152, bar_w_0, 7), border_radius=3)
        surf.blit(font_num.render(f"Laps: {self._state['agent_0']['laps']}   Progress: {self._state['agent_0']['progress']*100:4.1f}%", True, (210, 225, 245)), (SIDEBAR_X + 30, 168))
        surf.blit(font_num.render(f"Collisions: {self._state['agent_0']['collision_count']}   Respawns: {self._state['agent_0']['respawns']}", True, (210, 225, 245)), (SIDEBAR_X + 30, 190))
        rsp_str_0 = "INCIDENT - RESPAWNED" if self._state["agent_0"]["respawn"] else "CLEAR - NO INCIDENT"
        rsp_col_0 = (255, 90, 80) if self._state["agent_0"]["respawn"] else (120, 200, 150)
        surf.blit(font_stat.render(f"Status: {rsp_str_0}", True, rsp_col_0), (SIDEBAR_X + 30, 212))

        pygame.draw.rect(surf, (20, 26, 36), (SIDEBAR_X + 18, 248, SIDEBAR_W - 36, 142), border_radius=8)
        pygame.draw.rect(surf, (255, 95, 55), (SIDEBAR_X + 18, 248, SIDEBAR_W - 36, 142), 2, border_radius=8)
        pygame.draw.circle(surf, (255, 95, 55), (SIDEBAR_X + 34, 266), 5)
        surf.blit(font_body.render("AGENT 1 (CORAL)", True, (255, 255, 255)), (SIDEBAR_X + 46, 259))
        rank_1_col = (0, 230, 140) if "P1" in rank_1 else (200, 210, 225)
        surf.blit(font_badge.render(rank_1, True, rank_1_col), (SIDEBAR_X + 205, 259))

        speed_1 = self._state["agent_1"]["speed"]
        surf.blit(font_stat.render(f"Speed:  {speed_1:5.1f} u/s", True, (240, 245, 255)), (SIDEBAR_X + 30, 284))
        pygame.draw.rect(surf, (32, 40, 54), (SIDEBAR_X + 30, 304, 234, 7), border_radius=3)
        bar_w_1 = int(min(max(speed_1, 0.0) / MAX_SPEED, 1.0) * 234)
        if bar_w_1 > 0:
            pygame.draw.rect(surf, (255, 95, 55), (SIDEBAR_X + 30, 304, bar_w_1, 7), border_radius=3)
        surf.blit(font_num.render(f"Laps: {self._state['agent_1']['laps']}   Progress: {self._state['agent_1']['progress']*100:4.1f}%", True, (210, 225, 245)), (SIDEBAR_X + 30, 320))
        surf.blit(font_num.render(f"Collisions: {self._state['agent_1']['collision_count']}   Respawns: {self._state['agent_1']['respawns']}", True, (210, 225, 245)), (SIDEBAR_X + 30, 342))
        rsp_str_1 = "INCIDENT - RESPAWNED" if self._state["agent_1"]["respawn"] else "CLEAR - NO INCIDENT"
        rsp_col_1 = (255, 90, 80) if self._state["agent_1"]["respawn"] else (120, 200, 150)
        surf.blit(font_stat.render(f"Status: {rsp_str_1}", True, rsp_col_1), (SIDEBAR_X + 30, 364))

        pygame.draw.rect(surf, (20, 26, 36), (SIDEBAR_X + 18, 400, SIDEBAR_W - 36, 126), border_radius=8)
        pygame.draw.rect(surf, (40, 52, 70), (SIDEBAR_X + 18, 400, SIDEBAR_W - 36, 126), 1, border_radius=8)
        is_contact = self._state[AGENTS[0]]["prev_colliding"]
        if is_contact:
            pygame.draw.rect(surf, (75, 20, 25), (SIDEBAR_X + 28, 412, 238, 26), border_radius=5)
            surf.blit(font_badge.render("! CONTACT DETECTED !", True, (255, 80, 80)), (SIDEBAR_X + 64, 417))
        else:
            pygame.draw.rect(surf, (16, 50, 30), (SIDEBAR_X + 28, 412, 238, 26), border_radius=5)
            surf.blit(font_badge.render("CLEAN RACING - NO CONTACT", True, (80, 230, 140)), (SIDEBAR_X + 54, 417))

        surf.blit(font_num.render(f"Car Gap:        {dist_between:5.1f} units", True, (220, 230, 245)), (SIDEBAR_X + 30, 448))
        steps_contact = self._state[AGENTS[0]]["steps_in_contact"]
        surf.blit(font_num.render(f"Contact Steps:  {steps_contact} steps", True, (220, 230, 245)), (SIDEBAR_X + 30, 470))
        ratio_str = f"{self.track.half_width * 2.0 / (CAR_HALF_WIDTH * 2.0):.2f} : 1"
        surf.blit(font_num.render(f"Track Ratio:    {ratio_str}", True, (150, 180, 215)), (SIDEBAR_X + 30, 492))

        pygame.draw.rect(surf, (16, 22, 30), (SIDEBAR_X + 18, 536, SIDEBAR_W - 36, 108), border_radius=8)
        surf.blit(font_body.render("CONTROLS & SHORTCUTS:", True, (160, 185, 215)), (SIDEBAR_X + 28, 548))
        surf.blit(font_num.render("ESC / Q : Exit viewer", True, (210, 220, 235)), (SIDEBAR_X + 28, 570))
        surf.blit(font_num.render("R       : Restart race grid", True, (210, 220, 235)), (SIDEBAR_X + 28, 592))
        surf.blit(font_sub.render("FPS: 30 LOCK | 1180x660 VIEWPORT", True, (130, 150, 175)), (SIDEBAR_X + 28, 614))

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
        self._fonts = None

