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

        if hasattr(self.track, "get_starting_grid") and getattr(self.track, "circuit", "") == "shanghai":
            grid = self.track.get_starting_grid()
            for slot_key in ("agent_0", "agent_1"):
                pos_g, hdg_g = grid[slot_key]
                f_vec = np.array([np.cos(hdg_g), np.sin(hdg_g)]) * CAR_HALF_LEN
                s_vec = np.array([-np.sin(hdg_g), np.cos(hdg_g)]) * CAR_HALF_WIDTH
                box_pts = [pos_g + f_vec + s_vec, pos_g + f_vec - s_vec, pos_g - f_vec - s_vec, pos_g - f_vec + s_vec]
                pygame.draw.polygon(surf, (220, 225, 235), [(int(p[0]), int(p[1])) for p in box_pts], 1)
                pygame.draw.line(surf, (240, 245, 255), (int((pos_g + f_vec - s_vec)[0]), int((pos_g + f_vec - s_vec)[1])), (int((pos_g + f_vec + s_vec)[0]), int((pos_g + f_vec + s_vec)[1])), 2)
        else:
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

        W, H = 1280, 720
        SIDEBAR_X = 940
        SIDEBAR_W = 340

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
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
                "car_num": "#16 LEC",
                "primary": (220, 20, 35),
                "accent": (20, 20, 24),
                "detail": (255, 220, 0),
                "ray": (220, 50, 40),
            },
        }

        for agent in AGENTS:
            s = self._state[agent]
            pos, heading = s["pos"], s["heading"]
            cfg = team_configs[agent]

            rays = self.track.ray_distances(pos, heading, MAX_RAY_DIST)
            for a, dist in zip(RAY_ANGLES, rays):
                angle = heading + a
                end = pos + dist * np.array([np.cos(angle), np.sin(angle)])
                pygame.draw.line(
                    surf, cfg["ray"],
                    (int(pos[0]), int(pos[1])),
                    (int(end[0]), int(end[1])),
                    1,
                )
                pygame.draw.circle(surf, cfg["ray"], (int(end[0]), int(end[1])), 2)

            fwd = np.array([np.cos(heading), np.sin(heading)])
            left = np.array([-fwd[1], fwd[0]])

            for ax, lat in ((6.5, 4.3), (6.5, -4.3), (-6.5, 4.3), (-6.5, -4.3)):
                hub = pos + fwd * ax + left * lat
                chassis_pt = pos + fwd * ax + left * (lat * 0.45)
                pygame.draw.line(surf, (40, 44, 52), (int(chassis_pt[0]), int(chassis_pt[1])), (int(hub[0]), int(hub[1])), 2)

            for ax, lat, w_len, w_wid in ((6.5, 4.3, 5.2, 2.4), (6.5, -4.3, 5.2, 2.4), (-6.5, 4.3, 5.8, 3.0), (-6.5, -4.3, 5.8, 3.0)):
                t_center = pos + fwd * ax + left * lat
                t_fl = t_center + fwd * (w_len * 0.5) + left * (w_wid * 0.5)
                t_fr = t_center + fwd * (w_len * 0.5) - left * (w_wid * 0.5)
                t_rr = t_center - fwd * (w_len * 0.5) - left * (w_wid * 0.5)
                t_rl = t_center - fwd * (w_len * 0.5) + left * (w_wid * 0.5)
                pygame.draw.polygon(surf, (22, 22, 26), [(int(p[0]), int(p[1])) for p in (t_fl, t_fr, t_rr, t_rl)])
                pygame.draw.polygon(surf, (10, 10, 12), [(int(p[0]), int(p[1])) for p in (t_fl, t_fr, t_rr, t_rl)], 1)
                pygame.draw.circle(surf, (190, 195, 205), (int(t_center[0]), int(t_center[1])), 1)

            fw_c = pos + fwd * (CAR_HALF_LEN - 0.5)
            fw_l = fw_c + left * CAR_HALF_WIDTH
            fw_r = fw_c - left * CAR_HALF_WIDTH
            pygame.draw.line(surf, (22, 24, 28), (int(fw_l[0]), int(fw_l[1])), (int(fw_r[0]), int(fw_r[1])), 3)
            pygame.draw.line(surf, cfg["accent"], (int(fw_l[0]), int(fw_l[1])), (int(fw_r[0]), int(fw_r[1])), 1)
            for endpt in (fw_l, fw_r):
                ep_f = endpt + fwd * 2.2
                ep_r = endpt - fwd * 1.5
                pygame.draw.line(surf, cfg["primary"], (int(ep_f[0]), int(ep_f[1])), (int(ep_r[0]), int(ep_r[1])), 2)

            rw_c = pos - fwd * (CAR_HALF_LEN - 0.5)
            rw_l = rw_c + left * (CAR_HALF_WIDTH - 0.6)
            rw_r = rw_c - left * (CAR_HALF_WIDTH - 0.6)
            pygame.draw.line(surf, (20, 22, 26), (int(rw_l[0]), int(rw_l[1])), (int(rw_r[0]), int(rw_r[1])), 3)
            pygame.draw.line(surf, cfg["accent"], (int(rw_l[0]), int(rw_l[1])), (int(rw_r[0]), int(rw_r[1])), 1)
            for endpt in (rw_l, rw_r):
                ep_f = endpt + fwd * 1.8
                ep_r = endpt - fwd * 2.2
                pygame.draw.line(surf, cfg["primary"], (int(ep_f[0]), int(ep_f[1])), (int(ep_r[0]), int(ep_r[1])), 2)

            chassis_pts = [
                pos + fwd * CAR_HALF_LEN + left * 0.9,
                pos + fwd * 4.0 + left * 1.6,
                pos + fwd * 1.5 + left * 3.5,
                pos - fwd * 4.5 + left * 3.2,
                pos - fwd * 7.5 + left * 1.8,
                pos - fwd * (CAR_HALF_LEN - 0.8) + left * 1.2,
                pos - fwd * (CAR_HALF_LEN - 0.8) - left * 1.2,
                pos - fwd * 7.5 - left * 1.8,
                pos - fwd * 4.5 - left * 3.2,
                pos + fwd * 1.5 - left * 3.5,
                pos + fwd * 4.0 - left * 1.6,
                pos + fwd * CAR_HALF_LEN - left * 0.9,
            ]
            pygame.draw.polygon(surf, cfg["primary"], [(int(p[0]), int(p[1])) for p in chassis_pts])
            pygame.draw.polygon(surf, (15, 18, 22), [(int(p[0]), int(p[1])) for p in chassis_pts], 1)

            nose_top = pos + fwd * (CAR_HALF_LEN - 0.5)
            pygame.draw.circle(surf, cfg["detail"], (int(nose_top[0]), int(nose_top[1])), 2)

            stripe_f = pos + fwd * 4.5
            stripe_r = pos - fwd * 6.5
            pygame.draw.line(surf, cfg["accent"], (int(stripe_f[0]), int(stripe_f[1])), (int(stripe_r[0]), int(stripe_r[1])), 2)

            cockpit_f = pos + fwd * 1.5
            cockpit_r = pos - fwd * 2.2
            cockpit_pts = [
                cockpit_f,
                pos + fwd * 0.4 + left * 1.1,
                cockpit_r + left * 0.9,
                cockpit_r - left * 0.9,
                pos + fwd * 0.4 - left * 1.1,
            ]
            pygame.draw.polygon(surf, (14, 16, 20), [(int(p[0]), int(p[1])) for p in cockpit_pts])
            pygame.draw.polygon(surf, (45, 52, 65), [(int(p[0]), int(p[1])) for p in cockpit_pts], 1)

            helmet_pos = pos - fwd * 0.6
            pygame.draw.circle(surf, cfg["detail"], (int(helmet_pos[0]), int(helmet_pos[1])), 2)

            halo_center = pos + fwd * 0.3
            pygame.draw.circle(surf, (30, 35, 42), (int(halo_center[0]), int(halo_center[1])), 3, 1)

            tail_light = pos - fwd * (CAR_HALF_LEN - 0.5)
            pygame.draw.circle(surf, (255, 30, 30), (int(tail_light[0]), int(tail_light[1])), 2)

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
        surf.blit(font_sub.render("1280x720 DISPLAY | 30 FPS LOCK", True, (130, 150, 175)), (SIDEBAR_X + 26, 678))

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

