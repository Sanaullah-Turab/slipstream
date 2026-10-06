from __future__ import annotations

from typing import Optional

import math
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .track import Track
from .car import CarState, DEFAULT_PARAMS, DT, MAX_SPEED, CAR_HALF_WIDTH, step_physics
from .racing_env import MAX_STEPS, MAX_HEADING_RATE, MAX_RAY_DIST
from .rewards import compute_reward, AgentState, DEFAULT_CONTACT_PENALTY, DEFAULT_CONTACT_STEP_PENALTY
from .collision import obb_overlap, resolve_collision, classify_contact

AGENTS = ["agent_0", "agent_1"]
DEFAULT_SPAWN_OFFSET_IDX = 15
SPAWN_OFFSET_IDX = DEFAULT_SPAWN_OFFSET_IDX
MAX_OPP_DIST = 300.0
CAR_HALF_LEN = 13.0
LATERAL_HISTORY_LEN = 20


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
        enable_draft: bool = False,
        contact_penalty: float = DEFAULT_CONTACT_PENALTY,
        contact_step_penalty: float = DEFAULT_CONTACT_STEP_PENALTY,
        legacy_collision: bool = False,
        spawn_offset_idx: int = DEFAULT_SPAWN_OFFSET_IDX,
    ) -> None:
        super().__init__()
        self.track = Track()
        self.render_mode = render_mode
        self.enable_draft = enable_draft
        self.contact_penalty = contact_penalty
        self.contact_step_penalty = contact_step_penalty
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

        # Ego-frame opponent velocity components (dims 15-16)
        opp_vel_world = opp["speed"] * np.array([math.cos(opp["heading"]), math.sin(opp["heading"])])
        c, ss = math.cos(s["heading"]), math.sin(s["heading"])
        opp_vel_ego_fwd = float(np.clip((c * opp_vel_world[0] + ss * opp_vel_world[1]) / MAX_SPEED, -1.0, 1.0))
        opp_vel_ego_lat = float(np.clip((-ss * opp_vel_world[0] + c * opp_vel_world[1]) / MAX_SPEED, -1.0, 1.0))

        # Relative heading (dim 17)
        rel_heading = opp["heading"] - s["heading"]
        opp_rel_heading = float(np.clip(math.sin(rel_heading), -1.0, 1.0))

        # Draft flag continuous [0,1] (dim 18) - stub for 4.1, activated in 4.2
        draft_flag = 0.0

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
                draft_flag,
                draft_flag,
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
        }

    def step(self, actions: dict[str, np.ndarray]):
        rewards = {}
        terminations = {a: False for a in self.agents}
        truncations = {}
        infos = {}

        for agent in self.agents:
            s = self._state[agent]
            s["respawn"] = False
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
            overlapping, normal, _ = obb_overlap(
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

                    fault = classify_contact(
                        normal=normal,
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

            for s in (s0, s1):
                s["prev_colliding"] = overlapping

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

        surf = self._screen
        surf.fill((18, 26, 20))

        outer = [(int(x), int(y)) for x, y in self.track.outer]
        inner = [(int(x), int(y)) for x, y in self.track.inner]

        for i in range(len(self.track.outer) - 1):
            c = (215, 45, 45) if (i // 8) % 2 == 0 else (240, 242, 246)
            p0 = self.track.outer[i]
            p1 = self.track.outer[i + 1]
            n0 = self.track.normals[i]
            n1 = self.track.normals[i + 1]
            poly = [p0, p1, p1 - n1 * 5.0, p0 - n0 * 5.0]
            pygame.draw.polygon(surf, c, [(int(p[0]), int(p[1])) for p in poly])

        pygame.draw.polygon(surf, (34, 37, 43), outer)
        pygame.draw.polygon(surf, (18, 26, 20), inner)

        for i in range(len(self.track.inner) - 1):
            c = (215, 45, 45) if (i // 8) % 2 == 0 else (240, 242, 246)
            p0 = self.track.inner[i]
            p1 = self.track.inner[i + 1]
            n0 = self.track.normals[i]
            n1 = self.track.normals[i + 1]
            poly = [p0, p1, p1 + n1 * 5.0, p0 + n0 * 5.0]
            pygame.draw.polygon(surf, c, [(int(p[0]), int(p[1])) for p in poly])

        pygame.draw.lines(surf, (235, 238, 244), True, outer, 2)
        pygame.draw.lines(surf, (235, 238, 244), True, inner, 2)

        cl = self.track.centerline
        for i in range(0, len(cl) - 8, 16):
            pygame.draw.line(
                surf, (72, 80, 92),
                (int(cl[i, 0]), int(cl[i, 1])),
                (int(cl[i + 8, 0]), int(cl[i + 8, 1])),
                1,
            )

        p_out = np.array(outer[0], dtype=float)
        p_in = np.array(inner[0], dtype=float)
        for k in range(8):
            t0 = k / 8.0
            t1 = (k + 1) / 8.0
            pt0 = p_in + (p_out - p_in) * t0
            pt1 = p_in + (p_out - p_in) * t1
            c_chk = (240, 240, 245) if k % 2 == 0 else (30, 30, 30)
            pygame.draw.line(surf, c_chk, (int(pt0[0]), int(pt0[1])), (int(pt1[0]), int(pt1[1])), 4)

        colors = {"agent_0": (0, 204, 255), "agent_1": (255, 94, 58)}
        ray_colors = {"agent_0": (0, 160, 210), "agent_1": (220, 110, 45)}

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

            for ax in (10.0, -10.0):
                for lat in (6.8, -6.8):
                    t_center = pos + fwd * ax + left * lat
                    t_fl = t_center + fwd * 3.5 + left * 1.8
                    t_fr = t_center + fwd * 3.5 - left * 1.8
                    t_rr = t_center - fwd * 3.5 - left * 1.8
                    t_rl = t_center - fwd * 3.5 + left * 1.8
                    pygame.draw.polygon(
                        surf, (16, 16, 18),
                        [(int(p[0]), int(p[1])) for p in (t_fl, t_fr, t_rr, t_rl)],
                    )

            p_nose_l = pos + fwd * CAR_HALF_LEN + left * (CAR_HALF_WIDTH * 0.7)
            p_nose_r = pos + fwd * CAR_HALF_LEN - left * (CAR_HALF_WIDTH * 0.7)
            p_fwd_l = pos + fwd * (CAR_HALF_LEN * 0.75) + left * CAR_HALF_WIDTH
            p_fwd_r = pos + fwd * (CAR_HALF_LEN * 0.75) - left * CAR_HALF_WIDTH
            rl = pos - fwd * CAR_HALF_LEN + left * CAR_HALF_WIDTH
            rr = pos - fwd * CAR_HALF_LEN - left * CAR_HALF_WIDTH
            body_pts = [p_nose_l, p_fwd_l, rl, rr, p_fwd_r, p_nose_r]

            pygame.draw.polygon(surf, color, [(int(p[0]), int(p[1])) for p in body_pts])
            pygame.draw.polygon(surf, (20, 20, 24), [(int(p[0]), int(p[1])) for p in body_pts], 1)

            cp_fl = pos + fwd * 4.5 + left * 4.5
            cp_fr = pos + fwd * 4.5 - left * 4.5
            cp_rr = pos - fwd * 6.5 - left * 4.5
            cp_rl = pos - fwd * 6.5 + left * 4.5
            pygame.draw.polygon(
                surf, (22, 30, 40),
                [(int(p[0]), int(p[1])) for p in (cp_fl, cp_fr, cp_rr, cp_rl)],
            )
            pygame.draw.polygon(
                surf, (100, 160, 210),
                [(int(p[0]), int(p[1])) for p in (cp_fl, cp_fr, cp_rr, cp_rl)],
                1,
            )

            w_l = pos - fwd * (CAR_HALF_LEN - 1.0) + left * (CAR_HALF_WIDTH - 0.5)
            w_r = pos - fwd * (CAR_HALF_LEN - 1.0) - left * (CAR_HALF_WIDTH - 0.5)
            pygame.draw.line(surf, (25, 25, 30), (int(w_l[0]), int(w_l[1])), (int(w_r[0]), int(w_r[1])), 3)

            sp_l = pos + fwd * (CAR_HALF_LEN - 0.5) + left * (CAR_HALF_WIDTH - 1.0)
            sp_r = pos + fwd * (CAR_HALF_LEN - 0.5) - left * (CAR_HALF_WIDTH - 1.0)
            pygame.draw.line(surf, (25, 25, 30), (int(sp_l[0]), int(sp_l[1])), (int(sp_r[0]), int(sp_r[1])), 2)

            pygame.draw.circle(surf, (255, 255, 230), (int(p_nose_l[0]), int(p_nose_l[1])), 2)
            pygame.draw.circle(surf, (255, 255, 230), (int(p_nose_r[0]), int(p_nose_r[1])), 2)
            pygame.draw.circle(surf, (255, 30, 30), (int(rl[0]), int(rl[1])), 2)
            pygame.draw.circle(surf, (255, 30, 30), (int(rr[0]), int(rr[1])), 2)

            if self._state[AGENTS[0]]["prev_colliding"]:
                fl = pos + fwd * CAR_HALF_LEN + left * CAR_HALF_WIDTH
                fr = pos + fwd * CAR_HALF_LEN - left * CAR_HALF_WIDTH
                pygame.draw.polygon(
                    surf, (255, 220, 40),
                    [(int(p[0]), int(p[1])) for p in (fl, fr, rr, rl)],
                    2,
                )

        pygame.draw.rect(surf, (15, 20, 28), (SIDEBAR_X, 0, SIDEBAR_W, H))
        pygame.draw.line(surf, (38, 48, 64), (SIDEBAR_X, 0), (SIDEBAR_X, H), 2)

        font_title = pygame.font.SysFont("dejavusans", 16, bold=True)
        font_sub = pygame.font.SysFont("dejavusans", 10)
        font_bold = pygame.font.SysFont("dejavusans", 12, bold=True)
        font_mono = pygame.font.SysFont("monospace", 11)
        font_badge = pygame.font.SysFont("dejavusans", 10, bold=True)

        surf.blit(font_title.render("SLIPSTREAM", True, (240, 245, 252)), (SIDEBAR_X + 16, 14))
        surf.blit(font_sub.render("PHASE 4 COMPETITIVE RACING", True, (120, 140, 165)), (SIDEBAR_X + 16, 36))

        pygame.draw.rect(surf, (24, 32, 45), (SIDEBAR_X + 16, 54, 140, 20), border_radius=4)
        surf.blit(font_mono.render(f"STEP: {self._step_count:04d}", True, (160, 200, 240)), (SIDEBAR_X + 22, 57))
        pygame.draw.rect(surf, (24, 32, 45), (SIDEBAR_X + 166, 54, 138, 20), border_radius=4)
        surf.blit(font_mono.render(f"TIME: {self._step_count * 0.05:5.1f}s", True, (160, 200, 240)), (SIDEBAR_X + 172, 57))

        score_0 = self._state["agent_0"]["laps"] + self._state["agent_0"]["progress"]
        score_1 = self._state["agent_1"]["laps"] + self._state["agent_1"]["progress"]
        rank_0 = "P1 LEADER" if score_0 >= score_1 else "P2 CHASER"
        rank_1 = "P1 LEADER" if score_1 > score_0 else "P2 CHASER"
        dist_between = float(np.linalg.norm(self._state["agent_0"]["pos"] - self._state["agent_1"]["pos"]))

        pygame.draw.rect(surf, (22, 28, 38), (SIDEBAR_X + 16, 86, SIDEBAR_W - 32, 148), border_radius=8)
        pygame.draw.rect(surf, (0, 180, 230), (SIDEBAR_X + 16, 86, SIDEBAR_W - 32, 148), 1, border_radius=8)
        pygame.draw.circle(surf, (0, 210, 255), (SIDEBAR_X + 32, 104), 5)
        surf.blit(font_bold.render("AGENT 0 (CYAN)", True, (240, 245, 252)), (SIDEBAR_X + 44, 97))
        rank_0_col = (0, 220, 140) if "P1" in rank_0 else (180, 190, 205)
        surf.blit(font_badge.render(rank_0, True, rank_0_col), (SIDEBAR_X + 195, 98))

        speed_0 = self._state["agent_0"]["speed"]
        surf.blit(font_mono.render(f"Speed: {speed_0:5.1f} u/s", True, (200, 210, 225)), (SIDEBAR_X + 28, 122))
        pygame.draw.rect(surf, (36, 44, 58), (SIDEBAR_X + 28, 138, 240, 6), border_radius=3)
        bar_w_0 = int(min(max(speed_0, 0.0) / MAX_SPEED, 1.0) * 240)
        if bar_w_0 > 0:
            pygame.draw.rect(surf, (0, 210, 255), (SIDEBAR_X + 28, 138, bar_w_0, 6), border_radius=3)
        surf.blit(font_mono.render(f"Laps: {self._state['agent_0']['laps']}  Prog: {self._state['agent_0']['progress']*100:4.1f}%", True, (185, 195, 210)), (SIDEBAR_X + 28, 150))
        surf.blit(font_mono.render(f"Collisions: {self._state['agent_0']['collision_count']}  Respawns: {self._state['agent_0']['respawns']}", True, (185, 195, 210)), (SIDEBAR_X + 28, 172))
        rsp_str_0 = "YES" if self._state["agent_0"]["respawn"] else "NO"
        surf.blit(font_mono.render(f"Recent Rsp: {rsp_str_0}", True, (140, 155, 175)), (SIDEBAR_X + 28, 194))

        pygame.draw.rect(surf, (22, 28, 38), (SIDEBAR_X + 16, 246, SIDEBAR_W - 32, 148), border_radius=8)
        pygame.draw.rect(surf, (255, 95, 55), (SIDEBAR_X + 16, 246, SIDEBAR_W - 32, 148), 1, border_radius=8)
        pygame.draw.circle(surf, (255, 95, 55), (SIDEBAR_X + 32, 264), 5)
        surf.blit(font_bold.render("AGENT 1 (CORAL)", True, (240, 245, 252)), (SIDEBAR_X + 44, 257))
        rank_1_col = (0, 220, 140) if "P1" in rank_1 else (180, 190, 205)
        surf.blit(font_badge.render(rank_1, True, rank_1_col), (SIDEBAR_X + 195, 258))

        speed_1 = self._state["agent_1"]["speed"]
        surf.blit(font_mono.render(f"Speed: {speed_1:5.1f} u/s", True, (200, 210, 225)), (SIDEBAR_X + 28, 282))
        pygame.draw.rect(surf, (36, 44, 58), (SIDEBAR_X + 28, 298, 240, 6), border_radius=3)
        bar_w_1 = int(min(max(speed_1, 0.0) / MAX_SPEED, 1.0) * 240)
        if bar_w_1 > 0:
            pygame.draw.rect(surf, (255, 95, 55), (SIDEBAR_X + 28, 298, bar_w_1, 6), border_radius=3)
        surf.blit(font_mono.render(f"Laps: {self._state['agent_1']['laps']}  Prog: {self._state['agent_1']['progress']*100:4.1f}%", True, (185, 195, 210)), (SIDEBAR_X + 28, 310))
        surf.blit(font_mono.render(f"Collisions: {self._state['agent_1']['collision_count']}  Respawns: {self._state['agent_1']['respawns']}", True, (185, 195, 210)), (SIDEBAR_X + 28, 332))
        rsp_str_1 = "YES" if self._state["agent_1"]["respawn"] else "NO"
        surf.blit(font_mono.render(f"Recent Rsp: {rsp_str_1}", True, (140, 155, 175)), (SIDEBAR_X + 28, 354))

        pygame.draw.rect(surf, (22, 28, 38), (SIDEBAR_X + 16, 406, SIDEBAR_W - 32, 128), border_radius=8)
        pygame.draw.rect(surf, (40, 50, 68), (SIDEBAR_X + 16, 406, SIDEBAR_W - 32, 128), 1, border_radius=8)
        is_contact = self._state[AGENTS[0]]["prev_colliding"]
        if is_contact:
            pygame.draw.rect(surf, (75, 20, 25), (SIDEBAR_X + 26, 416, 246, 24), border_radius=4)
            surf.blit(font_badge.render("! CONTACT DETECTED !", True, (255, 80, 80)), (SIDEBAR_X + 66, 421))
        else:
            pygame.draw.rect(surf, (18, 44, 28), (SIDEBAR_X + 26, 416, 246, 24), border_radius=4)
            surf.blit(font_badge.render("CLEAN RACING", True, (75, 220, 130)), (SIDEBAR_X + 96, 421))
        surf.blit(font_mono.render(f"Separation:    {dist_between:5.1f} units", True, (185, 195, 210)), (SIDEBAR_X + 28, 452))
        steps_contact = self._state[AGENTS[0]]["steps_in_contact"]
        surf.blit(font_mono.render(f"Contact Steps: {steps_contact}", True, (185, 195, 210)), (SIDEBAR_X + 28, 474))
        f_fol = self._state[AGENTS[0]]["fault_log"]["follower"] + self._state[AGENTS[1]]["fault_log"]["follower"]
        f_lead = self._state[AGENTS[0]]["fault_log"]["leader"] + self._state[AGENTS[1]]["fault_log"]["leader"]
        f_neut = max(self._state[AGENTS[0]]["fault_log"]["neutral"], self._state[AGENTS[1]]["fault_log"]["neutral"])
        surf.blit(font_mono.render(f"Faults: F:{f_fol} L:{f_lead} N:{f_neut}", True, (185, 195, 210)), (SIDEBAR_X + 28, 496))

        pygame.draw.rect(surf, (18, 24, 32), (SIDEBAR_X + 16, 546, SIDEBAR_W - 32, 98), border_radius=8)
        surf.blit(font_mono.render("CONTROLS:", True, (140, 160, 185)), (SIDEBAR_X + 28, 558))
        surf.blit(font_mono.render("ESC / Q : Exit viewer", True, (185, 195, 210)), (SIDEBAR_X + 28, 578))
        surf.blit(font_mono.render("R       : Reset episode", True, (185, 195, 210)), (SIDEBAR_X + 28, 598))
        surf.blit(font_mono.render("Target FPS: 30", True, (120, 140, 160)), (SIDEBAR_X + 28, 618))

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

