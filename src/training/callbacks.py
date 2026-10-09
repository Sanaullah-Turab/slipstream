from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path
import numpy as np

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.env.car import DT
from src.env.multi_racing_env import (
    MultiRacingEnv as _MultiRacingEnv,
    AGENTS as _MULTI_AGENTS,
    aggregate_fault_counts as _aggregate_fault_counts,
)
from src.env.racing_env import RacingEnv


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, keep_last: int | None = None):
        super().__init__()
        self.save_freq = save_freq
        self.save_dir = Path(save_dir)
        self.keep_last = keep_last
        self._saved: list[tuple[Path, int]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir / f"model_{self.num_timesteps}"
            self.model.save(path)

            meta = {
                "timesteps": self.num_timesteps,
                "wandb_run_id": wandb.run.id if wandb.run else None,
                "git_sha": _git_sha(),
            }
            (self.save_dir / f"meta_{self.num_timesteps}.json").write_text(json.dumps(meta))

            self._saved.append((path, self.num_timesteps))
            if self.keep_last is not None and len(self._saved) > self.keep_last:
                old_path, old_ts = self._saved.pop(0)
                old_path.with_suffix(".zip").unlink(missing_ok=True)
                (self.save_dir / f"meta_{old_ts}.json").unlink(missing_ok=True)
        return True


class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_freq: int, n_episodes: int, save_best_path: str | None = None,
                 eval_seed_base: int = 1000):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self._save_best_path = Path(save_best_path) if save_best_path else None
        self._eval_seed_base = eval_seed_base
        self._best: tuple[float, float, float] = (-1.0, float("inf"), -float("inf"))

    def _on_training_start(self) -> None:
        self._eval_env = Monitor(RacingEnv())

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            rewards, laps, lengths, lat_ratios = [], [], [], []
            all_lap_times: list[float] = []
            for ep_idx in range(self.n_episodes):
                obs, _ = self._eval_env.reset(seed=self._eval_seed_base + ep_idx)
                done = False
                ep_reward, ep_len, last_info = 0.0, 0, {}
                ep_lat: list[float] = []
                ep_laps_curr, ep_lap_step_start = 0, 0
                ep_lap_times: list[float] = []
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self._eval_env.step(action)
                    ep_reward += float(reward)
                    ep_len += 1
                    last_info = info
                    if "lateral_ratio" in info:
                        ep_lat.append(float(info["lateral_ratio"]))
                    curr_laps = info.get("laps", 0)
                    if curr_laps > ep_laps_curr:
                        ep_lap_times.append((ep_len - ep_lap_step_start) * DT)
                        ep_lap_step_start = ep_len
                        ep_laps_curr = curr_laps
                    done = terminated or truncated
                rewards.append(ep_reward)
                laps.append(last_info.get("laps", 0))
                lengths.append(ep_len)
                if ep_lat:
                    lat_ratios.append(sum(ep_lat) / len(ep_lat))
                all_lap_times.extend(ep_lap_times)

            logs = {
                "eval/mean_reward": sum(rewards) / self.n_episodes,
                "eval/mean_laps": sum(laps) / self.n_episodes,
                "eval/mean_ep_len": sum(lengths) / self.n_episodes,
            }
            if lat_ratios:
                logs["eval/mean_lateral_ratio"] = sum(lat_ratios) / len(lat_ratios)

            name_to_val = self.logger.name_to_value
            for key in (
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "rollout/ep_rew_mean",
                "rollout/ep_len_mean",
            ):
                if key in name_to_val:
                    logs[key] = name_to_val[key]

            mean_laps = logs["eval/mean_laps"]
            mean_reward = logs["eval/mean_reward"]
            mean_lap_time = sum(all_lap_times) / len(all_lap_times) if all_lap_times else None
            if mean_lap_time is not None:
                logs["eval/mean_lap_time"] = mean_lap_time

            wandb.log(logs, step=self.num_timesteps)

            if self._save_best_path:
                lap_time_key = mean_lap_time if mean_lap_time is not None else float("inf")
                score = (mean_laps, -lap_time_key, mean_reward)
                if score > self._best:
                    self._best = score
                    self.model.save(self._save_best_path / "best")
                    (self._save_best_path / "best.json").write_text(
                        json.dumps({
                            "step": self.num_timesteps,
                            "mean_laps": mean_laps,
                            "mean_lap_time": mean_lap_time,
                            "mean_reward": mean_reward,
                        }, indent=2)
                    )
        return True

    def _on_training_end(self) -> None:
        self._eval_env.close()


def batch_predict(model, obs_dict: dict, agents: list[str], deterministic: bool = True) -> dict[str, np.ndarray]:
    obs_batch = np.stack([obs_dict[a] for a in agents])
    actions_batch, _ = model.predict(obs_batch, deterministic=deterministic)
    return {a: actions_batch[i] for i, a in enumerate(agents)}

class MultiEvalCallback(BaseCallback):
    GAP_EPS_FRAC = 0.005

    def __init__(self, eval_freq: int, n_episodes: int, env_kwargs: dict | None = None):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.env_kwargs = env_kwargs or {}

    def _on_training_start(self) -> None:
        self._eval_env = _MultiRacingEnv(**self.env_kwargs)
        self._track_length = self._eval_env.track.total_length
        self._gap_eps = self.GAP_EPS_FRAC * self._track_length

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        leader_laps, follower_laps = [], []
        total_collisions, total_contact_steps = [], []
        total_respawns, progress_gaps = [], []
        leader_paces, follower_paces, pair_paces, leader_speeds = [], [], [], []
        position_swaps = []
        faults_follower, faults_leader, faults_neutral = [], [], []
        global_col_crashes, global_solo_crashes = 0, 0
        total_steps = 0

        for ep in range(self.n_episodes):
            obs_dict, _ = self._eval_env.reset(seed=2000 + ep)
            done = False
            ep_infos: dict = {a: {} for a in _MULTI_AGENTS}
            prev_respawns = {a: 0 for a in _MULTI_AGENTS}
            steps_since_collision = {a: 9999 for a in _MULTI_AGENTS}

            while not done:
                total_steps += 1
                actions = batch_predict(self.model, obs_dict, list(_MULTI_AGENTS), deterministic=True)
                obs_dict, _, _, trunc_dict, info_dict = self._eval_env.step(actions)
                ep_infos = info_dict
                done = any(trunc_dict.values())

                for a in _MULTI_AGENTS:
                    if info_dict[a].get("collision", False):
                        steps_since_collision[a] = 0
                    else:
                        steps_since_collision[a] += 1

                    current_respawns = info_dict[a]["respawns"]
                    if current_respawns > prev_respawns[a]:
                        if steps_since_collision[a] < 30:
                            global_col_crashes += 1
                        else:
                            global_solo_crashes += 1
                    prev_respawns[a] = current_respawns

            race_pos = {a: ep_infos[a]["cumulative_distance"] + ep_infos[a].get("start_offset", 0.0) for a in _MULTI_AGENTS}
            gap = abs(race_pos[_MULTI_AGENTS[0]] - race_pos[_MULTI_AGENTS[1]])
            progress_gaps.append(gap / self._track_length)

            if gap < self._gap_eps:
                leader, follower = _MULTI_AGENTS[0], _MULTI_AGENTS[1]
            else:
                leader = max(_MULTI_AGENTS, key=lambda a: race_pos[a])
                follower = _MULTI_AGENTS[1] if leader == _MULTI_AGENTS[0] else _MULTI_AGENTS[0]

            leader_laps.append(ep_infos[leader]["laps"])
            follower_laps.append(ep_infos[follower]["laps"])
            total_collisions.append(ep_infos[_MULTI_AGENTS[0]]["collision_count"])
            total_contact_steps.append(ep_infos[_MULTI_AGENTS[0]].get("steps_in_contact", 0))
            position_swaps.append(ep_infos[_MULTI_AGENTS[0]].get("position_swaps", 0))

            fault_counts = _aggregate_fault_counts(ep_infos, _MULTI_AGENTS)
            faults_follower.append(fault_counts["follower"])
            faults_leader.append(fault_counts["leader"])
            faults_neutral.append(fault_counts["neutral"])

            total_respawns.append(
                ep_infos[_MULTI_AGENTS[0]]["respawns"] + ep_infos[_MULTI_AGENTS[1]]["respawns"]
            )

            leader_dist_laps = race_pos[leader] / self._track_length
            ep_pace = (leader_dist_laps / 2000.0) * 1000.0
            follower_dist_laps = race_pos[follower] / self._track_length
            ep_follower_pace = (follower_dist_laps / 2000.0) * 1000.0
            ep_pair_pace = (ep_pace + ep_follower_pace) / 2.0
            leader_paces.append(ep_pace)
            follower_paces.append(ep_follower_pace)
            pair_paces.append(ep_pair_pace)
            leader_speeds.append(ep_pace * self._track_length / 1000.0)

        total_agent_steps = max(1, total_steps * 2)
        col_crash_rate = (global_col_crashes / total_agent_steps) * 1000.0
        solo_crash_rate = (global_solo_crashes / total_agent_steps) * 1000.0

        logs = {
            "eval/leader/mean_laps": sum(leader_laps) / self.n_episodes,
            "eval/leader/std_laps": statistics.stdev(leader_laps) if len(leader_laps) > 1 else 0.0,
            "eval/follower/mean_laps": sum(follower_laps) / self.n_episodes,
            "eval/follower/std_laps": statistics.stdev(follower_laps) if len(follower_laps) > 1 else 0.0,
            "eval/mean_collisions_per_ep": sum(total_collisions) / self.n_episodes,
            "eval/mean_steps_in_contact_per_ep": sum(total_contact_steps) / self.n_episodes,
            "eval/mean_position_swaps_per_ep": sum(position_swaps) / self.n_episodes,
            "eval/pair_pace": sum(pair_paces) / self.n_episodes,
            "eval/faults/follower_per_ep": sum(faults_follower) / self.n_episodes,
            "eval/faults/leader_per_ep": sum(faults_leader) / self.n_episodes,
            "eval/faults/neutral_per_ep": sum(faults_neutral) / self.n_episodes,
            "eval/collision_crash_rate_per_1k": col_crash_rate,
            "eval/solo_crash_rate_per_1k": solo_crash_rate,
            "eval/mean_respawns_per_ep": sum(total_respawns) / self.n_episodes,
            "eval/mean_progress_gap": sum(progress_gaps) / self.n_episodes,
            "eval/leader/mean_pace_dist": sum(leader_paces) / self.n_episodes,
            "eval/follower/mean_pace_dist": sum(follower_paces) / self.n_episodes,
            "eval/leader/mean_speed": sum(leader_speeds) / self.n_episodes,
        }
        wandb.log(logs, step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self._eval_env.close()

class TrainCrashCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollout_respawns = 0
        self.rollout_steps = 0

    def _on_rollout_start(self) -> None:
        self.rollout_respawns = 0
        self.rollout_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for i, done in enumerate(dones):
            self.rollout_steps += 1
            if done and i < len(infos):
                # When done, we grab the final cumulative respawns for that episode
                # Wait, SB3 resets the env BEFORE returning info? 
                # vec_multi.py puts the terminal info in `infos[i]`. So `infos[i]["respawns"]` is correct.
                self.rollout_respawns += infos[i].get("respawns", 0)
        return True

    def _on_rollout_end(self) -> None:
        if self.rollout_steps > 0:
            rate = (self.rollout_respawns / self.rollout_steps) * 1000.0
            self.logger.record("train/respawns_per_1000_steps", rate)


class TrainingEpisodeCallback(BaseCallback):
    def __init__(self, log_path: Path | str | None = None):
        super().__init__()
        self.log_path = Path(log_path) if log_path else None
        self.episode_records: list[dict] = []
        self._cur_step = 0
        self._last_respawn_step = -9999
        self._in_contact = False
        self._events_near_respawn = 0

    def _on_step(self) -> bool:
        self._cur_step += 1
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if len(infos) >= 2:
            info0 = infos[0]
            info1 = infos[1]

            if info0.get("respawn", False) or info1.get("respawn", False):
                self._last_respawn_step = self._cur_step

            contact = info0.get("collision", False)
            if contact and not self._in_contact:
                if (self._cur_step - self._last_respawn_step) <= 30:
                    self._events_near_respawn += 1
            self._in_contact = contact

            is_done = dones.any() if hasattr(dones, "any") else any(dones)
            if is_done:
                col_count = info0.get("collision_count", 0)
                steps_contact = info0.get("steps_in_contact", 0)
                step_pen = steps_contact * (-0.02)
                event_pen = col_count * (-0.1)
                respawns = info0.get("respawns", 0) + info1.get("respawns", 0)

                rec = {
                    "episode": len(self.episode_records) + 1,
                    "global_step": self.num_timesteps,
                    "collision_count": col_count,
                    "steps_in_contact": steps_contact,
                    "step_contact_penalty": step_pen,
                    "event_contact_penalty": event_pen,
                    "total_contact_penalty": step_pen + event_pen,
                    "respawns": respawns,
                    "events_near_respawn": self._events_near_respawn,
                }
                self.episode_records.append(rec)

                if wandb.run is not None:
                    wandb.log({
                        "train_ep/step_contact_penalty": step_pen,
                        "train_ep/event_contact_penalty": event_pen,
                        "train_ep/total_contact_penalty": step_pen + event_pen,
                        "train_ep/respawns": respawns,
                        "train_ep/contact_events_near_respawn": self._events_near_respawn,
                    })

                self._events_near_respawn = 0
                self._in_contact = False

        return True

    def _on_training_end(self) -> None:
        if self.log_path and self.episode_records:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(json.dumps(self.episode_records, indent=2))



