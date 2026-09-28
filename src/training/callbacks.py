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
from src.env.multi_racing_env import MultiRacingEnv as _MultiRacingEnv, AGENTS as _MULTI_AGENTS
from src.env.racing_env import RacingEnv


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, keep_last: int = 3):
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
            if len(self._saved) > self.keep_last:
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

    def __init__(self, eval_freq: int, n_episodes: int):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes

    def _on_training_start(self) -> None:
        self._eval_env = _MultiRacingEnv()
        self._track_length = self._eval_env.track.total_length
        self._gap_eps = self.GAP_EPS_FRAC * self._track_length

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        leader_laps, follower_laps = [], []
        total_collisions, total_respawns, progress_gaps = [], [], []

        for ep in range(self.n_episodes):
            obs_dict, _ = self._eval_env.reset(seed=2000 + ep)
            done = False
            ep_infos: dict = {a: {} for a in _MULTI_AGENTS}

            while not done:
                actions = batch_predict(self.model, obs_dict, list(_MULTI_AGENTS), deterministic=True)
                
                obs_dict, _, _, trunc_dict, info_dict = self._eval_env.step(actions)
                ep_infos = info_dict
                done = any(trunc_dict.values())

            dist = {a: ep_infos[a]["cumulative_distance"] for a in _MULTI_AGENTS}
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
            total_respawns.append(
                ep_infos[_MULTI_AGENTS[0]]["respawns"] + ep_infos[_MULTI_AGENTS[1]]["respawns"]
            )
            
            # Pace (Laps per 1000 steps)
            leader_dist_laps = race_pos[leader] / self._track_length
            pace = (leader_dist_laps / 2000.0) * 1000.0
            speed = pace * self._track_length / 1000.0  # approximate mean speed
            # Here we just take the last episode's pace for simplicity since they are all 2000 steps
            
        logs = {
            "eval/leader/mean_laps": sum(leader_laps) / self.n_episodes,
            "eval/leader/std_laps": statistics.stdev(leader_laps) if len(leader_laps) > 1 else 0.0,
            "eval/follower/mean_laps": sum(follower_laps) / self.n_episodes,
            "eval/follower/std_laps": statistics.stdev(follower_laps) if len(follower_laps) > 1 else 0.0,
            "eval/mean_collisions_per_ep": sum(total_collisions) / self.n_episodes,
            "eval/mean_respawns_per_ep": sum(total_respawns) / self.n_episodes,
            "eval/mean_progress_gap": sum(progress_gaps) / self.n_episodes,
            "eval/leader/mean_pace_dist": pace,
            "eval/leader/mean_speed": speed,
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


