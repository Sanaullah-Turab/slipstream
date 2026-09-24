from __future__ import annotations

import json
import subprocess
from pathlib import Path

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.env.car import DT
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
