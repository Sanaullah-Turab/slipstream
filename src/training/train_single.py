from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.env.racing_env import RacingEnv
from src.env.rewards import WALL_ZONE
from src.training.callbacks import CheckpointCallback, WandbEvalCallback
from src.utils.config import load_config


def _git_short_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_config.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ppo_cfg = dict(cfg["ppo"])
    train_cfg = cfg["train"]

    total_timesteps = args.total_timesteps or train_cfg["total_timesteps"]
    seed = train_cfg["seed"]
    _seed_everything(seed)

    sha = _git_short_sha()
    wandb.init(
        project="slipstream",
        name=f"{train_cfg['run_name']}-{sha}",
        group=train_cfg.get("group", "single-agent"),
        tags=train_cfg.get("tags", []),
        config={**cfg, "env_constants": {"WALL_ZONE": WALL_ZONE}},
        mode="disabled" if args.no_wandb else "online",
    )

    env = Monitor(RacingEnv(render_mode=cfg["env"]["render_mode"]))

    policy = ppo_cfg.pop("policy")
    model = PPO(policy, env, **ppo_cfg, seed=seed, verbose=1, tensorboard_log=None)

    callbacks: list[BaseCallback] = [
        CheckpointCallback(
            save_freq=train_cfg["checkpoint_freq"],
            save_dir=train_cfg["checkpoint_dir"],
        ),
        WandbEvalCallback(
            eval_freq=train_cfg["eval_freq"],
            n_episodes=train_cfg["eval_episodes"],
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final = Path(train_cfg["checkpoint_dir"]) / "final"
    final.parent.mkdir(parents=True, exist_ok=True)
    model.save(final)

    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
