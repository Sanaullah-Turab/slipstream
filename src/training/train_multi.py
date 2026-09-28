from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.env.car import CAR_HALF_WIDTH, DT, MAX_SPEED
from src.env.rewards import WALL_ZONE
from src.env.vec_multi import TwoCarVecEnv
from src.training.callbacks import CheckpointCallback, MultiEvalCallback
from src.utils.config import load_config


def _git_short_sha() -> str:
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "nogit"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _warm_start(model: PPO, checkpoint: str) -> None:
    single = PPO.load(checkpoint)

    def _patch_net(multi_net, single_net):
        with torch.no_grad():
            w_single = single_net[0].weight.data
            multi_net[0].weight.data[:, :w_single.shape[1]] = w_single
            multi_net[0].weight.data[:, w_single.shape[1]:] = 0.0
            multi_net[0].bias.data = single_net[0].bias.data.clone()

    _patch_net(
        model.policy.mlp_extractor.policy_net,
        single.policy.mlp_extractor.policy_net,
    )
    _patch_net(
        model.policy.mlp_extractor.value_net,
        single.policy.mlp_extractor.value_net,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multi_ppo_config.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--warm-start",
        metavar="CHECKPOINT",
        default=None,
        help="Path to single-agent checkpoint (without .zip) for warm start.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ppo_cfg = dict(cfg["ppo"])
    train_cfg = cfg["train"]

    total_timesteps = args.total_timesteps or train_cfg["total_timesteps"]
    seed = train_cfg["seed"]
    _seed_everything(seed)

    sha = _git_short_sha()
    run_name = f"{train_cfg['run_name']}-{sha}"
    ckpt_dir = Path(train_cfg["checkpoint_dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env_constants = {
        "WALL_ZONE": WALL_ZONE,
        "MAX_SPEED": MAX_SPEED,
        "DT": DT,
        "CAR_HALF_WIDTH": CAR_HALF_WIDTH,
    }
    (ckpt_dir / "meta.json").write_text(
        json.dumps({"run_name": run_name, "git_sha": sha, "config": cfg}, indent=2, default=str)
    )

    wandb.init(
        project="slipstream",
        name=run_name,
        group=train_cfg.get("group", "multi-agent"),
        tags=train_cfg.get("tags", []),
        config={**cfg, "git_sha": sha, "env_constants": env_constants, "warm_start": args.warm_start},
        mode="disabled" if args.no_wandb else "online",
    )

    env = TwoCarVecEnv()
    policy = ppo_cfg.pop("policy")
    model = PPO(policy, env, **ppo_cfg, seed=seed, verbose=1, tensorboard_log=None)

    if args.warm_start:
        _warm_start(model, args.warm_start)

    callbacks: list[BaseCallback] = [
        CheckpointCallback(
            save_freq=train_cfg["checkpoint_freq"],
            save_dir=str(ckpt_dir),
        ),
        MultiEvalCallback(
            eval_freq=train_cfg["eval_freq"],
            n_episodes=train_cfg["eval_episodes"],
            save_best_path=str(ckpt_dir),
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(ckpt_dir / "final")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
