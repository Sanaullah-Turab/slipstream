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

class FreezeActorCallback(BaseCallback):
    def __init__(self, freeze_steps: int):
        super().__init__()
        self.freeze_steps = freeze_steps
        self.frozen = False

    def _on_training_start(self) -> None:
        if self.freeze_steps <= 0:
            return
        self.frozen = True
        print(f"Freezing actor (policy net, action net, log_std) for {self.freeze_steps} steps.")
        for name, param in self.model.policy.named_parameters():
            if "policy_net" in name or "action_net" in name or "log_std" in name:
                param.requires_grad = False
                print(f"  Frozen: {name}")

    def _on_step(self) -> bool:
        if self.frozen and self.num_timesteps >= self.freeze_steps:
            print(f"Unfreezing actor at step {self.num_timesteps}.")
            for name, param in self.model.policy.named_parameters():
                if "policy_net" in name or "action_net" in name or "log_std" in name:
                    param.requires_grad = True
                    print(f"  Unfrozen: {name}")
            self.frozen = False
        return True

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

    # Validate that the single-agent model expects exactly 11 dims
    first_layer = getattr(single.policy.mlp_extractor.policy_net, "0")
    single_input_dim = first_layer.weight.shape[1]
    if single_input_dim != 11:
        raise ValueError(
            f"Warm-start model has {single_input_dim} obs dims; expected 11 (legacy single-agent)."
        )

    def _patch_net(multi_net, single_net):
        with torch.no_grad():
            for i, (m_layer, s_layer) in enumerate(zip(multi_net, single_net)):
                if hasattr(m_layer, "weight"):
                    if i == 0:
                        # Pad the first layer
                        w_single = s_layer.weight.data
                        m_layer.weight.data[:, :w_single.shape[1]] = w_single
                        m_layer.weight.data[:, w_single.shape[1]:] = 0.0
                    else:
                        m_layer.weight.data.copy_(s_layer.weight.data)
                if hasattr(m_layer, "bias") and m_layer.bias is not None:
                    m_layer.bias.data.copy_(s_layer.bias.data)

    _patch_net(
        model.policy.mlp_extractor.policy_net,
        single.policy.mlp_extractor.policy_net,
    )
    _patch_net(
        model.policy.mlp_extractor.value_net,
        single.policy.mlp_extractor.value_net,
    )

    # Copy action_net and log_std (they have exact same dimensions)
    model.policy.action_net.load_state_dict(single.policy.action_net.state_dict())
    model.policy.log_std.data.copy_(single.policy.log_std.data)
    
    # Copy value_net (value head)
    model.policy.value_net.load_state_dict(single.policy.value_net.state_dict())


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
    parser.add_argument("--freeze-actor-steps", type=int, default=0, help="Number of timesteps to freeze the actor for.")
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

    wandb_config = {**cfg, "git_sha": sha, "env_constants": env_constants, "warm_start": args.warm_start}
    if args.warm_start:
        ws_path = Path(args.warm_start)
        meta_file = ws_path.parent / "meta.json"
        if meta_file.exists():
            try:
                wandb_config["warm_start_meta"] = json.loads(meta_file.read_text())
            except Exception:
                pass

    wandb.init(
        project="slipstream",
        name=run_name,
        group=train_cfg.get("group", "multi-agent"),
        tags=train_cfg.get("tags", []),
        config=wandb_config,
        mode="disabled" if args.no_wandb else "online",
        sync_tensorboard=True,
    )

    env = TwoCarVecEnv()
    policy = ppo_cfg.pop("policy")
    model = PPO(policy, env, **ppo_cfg, seed=seed, verbose=1, tensorboard_log="runs")

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
        ),
    ]
    if args.freeze_actor_steps > 0:
        callbacks.append(FreezeActorCallback(args.freeze_actor_steps))

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        model.save(ckpt_dir / "final")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving interrupted.zip...")
        model.save(ckpt_dir / "interrupted")
    finally:
        wandb.finish()
        env.close()


if __name__ == "__main__":
    main()
