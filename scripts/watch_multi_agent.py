import argparse

import numpy as np
from stable_baselines3 import PPO

from scripts.run_diagnostics import load_model
from src.env.multi_racing_env import MultiRacingEnv, AGENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    default="checkpoints/phase4/slipstream-p4-1b-7b1372f/final",
    help="Path to multi-agent checkpoint (with or without .zip).",
)
args = parser.parse_args()

env = MultiRacingEnv(render_mode="human")
model = load_model(args.checkpoint, env)
obs_dict, _ = env.reset()

while True:
    actions = {}
    for agent in AGENTS:
        action, _ = model.predict(obs_dict[agent], deterministic=True)
        actions[agent] = action

    obs_dict, _, terms, truncs, infos = env.step(actions)

    if any(truncs.values()):
        for agent in AGENTS:
            info = infos[agent]
            print(f"{agent}: laps={info['laps']}  collisions={info['collision_count']}")
        obs_dict, _ = env.reset()
