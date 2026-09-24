import argparse

from stable_baselines3 import PPO

from src.env.racing_env import RacingEnv

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/single/final",
                    help="Path to checkpoint (without .zip). Defaults to legacy path.")
args = parser.parse_args()

model = PPO.load(args.checkpoint)
env = RacingEnv(render_mode="human")
obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"laps={info.get('laps', 0)}")
        obs, _ = env.reset()
