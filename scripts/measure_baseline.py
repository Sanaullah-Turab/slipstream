import argparse
import numpy as np
from stable_baselines3 import PPO
from src.env.racing_env import RacingEnv

def measure_baseline(checkpoint_path: str, num_episodes: int, deterministic: bool):
    env = RacingEnv(render_mode="rgb_array") # no render to go fast
    model = PPO.load(checkpoint_path)
    
    total_steps = 0
    total_crashes = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            
            if terminated:
                total_crashes += 1
                done = True
            if truncated:
                done = True
                
    crash_rate = (total_crashes / total_steps) * 1000 if total_steps > 0 else 0
    print(f"Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"Episodes: {num_episodes}")
    print(f"Total Steps: {total_steps}")
    print(f"Total Crashes: {total_crashes}")
    print(f"Crash Rate (per 1000 steps): {crash_rate:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/single/slipstream-single-v1-ae7d617/final")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    
    print(f"Measuring baseline for: {args.checkpoint}")
    measure_baseline(args.checkpoint, args.episodes, deterministic=True)
    measure_baseline(args.checkpoint, args.episodes, deterministic=False)
