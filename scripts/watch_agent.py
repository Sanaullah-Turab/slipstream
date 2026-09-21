from stable_baselines3 import PPO

from src.env.racing_env import RacingEnv

model = PPO.load("checkpoints/single/final")
env = RacingEnv(render_mode="human")
obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"laps={info.get('laps', 0)}")
        obs, _ = env.reset()
