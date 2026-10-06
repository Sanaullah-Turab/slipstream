import argparse
import sys
import pygame
from stable_baselines3 import PPO

from scripts.run_diagnostics import load_model
from src.env.multi_racing_env import MultiRacingEnv, AGENTS


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/phase4/slipstream-p4-1b-7b1372f/final",
        help="Path to multi-agent checkpoint (with or without .zip).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Use stochastic actions instead of deterministic.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    env = MultiRacingEnv(render_mode="human")
    model = load_model(args.checkpoint, env)
    obs_dict, _ = env.reset()
    env.render()

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        break
                    elif event.key == pygame.K_r:
                        obs_dict, _ = env.reset()
                        env.render()

            if not running:
                break

            actions = {}
            for agent in AGENTS:
                action, _ = model.predict(obs_dict[agent], deterministic=not args.stochastic)
                actions[agent] = action

            obs_dict, _, terms, truncs, infos = env.step(actions)

            if any(truncs.values()):
                for agent in AGENTS:
                    info = infos[agent]
                    print(f"{agent}: laps={info['laps']}  collisions={info['collision_count']}")
                obs_dict, _ = env.reset()
                env.render()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
