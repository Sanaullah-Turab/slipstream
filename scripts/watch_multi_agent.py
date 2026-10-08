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
    parser.add_argument(
        "--size",
        default="1280x720",
        choices=["1280x720", "1920x1080"],
        help="Window resolution (1280x720 or 1920x1080).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Render frames per second (default 60).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    w, h = (int(x) for x in args.size.split("x"))
    env = MultiRacingEnv(render_mode="human", window_size=(w, h))
    model = load_model(args.checkpoint, env)
    obs_dict, _ = env.reset()
    env.render()

    clock = pygame.time.Clock()
    frame_idx = 0
    step_subdiv = 3 if args.fps == 60 else 1
    actions = {}

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
                        frame_idx = 0
                        env.render()

            if not running:
                break

            if frame_idx % step_subdiv == 0:
                for agent in AGENTS:
                    action, _ = model.predict(obs_dict[agent], deterministic=not args.stochastic)
                    actions[agent] = action

                obs_dict, _, terms, truncs, infos = env.step(actions)

                if any(truncs.values()):
                    for agent in AGENTS:
                        info = infos[agent]
                        print(f"{agent}: laps={info['laps']}  collisions={info['collision_count']}")
                    obs_dict, _ = env.reset()
                    frame_idx = 0

            env.render()
            clock.tick(args.fps)
            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
