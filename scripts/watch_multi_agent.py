import argparse
import sys
import pygame
from stable_baselines3 import PPO

from scripts.run_diagnostics import load_model
from src.env.multi_racing_env import MultiRacingEnv, AGENTS


def get_default_window_size() -> str:
    try:
        if not pygame.display.get_init():
            pygame.display.init()
        info = pygame.display.Info()
        sw, sh = info.current_w, info.current_h
        if sw > 0 and sh > 0:
            max_h = int(sh * 0.90)
            max_w = int(sw * 0.90)
            target_h = min(max_h, int(max_w * 9 / 16))
            target_w = int(target_h * 16 / 9)
            if target_w >= 1280 and target_h >= 720:
                return f"{target_w}x{target_h}"
    except Exception:
        pass
    return "1280x720"


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
        default=None,
        help="Window resolution override (e.g. 1280x720 or 1920x1080).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Render frames per second (default 60).",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=2,
        help="Simulation steps per rendered frame (default 2).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    size_str = args.size or get_default_window_size()
    w, h = (int(x) for x in size_str.split("x"))
    env = MultiRacingEnv(render_mode="human", window_size=(w, h))
    if getattr(env, "_interpolator", None) is not None:
        env._interpolator.step_subdivisions = 1
    model = load_model(args.checkpoint, env)
    obs_dict, _ = env.reset()
    env.render()

    clock = pygame.time.Clock()
    frame_idx = 0
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
                    elif event.key == pygame.K_h:
                        if getattr(env, "_hud", None) is not None:
                            env._hud.show_help = not env._hud.show_help
                    elif event.key == pygame.K_1:
                        env.set_camera_mode("ham")
                    elif event.key == pygame.K_2:
                        env.set_camera_mode("ver")
                    elif event.key in (pygame.K_a, pygame.K_F1):
                        env.set_camera_mode("auto")
                    elif event.key == pygame.K_o:
                        env.toggle_camera_overview()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        env.zoom_in()
                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                        env.zoom_out()

            if not running:
                break

            steps_per_frame = max(1, args.speed)
            for _ in range(steps_per_frame):
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
                    break

            env.render()
            clock.tick(args.fps)
            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
