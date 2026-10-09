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
        type=float,
        default=1.0,
        help="Playback speed multiplier relative to real time (default 1.0).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    size_str = args.size or get_default_window_size()
    w, h = (int(x) for x in size_str.split("x"))
    env = MultiRacingEnv(render_mode="human", window_size=(w, h))
    model = load_model(args.checkpoint, env)
    obs_dict, _ = env.reset()
    env.render()

    clock = pygame.time.Clock()
    frame_idx = 0
    actions = {}
    accumulator = 0.0
    sim_dt = 0.05
    speed_mult = max(0.1, float(args.speed))

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
                        accumulator = 0.0
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
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_mult = max(0.25, round(speed_mult - 0.25, 2))
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_mult = min(4.0, round(speed_mult + 0.25, 2))
                    elif event.key == pygame.K_0:
                        speed_mult = 1.0

            if not running:
                break

            dt_real = clock.tick(args.fps) / 1000.0
            dt_real = min(dt_real, 0.1)
            accumulator += dt_real * speed_mult

            while accumulator >= sim_dt:
                for agent in AGENTS:
                    action, _ = model.predict(obs_dict[agent], deterministic=not args.stochastic)
                    actions[agent] = action

                obs_dict, _, terms, truncs, infos = env.step(actions)
                accumulator -= sim_dt

                if any(truncs.values()):
                    for agent in AGENTS:
                        info = infos[agent]
                        print(f"{agent}: laps={info['laps']}  collisions={info['collision_count']}")

            if getattr(env, "_interpolator", None) is not None:
                env._interpolator.current_alpha = min(1.0, max(0.0, accumulator / sim_dt))

            env.render()
            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
