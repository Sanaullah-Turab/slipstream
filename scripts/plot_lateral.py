import argparse

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.env.racing_env import RacingEnv
from src.env.rewards import WALL_ZONE

DEFAULT_CHECKPOINT = "checkpoints/single/slipstream-single-v1-ae7d617/final"


def collect_laps(model: PPO, n_laps: int, seed: int) -> list[list[tuple[float, float, float]]]:
    env = RacingEnv()
    obs, _ = env.reset(seed=seed)

    laps: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []
    prev_lap_count = 0

    for _ in range(n_laps * 2500):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        current.append((env._progress, env._lateral, env._speed))

        lap_count = info.get("laps", 0)
        if lap_count > prev_lap_count:
            laps.append(current)
            current = []
            prev_lap_count = lap_count
            if len(laps) >= n_laps:
                break

        if terminated:
            if current:
                laps.append(current)
            obs, _ = env.reset(seed=seed + len(laps) + 1)
            current = []

    env.close()
    return laps


def plot_lateral(laps: list[list[tuple[float, float, float]]], half_width: float,
                 output: str | None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0f0f1a")

    ax_lat, ax_spd = axes
    colors = plt.cm.plasma(np.linspace(0.25, 0.85, max(len(laps), 1)))

    for i, lap in enumerate(laps):
        prog = [p for p, _, _ in lap]
        lat = [l for _, l, _ in lap]
        spd = [s for _, _, s in lap]
        label = f"Lap {i + 1}"
        ax_lat.plot(prog, lat, color=colors[i], linewidth=1.2, alpha=0.9, label=label)
        ax_spd.plot(prog, spd, color=colors[i], linewidth=1.2, alpha=0.9, label=label)

    wall_zone = half_width * WALL_ZONE
    for ax in (ax_lat,):
        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", label="centerline")
        ax.axhline(wall_zone, color="orange", linewidth=0.7, linestyle=":", label=f"WALL_ZONE ({WALL_ZONE})")
        ax.axhline(-wall_zone, color="orange", linewidth=0.7, linestyle=":")
        ax.axhline(half_width, color="#ff4444", linewidth=0.9, linestyle="-", label="wall")
        ax.axhline(-half_width, color="#ff4444", linewidth=0.9, linestyle="-")
        ax.set_ylim(-half_width * 1.15, half_width * 1.15)

    ax_lat.set_ylabel("Lateral offset (px, signed)", color="white", fontsize=11)
    ax_lat.set_title("Lateral position vs track progress\n"
                     "positive = outer-normal side   |   constant offset = bias,  varying = racing line",
                     color="white", fontsize=12)

    ax_spd.set_ylabel("Speed (px/s)", color="white", fontsize=11)
    ax_spd.set_xlabel("Track progress  (0 = start/finish line)", color="white", fontsize=11)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#cccccc")
        ax.spines[:].set_color("#333355")
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"saved to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot lateral offset and speed vs track progress.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Path to save PNG (e.g. lateral_plot.png). Omit to show interactively.")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint)

    tmp = RacingEnv()
    tmp.reset()
    half_width = tmp.track.half_width
    tmp.close()

    print(f"half_width = {half_width}  WALL_ZONE boundary = {half_width * WALL_ZONE:.1f}")
    print(f"Collecting {args.laps} laps from {args.checkpoint} ...")

    laps = collect_laps(model, args.laps, args.seed)
    print(f"Collected {len(laps)} lap(s)  ({sum(len(l) for l in laps)} steps total)")

    plot_lateral(laps, half_width, args.output)


if __name__ == "__main__":
    main()
