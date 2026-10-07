import json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)

from src.env.multi_racing_env import MultiRacingEnv, AGENTS
from scripts.run_diagnostics import load_model

def compute_bootstrap_ci(data: list[float] | np.ndarray, n_bootstraps: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    arr = np.array(data, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    if np.all(arr == arr[0]):
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(42)
    boot_means = np.mean(rng.choice(arr, size=(n_bootstraps, len(arr)), replace=True), axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return low, high

def evaluate_model_on_block(model, env: MultiRacingEnv, seed_start: int, n_episodes: int = 50, stochastic: bool = False) -> dict:
    ep_events = []
    ep_contact_steps = []
    ep_leader_paces = []
    ep_follower_paces = []
    ep_pair_paces = []

    global_col_crashes = 0
    global_solo_crashes = 0
    total_env_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        ep_steps = 0
        prev_respawns = {a: 0 for a in AGENTS}
        steps_since_collision = {a: 9999 for a in AGENTS}

        while not done:
            ep_steps += 1
            total_env_steps += 1
            obs_b = np.stack([obs[a] for a in AGENTS])
            act_b, _ = model.predict(obs_b, deterministic=not stochastic)
            actions = {a: act_b[i] for i, a in enumerate(AGENTS)}
            obs, _, _, trunc, info = env.step(actions)
            done = any(trunc.values())

            in_contact = info[AGENTS[0]].get("collision", False)
            for a in AGENTS:
                if in_contact:
                    steps_since_collision[a] = 0
                else:
                    steps_since_collision[a] += 1

                cur_resp = info[a]["respawns"]
                if cur_resp > prev_respawns[a]:
                    if steps_since_collision[a] < 30:
                        global_col_crashes += 1
                    else:
                        global_solo_crashes += 1
                prev_respawns[a] = cur_resp

        race_pos = {a: info[a]["cumulative_distance"] + info[a]["start_offset"] for a in AGENTS}
        leader = max(AGENTS, key=lambda a: race_pos[a])
        follower = AGENTS[1] if leader == AGENTS[0] else AGENTS[0]

        leader_pace = (info[leader]["cumulative_distance"] / env.track.total_length / ep_steps) * 1000.0
        follower_pace = (info[follower]["cumulative_distance"] / env.track.total_length / ep_steps) * 1000.0
        pair_pace = (leader_pace + follower_pace) / 2.0

        ep_events.append(info[AGENTS[0]]["collision_count"])
        ep_contact_steps.append(info[AGENTS[0]]["steps_in_contact"])
        ep_leader_paces.append(leader_pace)
        ep_follower_paces.append(follower_pace)
        ep_pair_paces.append(pair_pace)

    total_agent_steps = max(1, total_env_steps * 2)
    col_crash_rate = (global_col_crashes / total_agent_steps) * 1000.0
    solo_crash_rate = (global_solo_crashes / total_agent_steps) * 1000.0

    mean_events = float(np.mean(ep_events))
    events_ci = compute_bootstrap_ci(ep_events)

    mean_steps = float(np.mean(ep_contact_steps))
    steps_ci = compute_bootstrap_ci(ep_contact_steps)

    return {
        "mean_events": mean_events,
        "events_ci": events_ci,
        "mean_steps": mean_steps,
        "steps_ci": steps_ci,
        "col_crash_rate": float(col_crash_rate),
        "solo_crash_rate": float(solo_crash_rate),
        "leader_pace": float(np.mean(ep_leader_paces)),
        "follower_pace": float(np.mean(ep_follower_paces)),
        "pair_pace": float(np.mean(ep_pair_paces)),
        "ep_events": [int(x) for x in ep_events],
        "ep_contact_steps": [int(x) for x in ep_contact_steps],
    }


def format_results_markdown(results: list[dict]) -> str:
    lines = [
        "# Contact and Crash Re-baseline (Current Geometry)",
        "",
        "Geometry: car 26x11 (half_len=13.0, half_width=5.5), track_width=70.0, spawn_offset_idx=15.",
        "Evaluation: 50 episodes per block, 2000 max steps, 10000-sample bootstrap 95% CIs.",
        "",
        "| Checkpoint | Seed Range | Mode | Contact Events / ep (95% CI) | Contact Steps / ep (95% CI) | Col Crash / 1k | Solo Crash / 1k | Pair Pace (laps/1k) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        lines.append(
            f"| {r['checkpoint']} | {r['seeds']} | {r['mode']} | "
            f"{r['mean_events']:.2f} [{r['events_ci'][0]:.2f}, {r['events_ci'][1]:.2f}] | "
            f"{r['mean_steps']:.2f} [{r['steps_ci'][0]:.2f}, {r['steps_ci'][1]:.2f}] | "
            f"{r['col_crash_rate']:.4f} | {r['solo_crash_rate']:.4f} | {r['pair_pace']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _evaluate_worker(task: tuple[str, str, str, int, bool, int]) -> dict:
    checkpoint_name, checkpoint_path, seed_name, seed_start, stochastic, n_episodes = task
    torch.set_num_threads(1)
    env = MultiRacingEnv()
    model = load_model(checkpoint_path, env)
    res = evaluate_model_on_block(
        model,
        env,
        seed_start=seed_start,
        n_episodes=n_episodes,
        stochastic=stochastic,
    )
    return {
        "checkpoint": checkpoint_name,
        "seeds": seed_name,
        "mode": "Stochastic" if stochastic else "Deterministic",
        "mean_events": res["mean_events"],
        "events_ci": res["events_ci"],
        "mean_steps": res["mean_steps"],
        "steps_ci": res["steps_ci"],
        "col_crash_rate": res["col_crash_rate"],
        "solo_crash_rate": res["solo_crash_rate"],
        "pair_pace": res["pair_pace"],
    }


def run_evaluation(output_md_path: str = "docs/rebaseline_results.md", n_episodes: int = 50) -> list[dict]:
    import os
    from multiprocessing import Pool

    checkpoints = [
        ("Phase 3 Baseline", "checkpoints/multi-A/slipstream-multi-A-84a9ee1/final.zip"),
        ("slipstream-p4-1b-7b1372f final", "checkpoints/phase4/slipstream-p4-1b-7b1372f/final.zip"),
    ]
    seed_blocks = [
        ("1000-1049", 1000),
        ("2000-2049", 2000),
    ]
    tasks = []
    for ckpt_name, ckpt_path in checkpoints:
        for seed_name, seed_start in seed_blocks:
            for stoch in (False, True):
                tasks.append((ckpt_name, ckpt_path, seed_name, seed_start, stoch, n_episodes))

    num_processes = min(len(tasks), os.cpu_count() or 4)
    with Pool(processes=num_processes) as pool:
        results = pool.map(_evaluate_worker, tasks)

    if output_md_path:
        md_content = format_results_markdown(results)
        Path(output_md_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_md_path).write_text(md_content)

    return results


if __name__ == "__main__":
    run_evaluation()

