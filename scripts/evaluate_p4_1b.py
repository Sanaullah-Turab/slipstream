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
