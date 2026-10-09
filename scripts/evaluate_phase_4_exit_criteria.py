import json
import math
from pathlib import Path
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO

from src.env.multi_racing_env import MultiRacingEnv, AGENTS, aggregate_fault_counts
from src.env.track import Track, SHANGHAI_TRACK_WIDTH


def load_model_with_patch(path: str, env: MultiRacingEnv) -> PPO:
    model = PPO.load(path)
    target_dim = env.observation_space(AGENTS[0]).shape[0]
    first_layer = getattr(model.policy.mlp_extractor.policy_net, "0")
    single_input_dim = first_layer.weight.shape[1]
    if single_input_dim < target_dim:
        model.observation_space = Box(low=-np.inf, high=np.inf, shape=(target_dim,), dtype=np.float32)
        model.policy.observation_space = model.observation_space
        def _patch(net):
            with torch.no_grad():
                first = getattr(net, "0")
                w = torch.zeros((first.out_features, target_dim), device=first.weight.device)
                w[:, :single_input_dim] = first.weight.data
                first.weight.data = w
        _patch(model.policy.mlp_extractor.policy_net)
        _patch(model.policy.mlp_extractor.value_net)
    return model


def evaluate_overtaking_and_ablation(model_path: str, n_episodes: int = 20) -> dict:
    results = {}
    for draft_flag in [True, False]:
        env = MultiRacingEnv(enable_draft=draft_flag, continuous=False)
        model = load_model_with_patch(model_path, env)
        total_swaps = 0
        total_laps = 0
        total_contact_events = 0
        total_respawns = 0
        follower_faults = 0
        leader_faults = 0
        neutral_faults = 0
        speeds = []

        for ep in range(n_episodes):
            seed = 1000 + ep
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                obs_b = np.stack([obs[a] for a in AGENTS])
                act_b, _ = model.predict(obs_b, deterministic=True)
                actions = {a: act_b[i] for i, a in enumerate(AGENTS)}
                obs, _, _, trunc, info = env.step(actions)
                done = any(trunc.values())
                for a in AGENTS:
                    speeds.append(info[a]["speed"])

            total_swaps += info[AGENTS[0]].get("position_swaps", 0)
            total_laps += info[AGENTS[0]]["laps"] + info[AGENTS[1]]["laps"]
            total_contact_events += info[AGENTS[0]]["collision_count"]
            total_respawns += info[AGENTS[0]]["respawns"] + info[AGENTS[1]]["respawns"]
            fc = aggregate_fault_counts(info, AGENTS)
            follower_faults += fc["follower"]
            leader_faults += fc["leader"]
            neutral_faults += fc["neutral"]

        key = "draft_on" if draft_flag else "draft_off"
        swaps_per_10_laps = (total_swaps / max(1, total_laps)) * 10.0
        results[key] = {
            "total_swaps": total_swaps,
            "swaps_per_episode": float(total_swaps / n_episodes),
            "total_laps": total_laps,
            "swaps_per_10_laps": float(swaps_per_10_laps),
            "contact_events_per_episode": float(total_contact_events / n_episodes),
            "respawns_total": total_respawns,
            "follower_faults_per_episode": float(follower_faults / n_episodes),
            "leader_faults_per_episode": float(leader_faults / n_episodes),
            "neutral_faults_per_episode": float(neutral_faults / n_episodes),
            "mean_speed": float(np.mean(speeds)),
            "max_speed": float(np.max(speeds)),
        }
    return results


def evaluate_defending_and_lines(model_path: str, n_episodes: int = 15) -> dict:
    env = MultiRacingEnv(enable_draft=True, continuous=False)
    model = load_model_with_patch(model_path, env)

    threatened_lat_devs = []
    unthreatened_lat_devs = []
    apex_clip_steps = 0
    total_corner_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        done = False
        while not done:
            obs_b = np.stack([obs[a] for a in AGENTS])
            act_b, _ = model.predict(obs_b, deterministic=True)
            actions = {a: act_b[i] for i, a in enumerate(AGENTS)}
            obs, _, _, trunc, info = env.step(actions)
            done = any(trunc.values())

            s0, s1 = env._state[AGENTS[0]], env._state[AGENTS[1]]
            p0 = s0["cumulative_distance"] + s0["start_offset"]
            p1 = s1["cumulative_distance"] + s1["start_offset"]
            if p0 >= p1:
                lead, foll = s0, s1
            else:
                lead, foll = s1, s0

            gap_sec = foll.get("gap_to_leader_seconds", 0.0)
            lead_ts = env.track.get_track_state(lead["pos"])
            c_val = getattr(lead_ts, "curvature", 0.0)

            if abs(c_val) > 0.001:
                total_corner_steps += 1
                in_dir = 1.0 if c_val > 0.0 else -1.0
                if lead["lateral"] * in_dir > 1.0:
                    apex_clip_steps += 1

            if gap_sec < 1.2:
                threatened_lat_devs.append(abs(lead["lateral"]))
            elif gap_sec > 3.0:
                unthreatened_lat_devs.append(abs(lead["lateral"]))

    return {
        "mean_threatened_lateral_dev": float(np.mean(threatened_lat_devs)) if threatened_lat_devs else 0.0,
        "mean_unthreatened_lateral_dev": float(np.mean(unthreatened_lat_devs)) if unthreatened_lat_devs else 0.0,
        "apex_clipping_frequency": float(apex_clip_steps / max(1, total_corner_steps)),
    }


def evaluate_cross_play(model_p4_path: str, model_p3_path: str, n_episodes: int = 15) -> dict:
    env = MultiRacingEnv(enable_draft=True, continuous=False)
    m_p4 = load_model_with_patch(model_p4_path, env)
    m_p3 = load_model_with_patch(model_p3_path, env)

    scenarios = ["p4_follower_vs_p3_leader", "p4_leader_vs_p3_follower"]
    results = {}

    for scen in scenarios:
        p4_wins = 0
        p3_wins = 0
        swaps_total = 0
        col_crashes = 0
        solo_crashes = 0

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=3000 + ep)
            s0 = env._state[AGENTS[0]]["start_offset"]
            s1 = env._state[AGENTS[1]]["start_offset"]
            init_lead = AGENTS[0] if s0 >= s1 else AGENTS[1]
            init_foll = AGENTS[1] if init_lead == AGENTS[0] else AGENTS[0]

            if scen == "p4_follower_vs_p3_leader":
                agent_model = {init_lead: m_p3, init_foll: m_p4}
                p4_agent = init_foll
                p3_agent = init_lead
            else:
                agent_model = {init_lead: m_p4, init_foll: m_p3}
                p4_agent = init_lead
                p3_agent = init_foll

            done = False
            while not done:
                actions = {}
                for a in AGENTS:
                    act, _ = agent_model[a].predict(obs[a], deterministic=True)
                    actions[a] = act
                obs, _, _, trunc, info = env.step(actions)
                done = any(trunc.values())

            swaps_total += info[AGENTS[0]].get("position_swaps", 0)
            race_pos_p4 = info[p4_agent]["cumulative_distance"] + info[p4_agent]["start_offset"]
            race_pos_p3 = info[p3_agent]["cumulative_distance"] + info[p3_agent]["start_offset"]
            if race_pos_p4 > race_pos_p3:
                p4_wins += 1
            else:
                p3_wins += 1

            for a in AGENTS:
                resp = info[a]["respawns"]
                if resp > 0:
                    solo_crashes += resp

        results[scen] = {
            "p4_wins": p4_wins,
            "p3_wins": p3_wins,
            "p4_win_rate": float(p4_wins / n_episodes),
            "position_swaps_per_ep": float(swaps_total / n_episodes),
            "crashes": solo_crashes,
        }

    return results


def main():
    m4_path = "checkpoints/f1_tactics/shanghai-f1-smooth-tactics-e2b6007/final.zip"
    m3_path = "checkpoints/multi-A/slipstream-multi-A-84a9ee1/final.zip"

    print("Running Criterion 1: Overtaking & Draft Ablation...", flush=True)
    crit1 = evaluate_overtaking_and_ablation(m4_path, n_episodes=20)
    print("Criterion 1 complete!", flush=True)

    print("Running Criterion 2: Defending & Line Metrics...", flush=True)
    crit2 = evaluate_defending_and_lines(m4_path, n_episodes=15)
    print("Criterion 2 complete!", flush=True)

    print("Running Criterion 5: Cross-Play Verification...", flush=True)
    crit5 = evaluate_cross_play(m4_path, m3_path, n_episodes=15)
    print("Criterion 5 complete!", flush=True)

    out = {
        "criterion_1_draft_ablation": crit1,
        "criterion_2_defending": crit2,
        "criterion_5_cross_play": crit5,
    }
    out_path = Path("scratch/phase_4_exit_criteria_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"All Exit Criteria evaluated and saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
