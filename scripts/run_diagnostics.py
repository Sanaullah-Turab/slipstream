import argparse
import json
import math
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO

from src.env.multi_racing_env import MultiRacingEnv, AGENTS

def load_model(path: str, env: MultiRacingEnv) -> PPO:
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

def get_relative_state(env: MultiRacingEnv) -> tuple[float, float, float]:
    pos0 = env._state[AGENTS[0]]["pos"]
    pos1 = env._state[AGENTS[1]]["pos"]
    h0 = env._state[AGENTS[0]]["heading"]
    sp0 = env._state[AGENTS[0]]["speed"]
    sp1 = env._state[AGENTS[1]]["speed"]
    v0 = sp0 * np.array([math.cos(h0), math.sin(h0)])
    v1 = sp1 * np.array([math.cos(env._state[AGENTS[1]]["heading"]), math.sin(env._state[AGENTS[1]]["heading"])])

    dpos = pos1 - pos0
    fwd0 = np.array([math.cos(h0), math.sin(h0)])
    lat0 = np.array([-math.sin(h0), math.cos(h0)])

    abs_long = abs(float(np.dot(dpos, fwd0)))
    abs_lat = abs(float(np.dot(dpos, lat0)))
    rel_speed = float(np.linalg.norm(v1 - v0))
    return abs_long, abs_lat, rel_speed

def run_round2_eval(env: MultiRacingEnv, model: PPO, num_episodes: int = 50, stochastic: bool = False):
    episodes_data = []
    all_events = []
    separation_durations = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False

        ep_steps = 0
        ep_events = []
        was_in_contact = False
        current_event = None

        while not done:
            ep_steps += 1
            obs_batch = np.stack([obs[a] for a in AGENTS])
            act_batch, _ = model.predict(obs_batch, deterministic=not stochastic)
            actions = {a: act_batch[i] for i, a in enumerate(AGENTS)}

            obs, _, _, trunc, info = env.step(actions)
            done = any(trunc.values())

            in_contact = info[AGENTS[0]].get("collision", False)

            if in_contact:
                abs_long, abs_lat, rel_speed = get_relative_state(env)
                if not was_in_contact:
                    current_event = {
                        "episode": ep,
                        "start_step": ep_steps,
                        "duration": 1,
                        "step1": {
                            "abs_long": abs_long,
                            "abs_lat": abs_lat,
                            "rel_speed": rel_speed,
                        },
                        "step20": None,
                    }
                else:
                    current_event["duration"] += 1
                    if current_event["duration"] == 20:
                        current_event["step20"] = {
                            "abs_long": abs_long,
                            "abs_lat": abs_lat,
                            "rel_speed": rel_speed,
                        }
            else:
                if was_in_contact and current_event is not None:
                    current_event["end_step"] = ep_steps - 1
                    ep_events.append(current_event)
                    current_event = None

            was_in_contact = in_contact

        if was_in_contact and current_event is not None:
            current_event["end_step"] = ep_steps
            ep_events.append(current_event)
            current_event = None

        for i in range(len(ep_events) - 1):
            sep = ep_events[i + 1]["start_step"] - (ep_events[i]["end_step"] + 1)
            separation_durations.append(sep)

        total_cols = info[AGENTS[0]]["collision_count"]
        ep_contact_steps = info[AGENTS[0]]["steps_in_contact"]

        episodes_data.append({
            "episode": ep,
            "events": ep_events,
            "total_cols": total_cols,
            "contact_steps": ep_contact_steps,
        })
        all_events.extend(ep_events)

    return episodes_data, all_events, separation_durations

def analyze_round2(episodes_data: list[dict], all_events: list[dict], separation_durations: list[int]) -> dict:
    total_contact_steps = sum(ep["contact_steps"] for ep in episodes_data)
    total_events = len(all_events)

    hist = {
        "1": sum(1 for ev in all_events if ev["start_step"] == 1),
        "2_to_10": sum(1 for ev in all_events if 2 <= ev["start_step"] <= 10),
        "11_to_50": sum(1 for ev in all_events if 11 <= ev["start_step"] <= 50),
        "51_to_200": sum(1 for ev in all_events if 51 <= ev["start_step"] <= 200),
        "over_200": sum(1 for ev in all_events if ev["start_step"] > 200),
    }

    eps_with_contact_step1 = sum(
        1 for ep in episodes_data if any(ev["start_step"] == 1 for ev in ep["events"])
    )

    steps_start_1 = sum(ev["duration"] for ev in all_events if ev["start_step"] == 1)
    steps_start_first10 = sum(ev["duration"] for ev in all_events if ev["start_step"] <= 10)

    share_start_1 = steps_start_1 / total_contact_steps if total_contact_steps > 0 else 0.0
    share_start_first10 = steps_start_first10 / total_contact_steps if total_contact_steps > 0 else 0.0

    excl_contact_steps_per_ep = []
    excl_events_per_ep = []
    for ep in episodes_data:
        rem_events = [ev for ev in ep["events"] if ev["start_step"] > 10]
        rem_steps = sum(ev["duration"] for ev in rem_events)
        excl_events_per_ep.append(len(rem_events))
        excl_contact_steps_per_ep.append(rem_steps)

    mean_excl_contact_steps = float(np.mean(excl_contact_steps_per_ep)) if excl_contact_steps_per_ep else 0.0
    mean_excl_events = float(np.mean(excl_events_per_ep)) if excl_events_per_ep else 0.0

    long_events = [ev for ev in all_events if ev["duration"] > 50]
    long_first10 = [ev for ev in long_events if ev["start_step"] <= 10]
    long_later = [ev for ev in long_events if ev["start_step"] > 10]

    def geom_stats(ev_list: list[dict]) -> dict | None:
        if not ev_list:
            return None
        res = {
            "count": len(ev_list),
            "step1_abs_long": float(np.mean([ev["step1"]["abs_long"] for ev in ev_list])),
            "step1_abs_lat": float(np.mean([ev["step1"]["abs_lat"] for ev in ev_list])),
            "step1_rel_speed": float(np.mean([ev["step1"]["rel_speed"] for ev in ev_list])),
        }
        with_step20 = [ev for ev in ev_list if ev["step20"] is not None]
        if with_step20:
            res["step20_abs_long"] = float(np.mean([ev["step20"]["abs_long"] for ev in with_step20]))
            res["step20_abs_lat"] = float(np.mean([ev["step20"]["abs_lat"] for ev in with_step20]))
            res["step20_rel_speed"] = float(np.mean([ev["step20"]["rel_speed"] for ev in with_step20]))
        return res

    long_geom_first10 = geom_stats(long_first10)
    long_geom_later = geom_stats(long_later)

    sep_stats = {}
    if separation_durations:
        sep_stats = {
            "count": len(separation_durations),
            "mean": float(np.mean(separation_durations)),
            "median": float(np.median(separation_durations)),
            "p90": float(np.percentile(separation_durations, 90)),
            "min": int(np.min(separation_durations)),
            "max": int(np.max(separation_durations)),
        }

    return {
        "total_events": total_events,
        "total_contact_steps": total_contact_steps,
        "hist_start_steps": hist,
        "eps_with_contact_step1": eps_with_contact_step1,
        "steps_start_1": steps_start_1,
        "share_start_1": share_start_1,
        "steps_start_first10": steps_start_first10,
        "share_start_first10": share_start_first10,
        "mean_excl_contact_steps": mean_excl_contact_steps,
        "mean_excl_events": mean_excl_events,
        "long_geom_first10": long_geom_first10,
        "long_geom_later": long_geom_later,
        "sep_stats": sep_stats,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    env = MultiRacingEnv()
    model = load_model(args.checkpoint, env)
    ep_data, evs, seps = run_round2_eval(env, model, num_episodes=args.episodes, stochastic=args.stochastic)
    res = analyze_round2(ep_data, evs, seps)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
