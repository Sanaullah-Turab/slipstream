import argparse
import statistics
import numpy as np
from stable_baselines3 import PPO

from src.env.multi_racing_env import MultiRacingEnv, AGENTS

def main(args):
    env = MultiRacingEnv()
    
    if args.random:
        print("Using RANDOM policy for sanity check...")
        model = None
    else:
        model = PPO.load(args.checkpoint)
        
        # Auto-patch if testing step-0 single agent model directly
        single_input_dim = model.policy.mlp_extractor.policy_net[0].weight.shape[1]
        if single_input_dim == 11:
            import torch
            from gymnasium.spaces import Box
            
            model.observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
            model.policy.observation_space = model.observation_space
            
            def _patch_net(net):
                with torch.no_grad():
                    w_single = net[0].weight.data
                    w_multi = torch.zeros((net[0].out_features, 15), device=w_single.device)
                    w_multi[:, :11] = w_single
                    net[0].weight.data = w_multi
            
            _patch_net(model.policy.mlp_extractor.policy_net)
            _patch_net(model.policy.mlp_extractor.value_net)

    gap_eps = 0.005 * env.track.total_length

    leader_laps, follower_laps = [], []
    total_collisions, total_respawns, progress_gaps = [], [], []
    mean_speeds = {a: [] for a in AGENTS}
    leader_dist_laps, follower_dist_laps = [], []

    print(f"Evaluating {args.checkpoint} over {args.episodes} episodes...")

    for ep in range(args.episodes):
        obs_dict, _ = env.reset(seed=1000 + ep)
        done = False
        ep_infos = {a: {} for a in AGENTS}

        prev_respawns = {a: 0 for a in AGENTS}
        steps_since_collision = {a: 9999 for a in AGENTS}
        
        ep_solo_crashes = 0
        ep_col_crashes = 0

        while not done:
            if model is None:
                actions = {a: env.action_space(a).sample() for a in AGENTS}
            else:
                obs_batch = np.stack([obs_dict[a] for a in AGENTS])
                actions_batch, _ = model.predict(obs_batch, deterministic=not args.stochastic)
                actions = {a: actions_batch[i] for i, a in enumerate(AGENTS)}
            
            obs_dict, _, _, trunc_dict, info_dict = env.step(actions)
            ep_infos = info_dict
            done = any(trunc_dict.values())
            
            for a in AGENTS:
                if info_dict[a].get("collision", False):
                    steps_since_collision[a] = 0
                else:
                    steps_since_collision[a] += 1
                
                current_respawns = info_dict[a]["respawns"]
                if current_respawns > prev_respawns[a]:
                    if steps_since_collision[a] < 30:
                        ep_col_crashes += 1
                    else:
                        ep_solo_crashes += 1
                prev_respawns[a] = current_respawns

        # Rank by race position (distance traveled + starting offset)
        race_pos = {a: ep_infos[a]["cumulative_distance"] + ep_infos[a]["start_offset"] for a in AGENTS}
        gap = abs(race_pos[AGENTS[0]] - race_pos[AGENTS[1]])
        progress_gaps.append(gap / env.track.total_length)

        if gap < gap_eps:
            leader, follower = AGENTS[0], AGENTS[1]
        else:
            leader = max(AGENTS, key=lambda a: race_pos[a])
            follower = AGENTS[1] if leader == AGENTS[0] else AGENTS[0]

        leader_laps.append(ep_infos[leader]["laps"])
        follower_laps.append(ep_infos[follower]["laps"])
        
        # Calculate pacing in laps based on distance traveled
        for a in AGENTS:
            dist_laps = ep_infos[a]["cumulative_distance"] / env.track.total_length
            mean_speeds[a].append(ep_infos[a]["speed"])
            if a == leader:
                leader_dist_laps.append(dist_laps)
            else:
                follower_dist_laps.append(dist_laps)
        
        total_collisions.append(ep_infos[AGENTS[0]]["collision_count"])
        total_respawns.append(ep_solo_crashes + ep_col_crashes)
        
        print(f"Ep {ep+1}/{args.episodes}: Leader Laps={leader_laps[-1]:.2f}, "
              f"Follower Laps={follower_laps[-1]:.2f}, Collisions={total_collisions[-1]}, "
              f"Solo Crashes={ep_solo_crashes}, Col-Crashes={ep_col_crashes}")
        
        if not hasattr(env, '_global_solo'):
            env._global_solo = 0
            env._global_col = 0
        env._global_solo += ep_solo_crashes
        env._global_col += ep_col_crashes

    mean_leader = statistics.mean(leader_laps)
    mean_follower = statistics.mean(follower_laps)
    
    mean_leader_pace = (statistics.mean(leader_dist_laps) / 2000.0) * 1000.0
    mean_follower_pace = (statistics.mean(follower_dist_laps) / 2000.0) * 1000.0
    mean_leader_speed = statistics.mean(mean_speeds[leader]) if leader in mean_speeds else 0.0
    mean_follower_speed = statistics.mean(mean_speeds[follower]) if follower in mean_speeds else 0.0
    
    # 25 units is roughly 0.02 laps. If follower is more than 0.1 laps ahead, it's a bug.
    if mean_leader < mean_follower - 0.1:
        print(f"\nWARNING: Leader mean integer laps ({mean_leader:.2f}) < Follower mean integer laps ({mean_follower:.2f}). "
              f"This indicates a bug in lap counting vs cumulative distance due to spawn offsets crossing the start line.")
              
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Leader Integer Laps:   {mean_leader:.2f} ± {statistics.stdev(leader_laps) if len(leader_laps) > 1 else 0:.2f}")
    print(f"Follower Integer Laps: {mean_follower:.2f} ± {statistics.stdev(follower_laps) if len(follower_laps) > 1 else 0:.2f}")
    print(f"Leader Pace (Laps/1k):   {mean_leader_pace:.4f} (Avg Speed: {mean_leader_speed:.1f})")
    print(f"Follower Pace (Laps/1k): {mean_follower_pace:.4f} (Avg Speed: {mean_follower_speed:.1f})")
    print(f"Progress Gap:  {statistics.mean(progress_gaps):.4f}")
    print(f"Mean Collisions/Ep: {statistics.mean(total_collisions):.2f}")
    
    agent_steps = args.episodes * 2000 * 2
    solo_rate = (env._global_solo / agent_steps) * 1000
    col_rate = (env._global_col / agent_steps) * 1000
    total_rate = ((env._global_solo + env._global_col) / agent_steps) * 1000
    
    print(f"\nCrash Rates (per 1000 agent-steps):")
    print(f"  Solo Crashes:            {solo_rate:.4f}")
    print(f"  Collision-Induced:       {col_rate:.4f}")
    print(f"  Total Respawns:          {total_rate:.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/multi/final")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--random", action="store_true", help="Run a random policy sanity check")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.random:
        print(f"Evaluating RANDOM policy over {args.episodes} episodes...")
    else:
        print(f"Evaluating {args.checkpoint} over {args.episodes} episodes...")
    main(args)
