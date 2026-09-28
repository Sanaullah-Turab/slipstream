import argparse
import statistics
import numpy as np
from stable_baselines3 import PPO

from src.env.multi_racing_env import MultiRacingEnv, AGENTS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/multi/final")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    env = MultiRacingEnv()
    model = PPO.load(args.checkpoint)

    gap_eps = 0.005 * env.track.total_length

    leader_laps, follower_laps = [], []
    total_collisions, total_respawns, progress_gaps = [], [], []

    print(f"Evaluating {args.checkpoint} over {args.episodes} episodes...")

    for ep in range(args.episodes):
        obs_dict, _ = env.reset(seed=1000 + ep)
        done = False
        ep_infos = {a: {} for a in AGENTS}

        while not done:
            obs_batch = np.stack([obs_dict[a] for a in AGENTS])
            actions_batch, _ = model.predict(obs_batch, deterministic=True)
            actions = {a: actions_batch[i] for i, a in enumerate(AGENTS)}
            
            obs_dict, _, _, trunc_dict, info_dict = env.step(actions)
            ep_infos = info_dict
            done = any(trunc_dict.values())

        dist = {a: ep_infos[a]["cumulative_distance"] for a in AGENTS}
        gap = abs(dist[AGENTS[0]] - dist[AGENTS[1]])
        progress_gaps.append(gap / env.track.total_length)

        if gap < gap_eps:
            leader, follower = AGENTS[0], AGENTS[1]
        else:
            leader = max(AGENTS, key=lambda a: dist[a])
            follower = AGENTS[1] if leader == AGENTS[0] else AGENTS[0]

        leader_laps.append(ep_infos[leader]["laps"])
        follower_laps.append(ep_infos[follower]["laps"])
        
        # In multi-agent, collisions are rising edges, shared by both.
        # We can just count it for agent 0
        total_collisions.append(ep_infos[AGENTS[0]]["collision_count"])
        total_respawns.append(ep_infos[AGENTS[0]]["respawns"] + ep_infos[AGENTS[1]]["respawns"])
        
        print(f"Ep {ep+1}/{args.episodes}: Leader Laps={leader_laps[-1]:.2f}, "
              f"Follower Laps={follower_laps[-1]:.2f}, Collisions={total_collisions[-1]}")

    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Leader Laps:   {statistics.mean(leader_laps):.2f} ± {statistics.stdev(leader_laps) if len(leader_laps) > 1 else 0:.2f}")
    print(f"Follower Laps: {statistics.mean(follower_laps):.2f} ± {statistics.stdev(follower_laps) if len(follower_laps) > 1 else 0:.2f}")
    print(f"Progress Gap:  {statistics.mean(progress_gaps):.4f}")
    print(f"Mean Collisions/Ep: {statistics.mean(total_collisions):.2f}")
    
    # Crash rate per 1000 steps. 
    # Total steps per episode = 3000. For two agents, total agent steps = 6000 per episode.
    # We should define crash rate per 1000 environment steps (i.e., time). 
    # So total steps = args.episodes * 3000
    env_steps = args.episodes * 3000
    crash_rate = (sum(total_collisions) / env_steps) * 1000
    print(f"Crash Rate (per 1000 env steps): {crash_rate:.4f}")

if __name__ == "__main__":
    main()
