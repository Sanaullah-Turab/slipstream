# Phase 4: Competitive Racing & Advanced Interaction

## Goal
Transform the agents from independent hot-lappers into competitive racers. The agents will be given the physical means to overtake (aerodynamic slipstreaming) and the incentive to block and pass (positional rewards). The final goal is clean, aggressive racing, measured through empirical cross-play and ablation testing.

---

## 1. Physics Engine Upgrades

### A. Oriented Bounding Box (OBB) Collisions & Momentum
Before adding competitive rewards, the cars must exist as solid physical objects to prevent ghosting through blocks.
- **Mechanic:** Upgrade collision detection to OBB (rectangles) using the Separating Axis Theorem (SAT). Use a swept check or substeps to prevent high-speed tunneling.
- **Momentum Transfer:** Implement inelastic collision responses (restitution 0.2 to 0.4). Fully elastic bounces will launch cars into walls; low restitution mimics heavy cars scrubbing speed upon contact.

### B. Aerodynamic Slipstreaming (Drafting)
- **Mechanic:** Define a drafting zone oriented by the leader heading. The draft effect fades linearly with distance.
- **Implementation:** Raise the `MAX_SPEED` cap and acceleration limits dynamically when a car is inside the zone.
- **Calibration (Scripted Test):** Build a scripted setup (Leader on solo line, Follower inside zone) and measure how much gap the Follower closes on the longest straight. Tune the boost so a pass is possible on some laps, but not guaranteed.

---

## 2. Reward System Redesign

### A. Continuous Zero-Sum Positional Reward
To avoid flickering signals when cars are side-by-side:
- **Formula:** `r_pos = k * clip((my_dist - opp_dist) / g0, -1, 1)` where `g0` is ~1 car length.
- **Scaling:** Set `k` to 10-25% of the average step progress reward (e.g., if progress averages 0.2/step, `k` in range 0.02 to 0.05). Log both terms and tune.

### B. Fault-Assigned Collision Penalties
- **Baseline Penalty:** Keep a small symmetric penalty (default -0.1) for all contact to discourage sloppy driving without making them overly avoidant.
- **Severe Fault Penalty:** Define holding position duration T from logs (e.g., 50 steps). If T * k is approx 2.5, size the fault penalty higher than 2.5 (e.g., -5.0) so dirty racing is a net loss.
- **Concrete Rules:**
  - **Follower Fault (Rear-end):** Contact normal in the leader body frame is within +-45 deg of the rear longitudinal axis.
  - **Leader Fault (Brake-check/Swerve):** One-move lookback: If the leader moved laterally toward the follower by more than 0.1 track widths within the last N=10 steps prior to contact, the leader is at fault.
  - **Racing Incident (Neutral):** Contact near the +-45 deg boundary, side-by-side rubbing, or simultaneous swerving results in the baseline penalty only (no fault).

---

## 3. Observation Space Enhancements

Expand the observation space from 15 dims to 19 dims, zero-padding the new inputs during the warm-start.
- **Drop:** Rear-facing LiDAR rays (redundant).
- **Add (All in Ego-Frame):**
  1. `opp_lateral_velocity`: To predict opponent trajectory.
  2. `opp_relative_heading`: To anticipate opponent cornering lines.
  3. `is_drafting_flag`: Continuous [0, 1] based on draft fade.
  4. `is_being_drafted_flag`: Continuous [0, 1] based on draft fade.
- **Note:** Keep speed normalization on the base `MAX_SPEED` so the raised drafting cap (which will push normalized speed > 1.0) does not shift the input scale.

---

## 4. Training Curriculum & Empirical Gates

### Step 4.1: The Physical Baseline
- **Active:** OBB Collisions, momentum transfer, small symmetric contact penalty (-0.1 applied once per contact event), 19-dim observations (drafting flags zeroed), enable_draft=False.
- **Warm-start:** Load Phase 3 final (`checkpoints/multi-A/slipstream-multi-A-84a9ee1/final.zip`).
- **Measured OBB Baseline (20 episodes):**
  - Deterministic: 6.45 contact events/ep, 316.60 steps in contact/ep, 0.0125 col-crashes/1k, 0.0125 solo-crashes/1k, pace L=2.7201 / F=2.7154 / pair=2.7178.
  - Stochastic: 6.80 contact events/ep, 306.45 steps in contact/ep, 0.0625 col-crashes/1k, 0.0125 solo-crashes/1k, pace L=2.7025 / F=2.6921 / pair=2.6973.
- **Gate to 4.2 (Empirical Criteria):**
  1. **Contact Events / Episode:** At most 50% of the measured OBB baseline:
     - Deterministic: <= 3.22 contact events/ep
     - Stochastic: <= 3.40 contact events/ep
  2. **Collision-Induced Crash Rate:** <= 0.10 crashes per 1,000 agent-steps in both modes.
  3. **Solo Crash Rate:** <= 0.10 crashes per 1,000 agent-steps in both modes.
  4. **Pace Retention:** At least 95% of measured baseline pace in both modes:
     - Deterministic pair pace: >= 2.5819 laps/1k steps (Leader >= 2.5841, Follower >= 2.5796)
     - Stochastic pair pace: >= 2.5624 laps/1k steps (Leader >= 2.5674, Follower >= 2.5575)

### Step 4.2: The Draft & Position Incentive
- **Active:** Step 4.1 + Slipstream physics + Continuous Positional Rewards.
- **Gate to 4.3:** Agents demonstrate on-track overtakes (e.g., > 1 pass per 10 laps) and leader blocking behavior emerges without the overall crash rate exploding.

### Step 4.3: Clean Racing (Fault Penalties)
- **Active:** Step 4.2 + Severe Fault Penalties.
- **Goal:** Unlearn dirty passes/blocks developed in 4.2.

---

## 5. Exit Criteria & Evaluation

Provide evidence that competitive racing has been achieved:
1. **Overtaking (Draft Ablation):** The Follower executes >= 1 on-track pass (no respawn-induced rank changes) per 10 laps. Ablation test: Passes must drop significantly when the slipstream is turned off in eval.
2. **Defending:** Compare the leader lateral deviation against a Phase 3 solo control. The deviation must be significantly higher and directed toward the follower lateral position.
3. **Cleanliness:** Leader at-fault collisions and Follower at-fault collisions must independently average < 1.0 per episode.
4. **Pace:** The pair-average pace must remain >= 2.65 laps/1k steps under the new OBB physics.
5. **Cross-Play Verification:**
   - Eval Phase 4 Follower vs. Frozen Phase 3 Leader (Hypothesis: Phase 4 successfully overtakes).
   - Eval Phase 4 Leader vs. Frozen Phase 3 Follower (Hypothesis: Phase 4 successfully defends).
   - Use multiple seeds, both deterministic and stochastic eval modes.
