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
- **Mechanic:** Define a rear wake cone oriented by the leader heading with half-angle 0.26 rad (~15 deg) and total length 150.0 units. Draft intensity is continuous in [0, 1], exactly 0 at contact and below minimum gap 30.0 units, ramps linearly to 1.0 at peak gap 45.0 units, and decays linearly to 0 at 150.0 units. Lateral fade decreases linearly from 1.0 at wake center to 0 at the cone boundary `w = CAR_HALF_WIDTH + d_long * tan(0.26)`.
- **Implementation:** Reduce drag by up to 40% (multiplier 0.6) and raise the speed ceiling by +15.0 u/s (from 150.0 up to 165.0 u/s) proportional to draft intensity. Exiting the cone decays speed through aerodynamic drag rather than snapping to 150.0.
- **Calibration (Scripted Test):** Build a scripted setup (Leader on solo line, Follower inside zone) and measure how much gap the Follower closes on the longest straight. Tune the boost so a pass is possible on some laps, but not guaranteed.

---

## 2. Reward System Redesign

### A. Continuous Zero-Sum Positional Reward
To avoid flickering signals when cars are side-by-side:
- **Formula:** `r_pos = k * clip((dist_ego - dist_opp) / g0, -1, 1)` where `g0 = 50.0` units (~2 car lengths) and `k = 0.03`.
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
- **Active:** OBB Collisions, momentum transfer, small symmetric contact penalty (-0.1 applied once per contact event, -0.02 per contact step), 19-dim observations (drafting flags zeroed), enable_draft=False.
- **Warm-start:** Load Phase 3 final (`checkpoints/multi-A/slipstream-multi-A-84a9ee1/final.zip`).
- **Gate to 4.2 (Infrastructure Gate):**
  1. **Zero Initial Overlap:** Zero overlap at step 0 and step 1 for all valid spawn offsets.
  2. **Safe Respawn:** Respawns never overlap the opponent vehicle.
  3. **Collision Invariants:** SAT collision detection and inelastic impulse resolution hold without tunneling or physics explosions.
  4. **Deterministic Crash Rates:** Collision crash rate <= 0.05 and solo crash rate <= 0.05 per 1,000 agent-steps in deterministic mode.
  5. **Pace Retention:** At least 95% of baseline pace preserved in deterministic mode under the current geometry (car 26x11, track width 70).
  6. **Contact Reporting:** Contact events and steps are tracked and reported as baseline reference, not gating progression to 4.2.
- **Starting Checkpoint for 4.2:** `checkpoints/phase4/slipstream-p4-1b-7b1372f/final.zip` (4.1b final).
- **Pooled Re-baseline Reference (seeds 1000-1049 & 2000-2049, 100 episodes total, car 26x11, track width 70):**
  - Phase 3 Baseline:
    - Deterministic: 5.64 events/ep, 238.77 steps in contact/ep, 0.0150 col-crashes/1k, 0.1300 solo-crashes/1k, pair pace 2.6977 laps/1k steps.
    - Stochastic: 6.30 events/ep, 168.28 steps in contact/ep, 0.0150 col-crashes/1k, 0.0175 solo-crashes/1k, pair pace 2.6974 laps/1k steps.
  - 4.1b Final (`slipstream-p4-1b-7b1372f`):
    - Deterministic: 8.21 events/ep, 368.02 steps in contact/ep, 0.0000 col-crashes/1k, 0.0000 solo-crashes/1k, pair pace 2.7067 laps/1k steps.
    - Stochastic: 6.22 events/ep, 198.22 steps in contact/ep, 0.0050 col-crashes/1k, 0.0650 solo-crashes/1k, pair pace 2.6899 laps/1k steps.

### Step 4.2: The Draft & Position Incentive
- **Active:** Step 4.1 + Slipstream drafting physics (wake cone drag reduction and speed ceiling boost) + Continuous zero-sum positional reward + Per-step contact penalty (-0.02) + Position swap tracking.
- **Starting Checkpoint:** 4.1b final (`slipstream-p4-1b-7b1372f/final.zip`).
- **Pre-registered Gate to 4.3 (Pooled over seeds 1000-1049 and 2000-2049):**
  1. **Steps in Contact / Episode:** <= 184 in deterministic mode and <= 99 in stochastic mode.
  2. **Contact Events / Episode:** <= 5.64 in deterministic mode and <= 6.30 in stochastic mode.
  3. **Crash Rates:** Collision crash rate <= 0.05 and solo crash rate <= 0.05 per 1,000 agent-steps in deterministic mode.
  4. **Pace Retention:** Pair pace >= 2.57 laps/1k steps in deterministic mode.
  5. **Overtaking Performance:** Mean position swaps per episode > 0.5 in deterministic mode.

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
