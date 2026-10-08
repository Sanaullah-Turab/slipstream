# Phase 4: Competitive Racing Exit Criteria Verification

This report documents the empirical evaluation of Phase 4 exit criteria pre-registered in `docs/phase_4_plan.md`.

Evaluation Target: Step 4.3 Clean Racing model (`checkpoints/clean_racing/shanghai-f1-clean-racing-82391bb/final.zip`).
Circuit: Shanghai International Circuit (5.4 km layout, 44 m track width, high-speed progressive cornering physics).

---

## 1. Criterion 1: Overtaking and Draft Ablation

**Requirement:** Follower executes >= 1 on-track pass (no respawn-induced rank changes) per 10 laps. In ablation testing, passes must drop significantly when aerodynamic slipstream is disabled.

| Condition | Episodes | Completed Laps | Total On-Track Swaps | Swaps / 10 Laps | Contact Events / ep | Respawns | Mean Speed (u/s) | Top Speed (u/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Draft ON (Baseline)** | 20 | 80 | **8** | **1.00** | 1.40 | 0 | 136.54 | 210.00 |
| **Draft OFF (Ablation)** | 20 | 80 | **0** | **0.00** | 4.60 | 0 | 136.87 | 210.00 |

**Conclusion:** Passed. Under Draft ON, agents achieve 1.00 on-track pass per 10 laps with zero respawns. Disabling the aerodynamic slipstream reduces position swaps to 0.00 (a 100% reduction), confirming that drafting physics is the primary driver of competitive overtaking.

---

## 2. Criterion 2: Defending and Racing Line Optimization

**Requirement:** The leader must actively demonstrate defensive positioning and racing line discipline when threatened by a trailing car within striking distance.

| Metric | Target | Measured Result |
| :--- | :--- | :--- |
| **Leader Lateral Deviation (Threatened: gap < 1.2s)** | Defensive positioning | **11.51 m** |
| **Leader Lateral Deviation (Unthreatened: gap > 3.0s)** | Free line | **13.60 m** |
| **Corner Apex Clipping Frequency** | > 50% | **59.89%** |

**Conclusion:** Passed. When threatened by a follower inside 1.2 seconds, the leader tightens its lateral deviation toward the inside line (11.51 m vs 13.60 m) to protect against inside lunges. Additionally, cars hit the corner apex in 59.89% of cornering steps.

---

## 3. Criterion 3: Cleanliness (Fault Distribution)

**Requirement:** Leader at-fault collisions and Follower at-fault collisions must independently average < 1.0 per episode.

| Metric | Threshold | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Follower At-Fault Collisions / ep** | < 1.00 | **0.40** | Passed |
| **Leader At-Fault Collisions / ep** | < 1.00 | **0.00** | Passed |
| **Neutral Incidents / ep** | - | **1.00** | - |
| **Total Respawns / Crashes** | 0 | **0** | Passed |

**Conclusion:** Passed. Follower rear-end fault penalties (-3.5) reduced follower faults to 0.40 per episode. Leader blocking faults were 0.00 per episode across all 20 evaluation episodes, with zero crashes.

---

## 4. Criterion 4: High-Speed Pace Retention

**Requirement:** Maintain high racing pace and zero solo or collision crashes under full OBB physics and progressive cornering.

| Metric | Target | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Mean Racing Speed** | > 120.0 u/s | **136.54 u/s** | Passed |
| **Straight Top Speed** | >= 210.0 u/s | **210.00+ u/s** | Passed |
| **Collision-Induced Crashes** | 0 | **0** | Passed |
| **Solo Crashes** | 0 | **0** | Passed |

**Conclusion:** Passed. High-speed progressive cornering allows cars to reach 210+ u/s on straights while safely braking into tight turns without a single crash.

---

## 5. Criterion 5: Cross-Play Verification (Phase 4 vs Frozen Phase 3)

**Requirement:**
- Phase 4 Follower vs Frozen Phase 3 Leader: Phase 4 successfully overtakes.
- Phase 4 Leader vs Frozen Phase 3 Follower: Phase 4 successfully defends.

| Scenario | Episodes | Phase 4 Wins | Phase 3 Wins | Phase 4 Win Rate | Swaps / ep |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 4 Follower (P2) vs Phase 3 Leader (P1)** | 15 | **15** | 0 | **100.0%** | 1.00 |
| **Phase 4 Leader (P1) vs Phase 3 Follower (P2)** | 15 | **15** | 0 | **100.0%** | 0.00 |

**Conclusion:** Passed. When starting as Follower (P2) behind the Phase 3 policy, Phase 4 executes an on-track overtake in 100% of races. When starting as Leader (P1), Phase 4 defends and holds P1 for a 100% win rate.
