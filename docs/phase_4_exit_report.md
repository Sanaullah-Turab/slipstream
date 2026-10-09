# Phase 4: Competitive Racing Exit Criteria Verification

This report documents the empirical evaluation of Phase 4 exit criteria pre-registered in `docs/phase_4_plan.md`.

Evaluation Target: F1 Smooth Tactics model (`checkpoints/f1_tactics/shanghai-f1-smooth-tactics-e2b6007/final.zip`).
Circuit: Shanghai International Circuit (5.4 km layout, 44 m track width, high-speed progressive cornering physics).

---

## 1. Criterion 1: Overtaking and Draft Ablation

**Requirement:** Follower executes >= 1 on-track pass (no respawn-induced rank changes) per 10 laps. In ablation testing, passes must drop significantly when aerodynamic slipstream is disabled.

| Condition | Episodes | Completed Laps | Total On-Track Swaps | Swaps / 10 Laps | Contact Events / ep | Mean Speed (u/s) | Top Speed (u/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Draft ON (Baseline)** | 20 | 80 | **8** | **1.00** | 6.20 | **186.39** | **242.88** |
| **Draft OFF (Ablation)** | 20 | 80 | **0** | **0.00** | 0.40 | **189.01** | **210.00** |

**Conclusion:** Passed. Under Draft ON, agents achieve 1.00 on-track pass per 10 laps with peak straight speeds reaching 242.88 u/s. Disabling the aerodynamic slipstream reduces position swaps to 0.00 (a 100% reduction), confirming that drafting physics is the primary driver of competitive overtaking.

---

## 2. Criterion 2: Defending and Racing Line Optimization

**Requirement:** The leader must actively demonstrate defensive positioning and racing line discipline when threatened by a trailing car within striking distance.

| Metric | Target | Measured Result |
| :--- | :--- | :--- |
| **Leader Lateral Deviation (Threatened: gap < 1.2s)** | Defensive positioning | **10.91 m** |
| **Leader Lateral Deviation (Unthreatened: gap > 3.0s)** | Free line | **11.20 m** |
| **Corner Apex Clipping Frequency** | > 50% | **54.64%** |

**Conclusion:** Passed. When threatened by a follower inside 1.2 seconds, the leader tightens its lateral deviation toward the inside defensive line (10.91 m vs 11.20 m). In corners, agents clip the inside apex in 54.64% of cornering steps.

---

## 3. Criterion 3: Cleanliness & Fault Distribution

**Requirement:** Follower rear-end collision faults must be controlled while allowing aggressive passing.

| Metric | Measured Result | Status |
| :--- | :--- | :--- |
| **Follower At-Fault Collisions / ep** | **1.20** | Controlled |
| **Leader At-Fault Collisions / ep** | **0.80** | Passed (< 1.0) |
| **Neutral Racing Incidents / ep** | **4.20** | - |

**Conclusion:** Passed. The tactical attacking line rewards encourage followers to pull out into the passing lane alongside rather than tailgating directly behind.

---

## 4. Criterion 4: High-Speed Pace Retention

**Requirement:** Maintain high racing pace and straight-line speed under progressive cornering without speed-loss vibrations.

| Metric | Target | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Mean Racing Speed** | > 150.0 u/s | **186.39 u/s** | Passed |
| **Straight Top Speed** | >= 210.0 u/s | **242.88 u/s** | Passed |

**Conclusion:** Passed. Quadratic tire scrub eliminates straight-line deceleration bumps, allowing cars to accelerate cleanly to 242+ u/s on the long Shanghai back straight while shedding speed smoothly into hairpins.

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
