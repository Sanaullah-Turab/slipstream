# Contact and Crash Re-baseline (Current Geometry)

Geometry: car 26x11 (half_len=13.0, half_width=5.5), track_width=70.0, spawn_offset_idx=15.
Evaluation: 50 episodes per block, 2000 max steps, 10000-sample bootstrap 95% CIs.

| Checkpoint | Seed Range | Mode | Contact Events / ep (95% CI) | Contact Steps / ep (95% CI) | Col Crash / 1k | Solo Crash / 1k | Pair Pace (laps/1k) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Phase 3 Baseline | 1000-1049 | Deterministic | 5.82 [5.00, 6.62] | 219.64 [171.02, 272.66] | 0.0150 | 0.1450 | 2.6965 |
| Phase 3 Baseline | 1000-1049 | Stochastic | 6.54 [5.28, 7.86] | 157.76 [119.74, 198.94] | 0.0150 | 0.0050 | 2.6981 |
| Phase 3 Baseline | 2000-2049 | Deterministic | 5.46 [4.76, 6.20] | 257.90 [196.72, 326.78] | 0.0150 | 0.1150 | 2.6989 |
| Phase 3 Baseline | 2000-2049 | Stochastic | 6.06 [4.62, 7.62] | 178.80 [119.98, 251.42] | 0.0150 | 0.0300 | 2.6967 |
| slipstream-p4-1b-7b1372f final | 1000-1049 | Deterministic | 8.08 [7.46, 8.70] | 348.26 [287.28, 412.20] | 0.0000 | 0.0000 | 2.7069 |
| slipstream-p4-1b-7b1372f final | 1000-1049 | Stochastic | 6.50 [5.08, 7.98] | 220.54 [160.22, 287.76] | 0.0050 | 0.0750 | 2.6891 |
| slipstream-p4-1b-7b1372f final | 2000-2049 | Deterministic | 8.34 [7.78, 8.90] | 387.78 [320.60, 457.84] | 0.0000 | 0.0000 | 2.7065 |
| slipstream-p4-1b-7b1372f final | 2000-2049 | Stochastic | 5.94 [4.80, 7.16] | 175.90 [129.38, 227.00] | 0.0050 | 0.0550 | 2.6907 |
