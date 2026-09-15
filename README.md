# Slipstream: Multi-Agent Reinforcement Learning for Competitive Racing

**Author:** Sanaullah Turab
**Course:** Reinforcement Learning Semester Project
**Repository:** github.com/Sanaullah-Turab/slipstream
**Project Type:** MARL research + software engineering
**Tech Stack:** Python, Gymnasium, PettingZoo, Stable-Baselines3 (PPO), Pygame, Weights and Biases

---

## Overview

Slipstream is a multi-agent reinforcement learning project where autonomous racing agents learn competitive behavior on a custom 2D track. The system is intentionally built in stages so each stage produces a working artifact that can be evaluated before the next stage begins.

The core idea: start with a stable single-agent PPO racer, then introduce multiple agents in the same environment and train for competitive behavior. The final system should demonstrate measurable overtaking and defensive behavior, not only lap completion.

Agents learn actions from reward feedback rather than scripted racing rules. The environment is custom-built with Gymnasium for single-agent use and exposed through PettingZoo for multi-agent use. Development runs locally on CPU; heavy training runs on Colab or Kaggle GPU.

### Why This Project

- Shows a clear progression: single-agent PPO -> multi-agent PPO -> emergent competitive behavior.
- Creates a visually strong final demo through Pygame rendering and recorded replays.
- Makes reward engineering a central technical contribution, including racing-line progress, collisions, overtaking, and defensive behavior.
- Keeps the environment lightweight and cloud-portable so training remains realistic within the available hardware constraints.

### Hardware and Compute Strategy

- **Local CPU:** build the environment, implement rewards, run pytest, inspect renders, and perform short sanity checks.
- **Google Colab / Kaggle GPU:** run full PPO experiments, multi-agent training, parameter sweeps, and long runs.
- **State design:** compact vector observations rather than raw pixels. The track is kept small enough for fast iteration and stable memory use.

---

## Project Plan: What We Will Build

The implementation is organized as a controlled sequence. Each stage must pass functional tests and produce evidence before the next stage starts.

| Stage | What We Build | How We Do It | Main Resources |
|---|---|---|---|
| 1. Track + environment | Custom 2D racing track, vehicle state, actions, collisions, lap logic, reset/step API | Deterministic track geometry in Python. Gymnasium-compatible environment. Pygame rendering. pytest checks for shapes, reset behavior, determinism, and collision logic | Python, Gymnasium, Pygame, pytest |
| 2. Reward system | Progress reward, racing-line reward, speed/heading incentives, collision penalty, lap completion reward | Reward functions kept separate from environment code. Test signs, scales, terminal rewards, and edge cases before training | Custom reward module, pytest, short local runs |
| 3. Single-agent PPO | A baseline driver that learns to complete laps consistently | Train PPO with vector observations. Conservative hyperparameters. Compare against random and rule-based baselines. Log every run and checkpoint | Stable-Baselines3, W&B, Colab/Kaggle |
| 4. Multi-agent environment | Two or more agents racing in one shared environment | Wrap the environment with PettingZoo. Validate simultaneous stepping, per-agent observations/actions, termination, collisions, and no deadlocks | PettingZoo, pytest, Pygame |
| 5. Competitive learning | Policies that learn to overtake, defend, and compete rather than merely finish | Interaction-aware rewards. Start with dense shaping, then reduce shaping as behavior stabilizes. Compare trained agents head-to-head and against the single-agent baseline | Stable-Baselines3/PPO, PettingZoo, W&B, GPU |
| 6. Evaluation + demo | Reproducible evaluation suite, replay clips, charts, and final demonstration | Fixed evaluation seeds. Measure lap time, win rate, collision rate, overtakes, and reward curves. Record representative episodes and compile the final report | Evaluation scripts, Pygame recorder, W&B, report tooling |

### Definition of Done

- Each stage must work, pass tests, be reproducible, and be committed before the next stage begins.
- Final claims must be supported by quantitative evaluation, not visual intuition alone.

---

## Software Architecture and Repository

The repository keeps environment logic, rewards, training, evaluation, and experiment configuration separate. This is important because local development and cloud training use the same production code.

```
slipstream/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- env_config.yaml
|   `-- ppo_config.yaml
|-- src/
|   |-- env/
|   |   |-- track.py
|   |   |-- racing_env.py
|   |   |-- multi_racing_env.py
|   |   `-- rewards.py
|   |-- training/
|   |   |-- train_single.py
|   |   |-- train_multi.py
|   |   `-- callbacks.py
|   |-- evaluation/
|   |   |-- evaluate.py
|   |   `-- replay_recorder.py
|   `-- utils/
|       `-- config.py
|-- tests/
|   |-- test_env.py
|   |-- test_rewards.py
|   `-- test_multi_env.py
|-- notebooks/
`-- scripts/
    `-- colab_train.ipynb
```

### Module Responsibilities

- **track.py:** generate the geometry and reference path used by the agents.
- **racing_env.py:** Gymnasium single-agent API with reset, step, state transitions, termination, and rendering hooks.
- **multi_racing_env.py:** PettingZoo interface for 2 or more agents and shared-world interactions.
- **rewards.py:** all reward components, isolated so they can be unit-tested and tuned independently.
- **training/:** training entry points, callbacks, checkpointing, and logging.
- **evaluation/:** fixed-seed evaluation and replay recording for the final analysis.
- **configs/:** versioned YAML files so experiments change parameters without editing training code.
- **tests/:** regression protection before expensive GPU runs.

### Key Engineering Rules

- Notebooks are for exploration only. Production logic remains in `src/`.
- No hardcoded experiment settings inside training scripts.
- Every training checkpoint is tied to a W&B run ID and Git commit hash.
- `main` stays runnable and demoable at every milestone.

---

## Tools, Libraries, and Resources

The stack is deliberately conventional and lightweight so the engineering effort stays focused on reinforcement learning rather than infrastructure.

| Area | Resource | Why We Use It |
|---|---|---|
| Language | Python | Primary implementation and experiment language |
| Environment | Gymnasium + PettingZoo | Standard single-agent and multi-agent environment interfaces |
| Rendering | Pygame | Track visualization, debugging, and replay capture |
| RL algorithm | Stable-Baselines3, PPO | Reliable PPO implementation so the project can focus on environment and reward design |
| Configuration | YAML + custom loader | Versioned, reproducible experiment settings |
| Tracking | Weights and Biases | Remote run logging, charts, config capture, and comparison across experiments |
| Testing | pytest | Fast regression tests for environment and reward logic |
| Version control | Git + GitHub | Feature branches, review points, and recoverable project history |
| Compute | Local CPU + Google Colab / Kaggle GPU | Local iteration plus cloud training for expensive experiments |

### External Resources

- Official Gymnasium documentation for environment API behavior and validation.
- Official PettingZoo documentation for the multi-agent API and environment testing expectations.
- Stable-Baselines3 documentation for PPO configuration and callbacks.
- Weights and Biases documentation for run logging and artifact/checkpoint tracking.
- Google Colab or Kaggle notebook runtimes for GPU training when local CPU time becomes impractical.
- Course lecture notes and references for reinforcement learning theory, PPO, reward design, exploration, and multi-agent learning.

Community examples may be used for debugging, but the project report will distinguish our own implementation choices from borrowed patterns.

---

## Git Workflow and Experiment Protocol

The repository and the experiment process are treated as one system. A result should always be traceable to code, configuration, data, and a checkpoint.

### Git Workflow

- **main:** stable, working, and demoable.
- **feature/env-setup:** Phase 1 environment work.
- **feature/single-agent-ppo:** Phase 2 baseline training.
- **feature/multi-agent-ppo:** Phase 3 multi-agent extension.
- **feature/competitive-rewards:** Phase 4 reward iteration and competition experiments.

### Experiment Protocol

| Step | Action | Evidence Captured |
|---|---|---|
| 1 | Create or modify a YAML experiment config | Config file committed to Git |
| 2 | Run tests and a short sanity check locally | pytest output, render check, basic reward trace |
| 3 | Run the experiment on CPU or cloud GPU depending on cost | W&B run, Git commit hash, phase tag, system notes |
| 4 | Save checkpoints at fixed intervals | Checkpoint artifact + run ID |
| 5 | Evaluate with fixed seeds and compare against baselines | Lap time, win rate, collision rate, overtakes |
| 6 | Decide whether to keep, reject, or iterate the change | Brief experiment note linked to the run |

### Baseline Comparisons

- **Random policy:** establishes a floor for performance.
- **Simple rule-based policy:** provides a non-learning reference for lap completion.
- **Single-agent PPO:** establishes learned driving ability before competition is introduced.
- **Competitive PPO agents:** evaluated head-to-head to test whether interaction creates overtaking and defensive behavior.

---

## Evaluation, Risks, and Final Submission

The final report should answer one central question: did the agents learn useful competitive racing behavior, and can we prove it with controlled experiments?

### Evaluation Metrics

| Metric | What It Tells Us |
|---|---|
| Lap time | Whether the policy learned effective racing rather than merely surviving |
| Win rate | Competitive performance in head-to-head evaluation |
| Collision rate | Whether agents race cleanly or exploit crashing behavior |
| Overtakes per episode | Direct evidence that interaction produces the target behavior |
| Reward curve | Learning stability and convergence for the report |

### Main Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Local hardware is too slow for multi-agent training | Move heavy runs to Colab/Kaggle GPU and keep the environment lightweight |
| Reward function is silently wrong | Unit-test reward components before GPU training and inspect reward traces |
| Competitive behavior fails to emerge | Use dense shaping first, then anneal toward a more meaningful win/lose objective |
| Cloud session limits interrupt runs | Checkpoint regularly and resume from the latest saved artifact |
| Scope grows too large | Keep vector observations, small tracks, limited agents, and PPO as hard scope constraints |

### Semester Timeline

| Weeks | Milestone |
|---|---|
| 1-2 | Phase 1 complete: track, environment, rendering, reward tests |
| 3-5 | Phase 2 complete: single-agent PPO baseline trained and logged |
| 6-8 | Phase 3 complete: PettingZoo multi-agent wrapper works end-to-end |
| 9-12 | Phase 4: competitive training, reward iteration, and metric collection |
| 13-14 | Final evaluation, replay recording, report, and presentation preparation |

### Final Deliverables

- GitHub repository with feature-branch history merged to main.
- Single-agent and final multi-agent checkpoints.
- Replay clips showing overtaking and defensive behavior.
- W&B dashboard or exported training charts.
- Written report covering formulation, design, training, results, and limitations.
- Live or recorded final demonstration.

---

## Phase 1 Kickoff Checklist

The first milestone is deliberately small. The goal is to establish a trustworthy environment before any serious learning run is attempted.

### Day 1: Repository and Tooling

- [ ] Create the GitHub repository named slipstream and initialize the local Git repository.
- [ ] Create the folder structure from the architecture section and add the initial README, requirements file, YAML configs, source folders, and tests.
- [ ] Create `feature/env-setup` and make the first baseline commit.

### Day 2-3: Environment Skeleton

- [ ] Implement the track representation in `track.py` with a compact state representation.
- [ ] Implement Gymnasium-compatible `reset()` and `step()` behavior in `racing_env.py`.
- [ ] Add Pygame rendering so we can visually inspect the vehicle, track boundaries, progress, and collisions.

### Day 4-5: Reward and Tests

- [ ] Implement reward components in `rewards.py`, starting with forward progress and safe track driving.
- [ ] Add pytest coverage for reset/step behavior, observation and action shapes, deterministic seeds, collision handling, and reward signs.
- [ ] Run a random policy locally and inspect several episodes before beginning PPO training.

### Phase 1 Acceptance Gate

Phase 1 is approved only when the environment renders correctly, random episodes complete without crashes in the software, core tests pass, and reward values can be explained step by step. At that point the branch can be merged to main and Phase 2 can begin.

**Next milestone:** train the first single-agent PPO baseline and establish the initial lap-time benchmark.