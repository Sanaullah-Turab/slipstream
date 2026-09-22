# Slipstream

A multi-agent reinforcement learning project where autonomous racing agents learn competitive behavior on a custom 2D track. Agents are trained entirely through reward feedback -- no scripted racing lines, no hardcoded rules.

**Author:** Sanaullah Turab
**Course:** Reinforcement Learning Semester Project
**Stack:** Python, Gymnasium, PettingZoo, Stable-Baselines3 (PPO), Pygame, Weights and Biases

---

## What This Is

Slipstream trains agents to race, overtake, and defend on a lightweight 2D track. The project is staged:

1. **Custom Environment** -- a deterministic, Gymnasium-compatible track with vector observations and Pygame rendering.
2. **Single-Agent Baseline** -- a PPO driver trained to complete laps consistently and optimize the racing line.
3. **Multi-Agent Competition** -- the environment wrapped with PettingZoo, introducing a second agent and interaction-aware rewards to drive emergent competitive behavior.

---

## Repository Structure

```
slipstream/
|-- requirements.txt
|-- configs/
|   |-- env_config.yaml       # Environment parameters
|   `-- ppo_config.yaml       # PPO hyperparameters
|-- src/
|   |-- env/
|   |   |-- track.py              # Track geometry and reference path
|   |   |-- racing_env.py         # Gymnasium single-agent API
|   |   |-- multi_racing_env.py   # PettingZoo multi-agent wrapper
|   |   `-- rewards.py            # All reward components, isolated for testing
|   |-- training/
|   |   |-- train_single.py       # Single-agent PPO entry point
|   |   |-- train_multi.py        # Multi-agent PPO entry point
|   |   `-- callbacks.py          # Checkpointing and logging callbacks
|   |-- evaluation/
|   |   |-- evaluate.py           # Fixed-seed evaluation runner
|   |   `-- replay_recorder.py    # Episode recording for demos
|   `-- utils/
|       `-- config.py             # YAML config loader
|-- tests/
|   |-- test_env.py               # Environment reset, step, shapes, determinism
|   |-- test_rewards.py           # Reward signs, scales, and edge cases
|   `-- test_multi_env.py         # Multi-agent stepping and termination
|-- notebooks/                    # Exploration only, not production code
`-- scripts/
    `-- colab_train.ipynb         # Cloud training notebook
```

---

## Getting Started

```bash
git clone https://github.com/Sanaullah-Turab/slipstream.git
cd slipstream
pip install -r requirements.txt
```

---

## Running Tests

```bash
.venv/bin/pytest tests/
```

---

## Training

Single-agent:
```bash
python src/training/train_single.py --config configs/ppo_config.yaml
```

Multi-agent:
```bash
python src/training/train_multi.py --config configs/ppo_config.yaml
```

For GPU runs, use `scripts/colab_train.ipynb` on Google Colab or Kaggle.

---

## Roadmap

- [x] Project scaffold and repository setup
- [x] Track + Gymnasium environment with Pygame rendering
- [x] Reward system with isolated unit tests
- [x] Single-agent PPO baseline
- [x] Kinematic bicycle model car physics
- [ ] PettingZoo multi-agent wrapper
- [ ] Competitive training with interaction-aware rewards
- [ ] Evaluation suite, replay recording.