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
|-- configs/              # Versioned YAML experiment configs
|-- src/
|   |-- env/              # Track geometry, Gymnasium env, PettingZoo wrapper, rewards
|   |-- training/         # Training entry points and callbacks
|   |-- evaluation/       # Fixed-seed evaluation and replay recording
|   `-- utils/            # Config loader
|-- tests/                # pytest coverage for env, rewards, and multi-agent logic
|-- scripts/              # Colab training notebook
`-- notebooks/            # Exploration only, not production code
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
pytest tests/
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
- [ ] Track + Gymnasium environment with Pygame rendering
- [ ] Reward system with isolated unit tests
- [ ] Single-agent PPO baseline
- [ ] PettingZoo multi-agent wrapper
- [ ] Competitive training with interaction-aware rewards
- [ ] Evaluation suite, replay recording, and final report