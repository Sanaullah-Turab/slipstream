# Slipstream: Multi-Agent Reinforcement Learning for Competitive Racing

**Author:** Sanaullah Turab  
**Course:** Reinforcement Learning Semester Project  
**Tech Stack:** Python, Gymnasium, PettingZoo, Stable-Baselines3 (PPO), Pygame, Weights & Biases

## Overview
Slipstream is a multi-agent reinforcement learning (MARL) research project focused on training autonomous racing agents to exhibit competitive behavior—such as overtaking, defending, and adapting—on a custom 2D track. 

Rather than relying on scripted racing lines or hardcoded rules, agents in Slipstream learn dynamic racecraft entirely through reward feedback. The project is designed with a strict MLOps and software engineering mindset, utilizing configuration-driven experiments, isolated reward testing, and remote telemetry logging. 

**The progression pipeline includes:**
1. **Custom Environment:** A deterministic, Gymnasium-compatible 2D racing track with compact vector observations.
2. **Single-Agent Baseline:** Training a highly consistent PPO driver to optimize the racing line and achieve fast lap times.
3. **Multi-Agent Competition:** Wrapping the environment in PettingZoo to introduce a secondary agent, forcing the emergence of interaction-aware competitive behavior.

## Repository Architecture 
This repository separates environment logic, reward functions, and training configurations to ensure that production code remains clean and locally testable before deploying heavy PPO runs to cloud GPUs.