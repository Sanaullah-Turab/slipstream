import numpy as np
import pytest

from scripts.evaluate_p4_1b import compute_bootstrap_ci, evaluate_model_on_block
from src.env.multi_racing_env import MultiRacingEnv


def test_compute_bootstrap_ci_identical_values():
    data = [5.0] * 50
    low, high = compute_bootstrap_ci(data, n_bootstraps=500, ci=0.95)
    assert np.isclose(low, 5.0)
    assert np.isclose(high, 5.0)


def test_compute_bootstrap_ci_bounds():
    rng = np.random.default_rng(42)
    data = rng.normal(10.0, 2.0, size=50).tolist()
    mean = float(np.mean(data))
    low, high = compute_bootstrap_ci(data, n_bootstraps=500, ci=0.95)
    assert low < mean < high
    assert low >= min(data)
    assert high <= max(data)


def test_format_results_markdown():
    from scripts.evaluate_p4_1b import format_results_markdown

    results = [
        {
            "checkpoint": "Phase 3 Baseline",
            "seeds": "1000-1049",
            "mode": "Deterministic",
            "mean_events": 5.2,
            "events_ci": (4.5, 5.9),
            "mean_steps": 120.4,
            "steps_ci": (100.2, 140.6),
            "col_crash_rate": 0.01,
            "solo_crash_rate": 0.0,
            "pair_pace": 2.71,
        }
    ]
    md = format_results_markdown(results)
    assert "| Checkpoint |" in md
    assert "Phase 3 Baseline" in md
    assert "5.20 [4.50, 5.90]" in md
    assert "120.40 [100.20, 140.60]" in md
    assert "0.0100" in md
    assert "2.7100" in md

