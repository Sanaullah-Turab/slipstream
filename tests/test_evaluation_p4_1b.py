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
