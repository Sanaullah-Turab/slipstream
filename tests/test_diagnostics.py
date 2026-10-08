import pytest
import numpy as np

from scripts.run_diagnostics import analyze_round2


def test_analyze_round2_metrics():
    episodes_data = [
        {
            "episode": 0,
            "events": [
                {
                    "start_step": 1,
                    "end_step": 100,
                    "duration": 100,
                    "step1": {"abs_long": 30.0, "abs_lat": 2.0, "rel_speed": 0.5},
                    "step20": {"abs_long": 31.0, "abs_lat": 2.5, "rel_speed": 5.0},
                },
                {
                    "start_step": 150,
                    "end_step": 170,
                    "duration": 21,
                    "step1": {"abs_long": 35.0, "abs_lat": 5.0, "rel_speed": 20.0},
                    "step20": {"abs_long": 36.0, "abs_lat": 5.5, "rel_speed": 21.0},
                },
            ],
            "total_cols": 2,
            "contact_steps": 121,
        },
        {
            "episode": 1,
            "events": [
                {
                    "start_step": 250,
                    "end_step": 310,
                    "duration": 61,
                    "step1": {"abs_long": 40.0, "abs_lat": 6.0, "rel_speed": 30.0},
                    "step20": {"abs_long": 38.0, "abs_lat": 6.2, "rel_speed": 28.0},
                }
            ],
            "total_cols": 1,
            "contact_steps": 61,
        },
    ]

    all_events = [ev for ep in episodes_data for ev in ep["events"]]
    separation_durations = [150 - (100 + 1)]  # 49

    res = analyze_round2(episodes_data, all_events, separation_durations)

    assert res["total_events"] == 3
    assert res["total_contact_steps"] == 182
    assert res["hist_start_steps"]["1"] == 1
    assert res["hist_start_steps"]["51_to_200"] == 1
    assert res["hist_start_steps"]["over_200"] == 1
    assert res["eps_with_contact_step1"] == 1
    assert pytest.approx(res["share_start_1"], abs=1e-4) == 100 / 182
    assert pytest.approx(res["share_start_first10"], abs=1e-4) == 100 / 182
    assert res["mean_excl_events"] == 1.0
    assert res["mean_excl_contact_steps"] == 41.0
    assert res["long_geom_first10"]["count"] == 1
    assert res["long_geom_later"]["count"] == 1
    assert res["sep_stats"]["median"] == 49.0
