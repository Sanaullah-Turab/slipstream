from pathlib import Path
from unittest.mock import MagicMock
from src.training.callbacks import CheckpointCallback


def test_checkpoint_callback_keeps_all_checkpoints_by_default(tmp_path):
    cb = CheckpointCallback(save_freq=10, save_dir=str(tmp_path))
    assert cb.keep_last is None

    mock_model = MagicMock()
    def fake_save(path):
        Path(path).with_suffix(".zip").write_text("model_data")
    mock_model.save.side_effect = fake_save
    cb.model = mock_model

    for step in [10, 20, 30, 40]:
        cb.num_timesteps = step
        cb._on_step()

    for step in [10, 20, 30, 40]:
        assert (tmp_path / f"model_{step}.zip").exists()
        assert (tmp_path / f"meta_{step}.json").exists()


def test_checkpoint_callback_prunes_when_keep_last_specified(tmp_path):
    cb = CheckpointCallback(save_freq=10, save_dir=str(tmp_path), keep_last=2)
    assert cb.keep_last == 2

    mock_model = MagicMock()
    def fake_save(path):
        Path(path).with_suffix(".zip").write_text("model_data")
    mock_model.save.side_effect = fake_save
    cb.model = mock_model

    for step in [10, 20, 30]:
        cb.num_timesteps = step
        cb._on_step()

    assert not (tmp_path / "model_10.zip").exists()
    assert (tmp_path / "model_20.zip").exists()
    assert (tmp_path / "model_30.zip").exists()


def test_training_episode_callback_records_metrics(tmp_path):
    import numpy as np
    from src.training.callbacks import TrainingEpisodeCallback

    log_file = tmp_path / "episodes.json"
    cb = TrainingEpisodeCallback(log_path=log_file)
    cb.num_timesteps = 100

    cb.locals = {
        "infos": [
            {"respawn": True, "respawns": 1, "collision": False, "collision_count": 0, "steps_in_contact": 0},
            {"respawn": False, "respawns": 0, "collision": False, "collision_count": 0, "steps_in_contact": 0},
        ],
        "dones": np.array([False, False]),
    }
    cb._on_step()

    for _ in range(4):
        cb.locals = {
            "infos": [
                {"respawn": False, "respawns": 1, "collision": False, "collision_count": 0, "steps_in_contact": 0},
                {"respawn": False, "respawns": 0, "collision": False, "collision_count": 0, "steps_in_contact": 0},
            ],
            "dones": np.array([False, False]),
        }
        cb._on_step()

    cb.locals = {
        "infos": [
            {"respawn": False, "respawns": 1, "collision": True, "collision_count": 1, "steps_in_contact": 1},
            {"respawn": False, "respawns": 0, "collision": True, "collision_count": 1, "steps_in_contact": 1},
        ],
        "dones": np.array([False, False]),
    }
    cb._on_step()

    cb.locals = {
        "infos": [
            {"respawn": False, "respawns": 1, "collision": False, "collision_count": 1, "steps_in_contact": 10},
            {"respawn": False, "respawns": 0, "collision": False, "collision_count": 1, "steps_in_contact": 10},
        ],
        "dones": np.array([True, True]),
    }
    cb._on_step()
    cb._on_training_end()

    assert len(cb.episode_records) == 1
    rec = cb.episode_records[0]
    assert rec["step_contact_penalty"] == 10 * (-0.02)
    assert rec["event_contact_penalty"] == 1 * (-0.1)
    assert rec["respawns"] == 1
    assert rec["events_near_respawn"] == 1
    assert log_file.exists()


def test_multi_eval_callback_logs_position_swaps_and_pair_pace(monkeypatch):
    import numpy as np
    from src.training.callbacks import MultiEvalCallback

    logged_data = {}
    monkeypatch.setattr("wandb.log", lambda data, step: logged_data.update(data))
    monkeypatch.setattr("wandb.run", MagicMock())

    cb = MultiEvalCallback(eval_freq=1, n_episodes=1)
    cb.num_timesteps = 1
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.zeros((2, 2), dtype=np.float32), None)
    cb.model = mock_model
    cb._on_training_start()
    cb._on_step()
    cb._on_training_end()

    assert "eval/mean_position_swaps_per_ep" in logged_data
    assert "eval/pair_pace" in logged_data


