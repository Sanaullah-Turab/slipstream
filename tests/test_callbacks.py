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
