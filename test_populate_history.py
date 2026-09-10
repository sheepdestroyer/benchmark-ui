import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import populate_history
from populate_history import populate, main


@pytest.fixture
def temp_history_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with patch.object(populate_history, "HISTORY_DIR", tmp_path):
            yield tmp_path


def test_populate_generates_expected_runs(temp_history_dir):
    """Test that populate(force=False) generates 24 run files with valid structure."""
    populate(force=False)

    gitkeep = temp_history_dir / ".gitkeep"
    assert gitkeep.exists()

    run_files = list(temp_history_dir.glob("run_*.json"))
    assert len(run_files) == 24

    expected_profiles = {
        "Qwen3.6-27B",
        "Qwen3.6-27B-spec",
        "Qwen3.6-27B-spec2",
        "Qwen3.6-27B-spec3",
        "Qwen3.6-35B-A3B",
        "Qwen3.6-35B-A3B-spec",
    }
    expected_quants = {"f16", "q8_0", "q5_1", "q4_0"}

    seen_profiles = set()
    seen_quants = set()

    for run_file in run_files:
        with open(run_file, "r") as f:
            data = json.load(f)

        # Verify 5 mandatory top-level sections
        assert "run_metadata" in data
        assert "model_settings" in data
        assert "throughput_metrics" in data
        assert "reasoning_accuracy" in data
        assert "quantization_loss" in data

        # Validate run_metadata
        metadata = data["run_metadata"]
        assert "timestamp" in metadata
        assert metadata["target_endpoint"] == "http://127.0.0.1:8081"
        assert metadata["synthetic"] is True
        assert isinstance(metadata["cli_arguments"], list)
        assert "--mode" in metadata["cli_arguments"]

        # Validate model_settings
        settings = data["model_settings"]
        profile_alias = settings["profile_alias"]
        seen_profiles.add(profile_alias)
        assert profile_alias in expected_profiles
        assert settings["base_quantization"] == "Q4_K_S"
        kv_quant = settings["kv_cache_quant"]
        seen_quants.add(kv_quant)
        assert kv_quant in expected_quants
        assert settings["threads"] == 16
        assert settings["ubatch_size"] == 512
        assert settings["batch_size"] == 2048
        assert settings["flash_attn"] == "true"
        assert settings["parallel"] == "1"
        assert settings["fit"] == "true"

        # Validate throughput_metrics
        throughput = data["throughput_metrics"]
        assert isinstance(throughput["prefill_speed"], float)
        assert throughput["prefill_speed"] > 0
        assert isinstance(throughput["decode_speed"], float)
        assert throughput["decode_speed"] > 0
        assert isinstance(throughput["ttft"], float)
        assert throughput["ttft"] > 0

        # Validate reasoning_accuracy
        accuracy = data["reasoning_accuracy"]
        for benchmark in ["needle", "ruler", "longbench", "swe_bench"]:
            assert accuracy[benchmark] in ["Pass", "Fail"]

        # Validate quantization_loss
        loss = data["quantization_loss"]
        assert isinstance(loss["perplexity"], float)
        assert isinstance(loss["mean_kld"], float)
        assert isinstance(loss["same_top_match_percent"], (float, int))

    assert seen_profiles == expected_profiles
    assert seen_quants == expected_quants


def test_populate_preserves_existing_files_when_force_is_false(temp_history_dir):
    """Test that populate(force=False) does not delete existing run_*.json files."""
    dummy_run = temp_history_dir / "run_dummy_old.json"
    dummy_run.write_text(json.dumps({"dummy": True}))
    unrelated_file = temp_history_dir / "note.txt"
    unrelated_file.write_text("important note")

    populate(force=False)

    assert dummy_run.exists()
    assert unrelated_file.exists()
    run_files = list(temp_history_dir.glob("run_*.json"))
    assert len(run_files) == 25  # 24 newly generated + 1 dummy run


def test_populate_purges_existing_files_when_force_is_true(temp_history_dir):
    """Test that populate(force=True) purges existing run_*.json files and recreates them."""
    old_run1 = temp_history_dir / "run_old1.json"
    old_run1.write_text(json.dumps({"old": 1}))
    old_run2 = temp_history_dir / "run_old2.json"
    old_run2.write_text(json.dumps({"old": 2}))
    unrelated_file = temp_history_dir / "preserve.txt"
    unrelated_file.write_text("do not delete")

    populate(force=True)

    assert not old_run1.exists()
    assert not old_run2.exists()
    assert unrelated_file.exists()
    run_files = list(temp_history_dir.glob("run_*.json"))
    assert len(run_files) == 24


def test_populate_force_true_tolerates_unlink_failure(temp_history_dir):
    """Test that unlink exceptions during purge are caught and ignored."""
    old_run = temp_history_dir / "run_locked.json"
    old_run.write_text(json.dumps({"locked": True}))

    orig_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self.name == "run_locked.json":
            raise PermissionError("Cannot delete locked file")
        return orig_unlink(self, *args, **kwargs)

    with patch.object(Path, "unlink", side_effect=failing_unlink):
        populate(force=True)

    assert old_run.exists()
    run_files = list(temp_history_dir.glob("run_*.json"))
    assert len(run_files) == 25  # 24 newly generated + 1 locked run that failed to delete


def test_populate_creates_history_dir_if_nonexistent():
    """Test that populate creates the history directory if it doesn't exist yet."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        non_existent_subdir = Path(tmp_dir) / "sub" / "history"
        assert not non_existent_subdir.exists()

        with patch.object(populate_history, "HISTORY_DIR", non_existent_subdir):
            populate(force=True)

        assert non_existent_subdir.exists()
        assert (non_existent_subdir / ".gitkeep").exists()
        assert len(list(non_existent_subdir.glob("run_*.json"))) == 24


def test_main_cli_default_flag(monkeypatch):
    """Test main() parses default flag force=False."""
    with patch("populate_history.populate") as mock_populate:
        monkeypatch.setattr(sys, "argv", ["populate_history.py"])
        main()
        mock_populate.assert_called_once_with(force=False)


def test_main_cli_force_flag(monkeypatch):
    """Test main() parses --force flag."""
    with patch("populate_history.populate") as mock_populate:
        monkeypatch.setattr(sys, "argv", ["populate_history.py", "--force"])
        main()
        mock_populate.assert_called_once_with(force=True)
