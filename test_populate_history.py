import json
from unittest.mock import patch
import pytest
import populate_history

def test_populate_basic(tmp_path, monkeypatch):
    test_history_dir = tmp_path / "history"
    monkeypatch.setattr(populate_history, "HISTORY_DIR", test_history_dir)

    populate_history.populate()

    assert test_history_dir.exists()
    assert (test_history_dir / ".gitkeep").exists()

    run_files = list(test_history_dir.glob("run_*.json"))
    assert len(run_files) == 24


def test_populate_force_true(tmp_path, monkeypatch):
    test_history_dir = tmp_path / "history"
    test_history_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(populate_history, "HISTORY_DIR", test_history_dir)

    old_run = test_history_dir / "run_old.json"
    old_run.write_text("{}")
    other_file = test_history_dir / "other.txt"
    other_file.write_text("keep me")

    populate_history.populate(force=True)

    assert not old_run.exists()
    assert other_file.exists()
    assert (test_history_dir / ".gitkeep").exists()

    run_files = list(test_history_dir.glob("run_*.json"))
    assert len(run_files) == 24


def test_populate_force_false(tmp_path, monkeypatch):
    test_history_dir = tmp_path / "history"
    test_history_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(populate_history, "HISTORY_DIR", test_history_dir)

    old_run = test_history_dir / "run_old.json"
    old_run.write_text("{}")

    populate_history.populate(force=False)

    assert old_run.exists()
    run_files = list(test_history_dir.glob("run_*.json"))
    # 24 newly created run_*.json files + 1 old run_old.json
    assert len(run_files) == 25


def test_json_structure_integrity(tmp_path, monkeypatch):
    test_history_dir = tmp_path / "history"
    monkeypatch.setattr(populate_history, "HISTORY_DIR", test_history_dir)

    populate_history.populate()

    run_files = list(test_history_dir.glob("run_*.json"))
    assert len(run_files) > 0

    required_sections = [
        "run_metadata",
        "model_settings",
        "throughput_metrics",
        "reasoning_accuracy",
        "quantization_loss",
    ]

    for rf in run_files:
        with open(rf, "r") as f:
            data = json.load(f)

        for section in required_sections:
            assert section in data, f"Missing section '{section}' in {rf}"

        # Validate run_metadata fields
        assert data["run_metadata"].get("synthetic") is True
        assert "timestamp" in data["run_metadata"]
        assert "target_endpoint" in data["run_metadata"]

        # Validate model_settings fields
        assert "model_name" in data["model_settings"]
        assert "profile_alias" in data["model_settings"]
        assert "kv_cache_quant" in data["model_settings"]

        # Validate throughput_metrics fields
        assert isinstance(data["throughput_metrics"].get("prefill_speed"), (int, float))
        assert isinstance(data["throughput_metrics"].get("decode_speed"), (int, float))
        assert isinstance(data["throughput_metrics"].get("ttft"), (int, float))

        # Validate reasoning_accuracy fields
        assert "needle" in data["reasoning_accuracy"]
        assert "ruler" in data["reasoning_accuracy"]
        assert "longbench" in data["reasoning_accuracy"]
        assert "swe_bench" in data["reasoning_accuracy"]

        # Validate quantization_loss fields
        assert isinstance(data["quantization_loss"].get("perplexity"), (int, float))
        assert isinstance(data["quantization_loss"].get("mean_kld"), (int, float))
        assert isinstance(data["quantization_loss"].get("same_top_match_percent"), (int, float))


def test_main_cli_default(monkeypatch):
    with patch("populate_history.populate") as mock_populate:
        monkeypatch.setattr("sys.argv", ["populate_history.py"])
        populate_history.main()
        mock_populate.assert_called_once_with(force=False)


def test_main_cli_force(monkeypatch):
    with patch("populate_history.populate") as mock_populate:
        monkeypatch.setattr("sys.argv", ["populate_history.py", "--force"])
        populate_history.main()
        mock_populate.assert_called_once_with(force=True)
