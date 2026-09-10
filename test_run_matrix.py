import io
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import run_matrix
from run_matrix import (
    TeeLogger,
    log_error,
    resolve_latest_snapshot,
    get_default_gguf_paths,
    restart_router_service,
    wait_for_endpoint_health,
    get_completed_runs,
    run_matrix as run_matrix_func,
)


class TestTeeLoggerAndLogError:
    def test_tee_logger_write_and_flush(self, tmp_path):
        stream = io.StringIO()
        log_file_path = tmp_path / "test.log"
        logger = TeeLogger(stream, log_file_path)

        logger.write("hello world\n")
        logger.flush()

        assert stream.getvalue() == "hello world\n"
        assert log_file_path.read_text(encoding="utf-8") == "hello world\n"

        logger.close()

    def test_tee_logger_stream_or_path_and_close(self, tmp_path):
        stream = io.StringIO()
        log_file_path = tmp_path / "test_stream.log"

        f = open(log_file_path, "w", encoding="utf-8")
        logger = TeeLogger(stream, f)

        logger.write("stream output\n")
        logger.flush()
        logger.close()

        # External file handle shouldn't be closed by logger.close()
        assert not f.closed
        f.close()
        assert log_file_path.read_text(encoding="utf-8") == "stream output\n"

    def test_tee_logger_properties(self, tmp_path):
        mock_stream = MagicMock()
        mock_stream.fileno.return_value = 1
        mock_stream.isatty.return_value = True
        mock_stream.encoding = "utf-8"

        log_file_path = tmp_path / "prop.log"
        logger = TeeLogger(mock_stream, log_file_path)

        assert logger.fileno() == 1
        assert logger.isatty() is True
        assert logger.encoding == "utf-8"

        logger.close()

    def test_log_error(self, tmp_path, monkeypatch):
        err_log_path = tmp_path / "matrix_errors.log"
        monkeypatch.setattr(run_matrix, "ERROR_LOG", err_log_path)

        log_error("TestModel", 4096, "Test Error", stdout="Out", stderr="Err")

        assert err_log_path.exists()
        content = err_log_path.read_text(encoding="utf-8")
        assert "Model=TestModel | Context=4096" in content
        assert "ERROR: Test Error" in content
        assert "STDOUT:\nOut" in content
        assert "STDERR:\nErr" in content


class TestGgufAndHistoryParsing:
    def test_resolve_latest_snapshot_with_subdirs(self, tmp_path):
        cache_dir = tmp_path / "cache"
        repo_folder = "models--test--repo"
        gguf_filename = "model.gguf"
        fallback_hash = "fallback123"

        snapshots_dir = cache_dir / "hub" / repo_folder / "snapshots"
        snap1 = snapshots_dir / "dir1"
        snap2 = snapshots_dir / "dir2"
        snap1.mkdir(parents=True)
        snap2.mkdir(parents=True)

        os.utime(snap1, (100, 100))
        os.utime(snap2, (200, 200))

        resolved = resolve_latest_snapshot(cache_dir, repo_folder, gguf_filename, fallback_hash)
        assert resolved == str(snap2 / gguf_filename)

    def test_resolve_latest_snapshot_fallback(self, tmp_path):
        cache_dir = tmp_path / "cache"
        repo_folder = "models--test--repo"
        gguf_filename = "model.gguf"
        fallback_hash = "fallback123"

        resolved = resolve_latest_snapshot(cache_dir, repo_folder, gguf_filename, fallback_hash)
        expected = str(cache_dir / "hub" / repo_folder / "snapshots" / fallback_hash / gguf_filename)
        assert resolved == expected

    def test_get_default_gguf_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGUF_PATH_QWEN27B", "/custom/qwen27b.gguf")
        paths = get_default_gguf_paths(cache_dir=str(tmp_path))
        assert paths["Qwen3.6-27B"] == "/custom/qwen27b.gguf"
        assert "Qwen3.6-27B-spec3" in paths

    def test_get_completed_runs_missing_history_dir(self, tmp_path, monkeypatch):
        non_existent = tmp_path / "no_history"
        monkeypatch.setattr(run_matrix, "HISTORY_DIR", non_existent)
        completed = get_completed_runs()
        assert completed == set()

    def test_get_completed_runs_parsing(self, tmp_path, monkeypatch):
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.setattr(run_matrix, "HISTORY_DIR", history_dir)

        # 1. KLD file that should be ignored
        (history_dir / "run_model_f16.json").write_text("{}", encoding="utf-8")

        # 2. File with metadata cli_arguments
        run1 = {
            "run_metadata": {
                "cli_arguments": ["--model", "Qwen3.6-27B", "--tokens", "8192"]
            }
        }
        (history_dir / "run_1.json").write_text(json.dumps(run1), encoding="utf-8")

        # 3. File with model_settings fallback (profile_alias)
        run2 = {
            "model_settings": {
                "profile_alias": "Qwen3.6-27B-spec3"
            },
            "run_metadata": {
                "cli_arguments": ["--tokens", "32000"]
            }
        }
        (history_dir / "run_2.json").write_text(json.dumps(run2), encoding="utf-8")

        # 4. File with unsloth mapping profile and invalid tokens string handling
        run3 = {
            "model_settings": {
                "model_name": "unsloth/Qwen3.6-27B-MTP-GGUF"
            },
            "run_metadata": {
                "cli_arguments": ["--tokens", "invalid"]
            }
        }
        (history_dir / "run_3.json").write_text(json.dumps(run3), encoding="utf-8")

        # Create presets file
        presets_file = tmp_path / "presets.ini"
        presets_file.write_text("[Qwen3.6-27B-spec4]\n", encoding="utf-8")

        completed = get_completed_runs(presets_file=str(presets_file))
        assert ("Qwen3.6-27B", 8192) in completed
        assert ("Qwen3.6-27B-spec3", 32000) in completed


class TestServiceAndExecutionMatrix:
    @patch("subprocess.run")
    def test_restart_router_service_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert restart_router_service("m1", 1024) is True
        mock_run.assert_called_once_with(["systemctl", "--user", "restart", "llama-router"], capture_output=True, text=True)

    @patch("subprocess.run")
    @patch("run_matrix.log_error")
    def test_restart_router_service_failure(self, mock_log_error, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="out", stderr="err")
        assert restart_router_service("m1", 1024) is False
        mock_log_error.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_wait_for_endpoint_health_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        assert wait_for_endpoint_health("http://127.0.0.1:8081", timeout=5, poll_interval=0.1) is True

    @patch("urllib.request.urlopen", side_effect=Exception("Connection refused"))
    def test_wait_for_endpoint_health_timeout(self, mock_urlopen):
        assert wait_for_endpoint_health("http://127.0.0.1:8081", timeout=0.2, poll_interval=0.1) is False

    @patch("run_matrix.wait_for_endpoint_health", return_value=True)
    @patch("run_matrix.get_completed_runs", return_value=set())
    @patch("run_matrix.get_default_gguf_paths", return_value={})
    @patch("subprocess.run")
    def test_run_matrix_success(self, mock_sub_run, mock_gguf, mock_completed, mock_health, tmp_path, monkeypatch):
        history_dir = tmp_path / "history"
        err_log = tmp_path / "errors.log"
        run_log = tmp_path / "run.log"

        monkeypatch.setattr(run_matrix, "HISTORY_DIR", history_dir)
        monkeypatch.setattr(run_matrix, "ERROR_LOG", err_log)
        monkeypatch.setattr(run_matrix, "RUN_LOG", run_log)

        mock_sub_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        run_matrix_func(endpoint="http://127.0.0.1:8081")

        assert history_dir.exists()
        assert (history_dir / ".gitkeep").exists()
        assert mock_sub_run.call_count == 24  # 4 models * 6 contexts

    @patch("run_matrix.wait_for_endpoint_health", return_value=True)
    @patch("run_matrix.get_completed_runs", return_value=set())
    @patch("run_matrix.get_default_gguf_paths", return_value={})
    @patch("run_matrix.restart_router_service")
    @patch("subprocess.run")
    def test_run_matrix_failure_and_skip(self, mock_sub_run, mock_restart, mock_gguf, mock_completed, mock_health, tmp_path, monkeypatch):
        history_dir = tmp_path / "history"
        err_log = tmp_path / "errors.log"
        run_log = tmp_path / "run.log"

        monkeypatch.setattr(run_matrix, "HISTORY_DIR", history_dir)
        monkeypatch.setattr(run_matrix, "ERROR_LOG", err_log)
        monkeypatch.setattr(run_matrix, "RUN_LOG", run_log)

        # First run returns exit code 1 (failure)
        mock_sub_run.return_value = MagicMock(returncode=1, stdout="failed", stderr="err")

        run_matrix_func(endpoint="http://127.0.0.1:8081")

        # Should attempt 1 run for each model (since first context fails and marks model as broken)
        assert mock_sub_run.call_count == 4
        assert mock_restart.call_count == 4
