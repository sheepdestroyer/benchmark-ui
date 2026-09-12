import os
import io
import sys
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess

import run_matrix


class TestTeeLogger(unittest.TestCase):
    def setUp(self):
        self.orig_stdout = io.StringIO()

    def test_init_with_file_path_str(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "test.log")
            logger = run_matrix.TeeLogger(self.orig_stdout, log_path)
            self.assertTrue(logger._owns_file)
            self.assertFalse(logger.log_file.closed)
            logger.write("hello file path\n")
            logger.flush()
            logger.close()
            self.assertTrue(logger.log_file.closed)
            with open(log_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "hello file path\n")

    def test_init_with_file_path_obj(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "test_path.log"
            logger = run_matrix.TeeLogger(self.orig_stdout, log_path)
            self.assertTrue(logger._owns_file)
            logger.write("pathlib log\n")
            logger.close()
            self.assertTrue(logger.log_file.closed)
            self.assertEqual(log_path.read_text(encoding="utf-8"), "pathlib log\n")

    def test_init_with_stream(self):
        log_stream = io.StringIO()
        logger = run_matrix.TeeLogger(self.orig_stdout, log_stream)
        self.assertFalse(logger._owns_file)
        logger.write("stream log\n")
        logger.flush()
        logger.close()
        self.assertFalse(log_stream.closed)
        self.assertEqual(log_stream.getvalue(), "stream log\n")
        self.assertEqual(self.orig_stdout.getvalue(), "stream log\n")

    def test_write_and_flush_when_log_file_closed_or_none(self):
        log_stream = io.StringIO()
        logger = run_matrix.TeeLogger(self.orig_stdout, log_stream)
        log_stream.close()
        # Writing when closed should not throw
        logger.write("test closed\n")
        logger.flush()
        self.assertIn("test closed\n", self.orig_stdout.getvalue())

        # Test when log_file is None
        logger.log_file = None
        logger.write("test none\n")
        logger.flush()
        logger.close()
        self.assertIn("test none\n", self.orig_stdout.getvalue())

    def test_fileno_isatty_and_encoding(self):
        mock_stream = MagicMock()
        mock_stream.fileno.return_value = 5
        mock_stream.isatty.return_value = True
        mock_stream.encoding = "latin-1"

        logger = run_matrix.TeeLogger(mock_stream, io.StringIO())
        self.assertEqual(logger.fileno(), 5)
        self.assertTrue(logger.isatty())
        self.assertEqual(logger.encoding, "latin-1")

        # Test fallback encoding when stream has no encoding attribute
        mock_stream_no_enc = MagicMock(spec=["write", "flush", "fileno", "isatty"])
        logger2 = run_matrix.TeeLogger(mock_stream_no_enc, io.StringIO())
        self.assertEqual(logger2.encoding, "utf-8")


class TestResolveLatestSnapshot(unittest.TestCase):
    def test_snapshots_with_subdirs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshots_dir = Path(tmp_dir) / "hub" / "models--test" / "snapshots"
            dir1 = snapshots_dir / "hash1"
            dir2 = snapshots_dir / "hash2"
            dir1.mkdir(parents=True)
            dir2.mkdir(parents=True)

            os.utime(dir1, (1000, 1000))
            os.utime(dir2, (2000, 2000))

            res = run_matrix.resolve_latest_snapshot(tmp_dir, "models--test", "model.gguf", "fallback")
            self.assertEqual(res, str(dir2 / "model.gguf"))

    def test_snapshots_empty_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshots_dir = Path(tmp_dir) / "hub" / "models--test" / "snapshots"
            snapshots_dir.mkdir(parents=True)

            res = run_matrix.resolve_latest_snapshot(tmp_dir, "models--test", "model.gguf", "fallback")
            expected = str(snapshots_dir / "fallback" / "model.gguf")
            self.assertEqual(res, expected)

    def test_snapshots_nonexistent_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = run_matrix.resolve_latest_snapshot(tmp_dir, "models--missing", "model.gguf", "fallback_hash")
            expected = str(Path(tmp_dir) / "hub" / "models--missing" / "snapshots" / "fallback_hash" / "model.gguf")
            self.assertEqual(res, expected)


class TestGetDefaultGgufPaths(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch("run_matrix.resolve_latest_snapshot")
    def test_default_gguf_paths_no_env(self, mock_resolve):
        mock_resolve.return_value = "/resolved/model.gguf"
        paths = run_matrix.get_default_gguf_paths(cache_dir="/custom/cache")
        self.assertIn("Qwen3.6-27B", paths)
        self.assertIn("Qwen3.6-27B-spec3", paths)
        self.assertIn("Qwen3.6-27B-spec4", paths)
        self.assertIn("Qwen3.6-35B-A3B-spec", paths)
        self.assertEqual(paths["Qwen3.6-27B"], "/resolved/model.gguf")
        self.assertEqual(mock_resolve.call_count, 4)

    @patch.dict(os.environ, {"GGUF_PATH_QWEN27B": "/env/path/qwen27b.gguf"}, clear=True)
    def test_default_gguf_paths_env_override(self):
        paths = run_matrix.get_default_gguf_paths()
        self.assertEqual(paths["Qwen3.6-27B"], "/env/path/qwen27b.gguf")


class TestRestartRouterService(unittest.TestCase):
    @patch("subprocess.run")
    def test_restart_success(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["systemctl", "--user", "restart", "llama-router"],
            returncode=0,
            stdout="restarted",
            stderr=""
        )
        res = run_matrix.restart_router_service("test_model", 1024)
        self.assertTrue(res)
        mock_run.assert_called_once_with(["systemctl", "--user", "restart", "llama-router"], capture_output=True, text=True)

    @patch("run_matrix.log_error")
    @patch("subprocess.run")
    def test_restart_failure_with_stderr(self, mock_run, mock_log_error):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["systemctl", "--user", "restart", "llama-router"],
            returncode=1,
            stdout="",
            stderr="Failed to restart llama-router.service: Unit not found."
        )
        res = run_matrix.restart_router_service("test_model", 2048)
        self.assertFalse(res)
        mock_log_error.assert_called_once_with(
            "test_model",
            2048,
            "systemctl restart llama-router failed with exit code 1",
            stdout="",
            stderr="Failed to restart llama-router.service: Unit not found."
        )

    @patch("run_matrix.log_error")
    @patch("subprocess.run")
    def test_restart_failure_no_stderr(self, mock_run, mock_log_error):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["systemctl", "--user", "restart", "llama-router"],
            returncode=2,
            stdout="some stdout",
            stderr=""
        )
        res = run_matrix.restart_router_service()
        self.assertFalse(res)
        mock_log_error.assert_called_once_with(
            "N/A",
            0,
            "systemctl restart llama-router failed with exit code 2",
            stdout="some stdout",
            stderr=""
        )


class TestWaitForEndpointHealth(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_endpoint_ready_immediately(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = run_matrix.wait_for_endpoint_health("http://127.0.0.1:8081", timeout=5, poll_interval=0.01)
        self.assertTrue(res)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_endpoint_retry_then_success(self, mock_urlopen, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.side_effect = [
            Exception("Connection refused"),
            mock_resp
        ]
        mock_resp.__enter__.return_value = mock_resp

        res = run_matrix.wait_for_endpoint_health("http://127.0.0.1:8081", timeout=5, poll_interval=0.01)
        self.assertTrue(res)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_endpoint_timeout(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = Exception("Connection refused")
        times = [100.0, 101.0, 102.0, 105.0]
        with patch("time.time", side_effect=times):
            res = run_matrix.wait_for_endpoint_health("http://127.0.0.1:8081", timeout=3, poll_interval=0.01)
            self.assertFalse(res)


class TestLogError(unittest.TestCase):
    def test_log_error_full(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            error_file = Path(tmp_dir) / "errors.log"
            with patch("run_matrix.ERROR_LOG", error_file):
                run_matrix.log_error("Qwen3.6-27B", 8192, "OOM Error", stdout="Out message", stderr="Err message")
                content = error_file.read_text(encoding="utf-8")
                self.assertIn("RUN CONFIGURATION: Model=Qwen3.6-27B | Context=8192", content)
                self.assertIn("ERROR: OOM Error", content)
                self.assertIn("STDOUT:\nOut message\n", content)
                self.assertIn("STDERR:\nErr message\n", content)
                self.assertIn("TIMESTAMP:", content)

    def test_log_error_minimal(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            error_file = Path(tmp_dir) / "errors.log"
            with patch("run_matrix.ERROR_LOG", error_file):
                run_matrix.log_error("Qwen3.6-27B", 1024, "Simple Error")
                content = error_file.read_text(encoding="utf-8")
                self.assertIn("RUN CONFIGURATION: Model=Qwen3.6-27B | Context=1024", content)
                self.assertIn("ERROR: Simple Error", content)
                self.assertNotIn("STDOUT:", content)
                self.assertNotIn("STDERR:", content)


class TestExtractRunIdentifiers(unittest.TestCase):
    def test_extract_from_cli_arguments(self):
        data = {
            "run_metadata": {
                "cli_arguments": ["--model", "Qwen3.6-27B", "--tokens", "8192"]
            }
        }
        res = run_matrix.extract_run_identifiers(data)
        self.assertEqual(res, ("Qwen3.6-27B", 8192))

    def test_extract_fallback_profile_alias(self):
        data = {
            "model_settings": {"profile_alias": "profile-alpha"},
            "run_metadata": {
                "cli_arguments": ["--tokens", "4096"]
            }
        }
        res = run_matrix.extract_run_identifiers(data)
        self.assertEqual(res, ("profile-alpha", 4096))

    def test_extract_fallback_model_name(self):
        data = {
            "model_settings": {"model_name": "model-beta"},
            "run_metadata": {
                "cli_arguments": ["--tokens", "2048"]
            }
        }
        res = run_matrix.extract_run_identifiers(data)
        self.assertEqual(res, ("model-beta", 2048))

    def test_extract_invalid_tokens(self):
        data = {
            "run_metadata": {
                "cli_arguments": ["--model", "Qwen3.6-27B", "--tokens", "invalid_num"]
            }
        }
        res = run_matrix.extract_run_identifiers(data)
        self.assertIsNone(res)

    def test_extract_missing_model_or_tokens(self):
        self.assertIsNone(run_matrix.extract_run_identifiers({}))
        self.assertIsNone(run_matrix.extract_run_identifiers({"run_metadata": {"cli_arguments": ["--model", "m"]}}))
        self.assertIsNone(run_matrix.extract_run_identifiers({"run_metadata": {"cli_arguments": ["--tokens", "1024"]}}))

    def test_extract_presets_exact_match(self):
        data = {
            "run_metadata": {
                "cli_arguments": ["--model", "unsloth/Qwen3.6-27B:latest", "--tokens", "1024"]
            }
        }
        presets = {"Qwen3.6-27B", "unsloth/Qwen3.6-27B:latest"}
        res = run_matrix.extract_run_identifiers(data, presets_sections=presets)
        self.assertEqual(res, ("unsloth/Qwen3.6-27B:latest", 1024))

    def test_extract_presets_partial_match(self):
        data = {
            "run_metadata": {
                "cli_arguments": ["--model", "models:unsloth-qwen3.6-27b-mtp", "--tokens", "1024"]
            }
        }
        presets = {"Qwen3.6-27B-MTP"}
        res = run_matrix.extract_run_identifiers(data, presets_sections=presets)
        self.assertEqual(res, ("Qwen3.6-27B-MTP", 1024))

    def test_extract_presets_no_match(self):
        data = {
            "run_metadata": {
                "cli_arguments": ["--model", "unsloth/other-model", "--tokens", "1024"]
            }
        }
        presets = {"Qwen3.6-27B"}
        res = run_matrix.extract_run_identifiers(data, presets_sections=presets)
        self.assertEqual(res, ("unsloth/other-model", 1024))


class TestPresetSectionsCaching(unittest.TestCase):
    def setUp(self):
        run_matrix._get_preset_sections.cache_clear()

    def tearDown(self):
        run_matrix._get_preset_sections.cache_clear()

    def test_preset_sections_caching(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[Qwen3.6-27B]\nthreads=8\n", encoding="utf-8")

            s1 = run_matrix._get_preset_sections(str(presets_file))
            self.assertEqual(s1, {"Qwen3.6-27B"})

            # Modify file on disk without clearing cache
            presets_file.write_text("[Qwen3.6-27B]\n[NewModel]\n", encoding="utf-8")

            s2 = run_matrix._get_preset_sections(str(presets_file))
            self.assertEqual(s2, {"Qwen3.6-27B"})

            # Clear cache and verify new content loaded
            run_matrix._get_preset_sections.cache_clear()
            s3 = run_matrix._get_preset_sections(str(presets_file))
            self.assertEqual(s3, {"Qwen3.6-27B", "NewModel"})

    def test_preset_sections_nonexistent_and_malformed_fallback(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent = str(Path(tmp_dir) / "missing.ini")
            s_missing = run_matrix._get_preset_sections(non_existent)
            self.assertEqual(s_missing, set())

            malformed_file = Path(tmp_dir) / "malformed.ini"
            malformed_file.write_text("corrupted [ini without closing", encoding="utf-8")

            run_matrix._get_preset_sections.cache_clear()
            s_malformed = run_matrix._get_preset_sections(str(malformed_file))
            self.assertEqual(s_malformed, set())


class TestGetCompletedRuns(unittest.TestCase):
    def test_history_dir_not_exists(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent = Path(tmp_dir) / "non_existent_history"
            with patch("run_matrix.HISTORY_DIR", non_existent):
                res = run_matrix.get_completed_runs()
                self.assertEqual(res, set())

    def test_get_completed_runs_valid_and_kld_filtered(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            hist_dir = Path(tmp_dir) / "history"
            hist_dir.mkdir()

            run1 = hist_dir / "run_20260101_120000.json"
            run1.write_text(json.dumps({
                "run_metadata": {"cli_arguments": ["--model", "Qwen3.6-27B", "--tokens", "8192"]}
            }), encoding="utf-8")

            run2 = hist_dir / "run_20260101_130000.json"
            run2.write_text(json.dumps({
                "run_metadata": {"cli_arguments": ["--model", "Qwen3.6-35B-A3B-spec", "--tokens", "32000"]}
            }), encoding="utf-8")

            for quant in ["f16", "q8_0", "q5_1", "q4_0"]:
                kld_file = hist_dir / f"run_kld_{quant}.json"
                kld_file.write_text(json.dumps({
                    "run_metadata": {"cli_arguments": ["--model", "Qwen3.6-27B", "--tokens", "1024"]}
                }), encoding="utf-8")

            with patch("run_matrix.HISTORY_DIR", hist_dir):
                completed = run_matrix.get_completed_runs()
                self.assertEqual(completed, {
                    ("Qwen3.6-27B", 8192),
                    ("Qwen3.6-35B-A3B-spec", 32000),
                })

    def test_get_completed_runs_with_corrupt_and_empty_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            hist_dir = Path(tmp_dir) / "history"
            hist_dir.mkdir()

            corrupt = hist_dir / "run_corrupt.json"
            corrupt.write_text("{corrupt json", encoding="utf-8")

            empty = hist_dir / "run_empty.json"
            empty.write_text("", encoding="utf-8")

            valid = hist_dir / "run_valid.json"
            valid.write_text(json.dumps({
                "run_metadata": {"cli_arguments": ["--model", "Qwen3.6-27B", "--tokens", "1024"]}
            }), encoding="utf-8")

            with patch("run_matrix.HISTORY_DIR", hist_dir):
                completed = run_matrix.get_completed_runs()
                self.assertEqual(completed, {("Qwen3.6-27B", 1024)})

    def test_get_completed_runs_with_presets_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            hist_dir = Path(tmp_dir) / "history"
            hist_dir.mkdir()
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[Qwen3.6-27B]\nthreads = 8\n", encoding="utf-8")

            run_file = hist_dir / "run_presets.json"
            run_file.write_text(json.dumps({
                "run_metadata": {"cli_arguments": ["--model", "unsloth/qwen3.6-27b", "--tokens", "8192"]}
            }), encoding="utf-8")

            with patch("run_matrix.HISTORY_DIR", hist_dir):
                completed = run_matrix.get_completed_runs(presets_file=str(presets_file))
                self.assertEqual(completed, {("Qwen3.6-27B", 8192)})

    def test_get_completed_runs_corrupt_presets_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            hist_dir = Path(tmp_dir) / "history"
            hist_dir.mkdir()
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("corrupted [ini without closing", encoding="utf-8")

            run_file = hist_dir / "run_presets.json"
            run_file.write_text(json.dumps({
                "run_metadata": {"cli_arguments": ["--model", "NormalModel", "--tokens", "8192"]}
            }), encoding="utf-8")

            with patch("run_matrix.HISTORY_DIR", hist_dir):
                completed = run_matrix.get_completed_runs(presets_file=str(presets_file))
                self.assertEqual(completed, {("NormalModel", 8192)})


class TestRunMatrix(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.hist_dir = Path(self.tmp_dir.name) / "history"
        self.error_log = Path(self.tmp_dir.name) / "errors.log"
        self.run_log = Path(self.tmp_dir.name) / "matrix_run.log"

    def tearDown(self):
        self.tmp_dir.cleanup()

    @patch("run_matrix.wait_for_endpoint_health")
    @patch("subprocess.run")
    def test_run_matrix_all_already_completed(self, mock_subproc, mock_health):
        all_completed = {
            (m, c)
            for m in ["Qwen3.6-27B", "Qwen3.6-27B-spec3", "Qwen3.6-27B-spec4", "Qwen3.6-35B-A3B-spec"]
            for c in [1024, 8192, 32000, 64000, 128000, 228000]
        }
        with patch("run_matrix.HISTORY_DIR", self.hist_dir), \
             patch("run_matrix.ERROR_LOG", self.error_log), \
             patch("run_matrix.RUN_LOG", self.run_log), \
             patch("run_matrix.get_completed_runs", return_value=all_completed):
            run_matrix.run_matrix()

        mock_health.assert_not_called()
        mock_subproc.assert_not_called()

    @patch("run_matrix.log_error")
    @patch("run_matrix.wait_for_endpoint_health", return_value=False)
    @patch("subprocess.run")
    def test_run_matrix_endpoint_unhealthy(self, mock_subproc, mock_health, mock_log_err):
        with patch("run_matrix.HISTORY_DIR", self.hist_dir), \
             patch("run_matrix.ERROR_LOG", self.error_log), \
             patch("run_matrix.RUN_LOG", self.run_log), \
             patch("run_matrix.get_completed_runs", return_value=set()):
            run_matrix.run_matrix()

        self.assertEqual(mock_health.call_count, 4)
        self.assertEqual(mock_log_err.call_count, 4)
        mock_subproc.assert_not_called()

    @patch("run_matrix.wait_for_endpoint_health", return_value=True)
    @patch("subprocess.run")
    def test_run_matrix_success_run(self, mock_subproc, mock_health):
        mock_subproc.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        all_but_one = {
            (m, c)
            for m in ["Qwen3.6-27B", "Qwen3.6-27B-spec3", "Qwen3.6-27B-spec4", "Qwen3.6-35B-A3B-spec"]
            for c in [1024, 8192, 32000, 64000, 128000, 228000]
        }
        all_but_one.remove(("Qwen3.6-27B", 1024))

        with patch("run_matrix.HISTORY_DIR", self.hist_dir), \
             patch("run_matrix.ERROR_LOG", self.error_log), \
             patch("run_matrix.RUN_LOG", self.run_log), \
             patch("run_matrix.get_completed_runs", return_value=all_but_one), \
             patch("run_matrix.get_default_gguf_paths", return_value={"Qwen3.6-27B": "/mock/model.gguf"}), \
             patch("os.path.exists", return_value=True):
            run_matrix.run_matrix()

        mock_subproc.assert_called_once()
        cmd = mock_subproc.call_args[0][0]
        self.assertIn("run_suite.py", cmd)
        self.assertIn("--gguf-path", cmd)
        self.assertIn("/mock/model.gguf", cmd)
        self.assertEqual(mock_subproc.call_args[1]["timeout"], 300)

    @patch("run_matrix.restart_router_service")
    @patch("run_matrix.log_error")
    @patch("run_matrix.wait_for_endpoint_health", return_value=True)
    @patch("subprocess.run")
    def test_run_matrix_nonzero_returncode(self, mock_subproc, mock_health, mock_log_err, mock_restart):
        mock_subproc.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="out", stderr="err")
        all_but_one_model = {
            (m, c)
            for m in ["Qwen3.6-27B-spec3", "Qwen3.6-27B-spec4", "Qwen3.6-35B-A3B-spec"]
            for c in [1024, 8192, 32000, 64000, 128000, 228000]
        }

        with patch("run_matrix.HISTORY_DIR", self.hist_dir), \
             patch("run_matrix.ERROR_LOG", self.error_log), \
             patch("run_matrix.RUN_LOG", self.run_log), \
             patch("run_matrix.get_completed_runs", return_value=all_but_one_model):
            run_matrix.run_matrix()

        mock_log_err.assert_called_once_with("Qwen3.6-27B", 1024, "run_suite.py returned exit code 1", "out", "err")
        mock_restart.assert_called_once_with("Qwen3.6-27B", 1024)

    @patch("run_matrix.log_error")
    @patch("run_matrix.wait_for_endpoint_health", return_value=True)
    @patch("subprocess.run")
    def test_run_matrix_timeout_exception(self, mock_subproc, mock_health, mock_log_err):
        mock_subproc.side_effect = [
            subprocess.TimeoutExpired(cmd=["run_suite.py"], timeout=300, output="timeout out", stderr="timeout err"),
            subprocess.CompletedProcess(args=[], returncode=0),
        ]
        all_but_one_model = {
            (m, c)
            for m in ["Qwen3.6-27B-spec3", "Qwen3.6-27B-spec4", "Qwen3.6-35B-A3B-spec"]
            for c in [1024, 8192, 32000, 64000, 128000, 228000]
        }

        with patch("run_matrix.HISTORY_DIR", self.hist_dir), \
             patch("run_matrix.ERROR_LOG", self.error_log), \
             patch("run_matrix.RUN_LOG", self.run_log), \
             patch("run_matrix.get_completed_runs", return_value=all_but_one_model):
            run_matrix.run_matrix()

        mock_log_err.assert_called_once()
        self.assertIn("Subprocess timed out", mock_log_err.call_args[0][2])

    @patch("run_matrix.log_error")
    @patch("run_matrix.wait_for_endpoint_health", return_value=True)
    @patch("subprocess.run")
    def test_run_matrix_general_exception(self, mock_subproc, mock_health, mock_log_err):
        mock_subproc.side_effect = [
            RuntimeError("Unexpected subprocess crash"),
            subprocess.CompletedProcess(args=[], returncode=0),
        ]
        all_but_one_model = {
            (m, c)
            for m in ["Qwen3.6-27B-spec3", "Qwen3.6-27B-spec4", "Qwen3.6-35B-A3B-spec"]
            for c in [1024, 8192, 32000, 64000, 128000, 228000]
        }

        with patch("run_matrix.HISTORY_DIR", self.hist_dir), \
             patch("run_matrix.ERROR_LOG", self.error_log), \
             patch("run_matrix.RUN_LOG", self.run_log), \
             patch("run_matrix.get_completed_runs", return_value=all_but_one_model):
            run_matrix.run_matrix()

        mock_log_err.assert_called_once()
        self.assertIn("Subprocess exception", mock_log_err.call_args[0][2])

    @patch("run_matrix.wait_for_endpoint_health", return_value=True)
    @patch("subprocess.run")
    def test_run_matrix_timeout_thresholds(self, mock_subproc, mock_health):
        mock_subproc.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        for ctx, expected_timeout in [(1024, 300), (32000, 600), (64000, 1000), (128000, 2400)]:
            all_runs_except_ctx = {
                (m, c)
                for m in ["Qwen3.6-27B", "Qwen3.6-27B-spec3", "Qwen3.6-27B-spec4", "Qwen3.6-35B-A3B-spec"]
                for c in [1024, 8192, 32000, 64000, 128000, 228000]
            }
            all_runs_except_ctx.remove(("Qwen3.6-27B", ctx))

            mock_subproc.reset_mock()
            with patch("run_matrix.HISTORY_DIR", self.hist_dir), \
                 patch("run_matrix.ERROR_LOG", self.error_log), \
                 patch("run_matrix.RUN_LOG", self.run_log), \
                 patch("run_matrix.get_completed_runs", return_value=all_runs_except_ctx):
                run_matrix.run_matrix()

            self.assertEqual(mock_subproc.call_args[1]["timeout"], expected_timeout)


class TestMain(unittest.TestCase):
    @patch("atexit.register")
    @patch("run_matrix.run_matrix")
    def test_main(self, mock_run_matrix, mock_atexit):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_run_log = Path(tmp_dir) / "matrix_run.log"
            test_hist_dir = Path(tmp_dir) / "history"
            with patch("run_matrix.RUN_LOG", test_run_log), \
                 patch("run_matrix.HISTORY_DIR", test_hist_dir), \
                 patch("sys.argv", ["run_matrix.py", "--endpoint", "http://127.0.0.1:9090", "--presets-file", "presets.ini", "--cache-dir", "/cache"]):
                
                orig_stdout = sys.stdout
                orig_stderr = sys.stderr
                try:
                    run_matrix.main()
                    mock_run_matrix.assert_called_once_with(
                        endpoint="http://127.0.0.1:9090",
                        presets_file="presets.ini",
                        cache_dir="/cache"
                    )
                    mock_atexit.assert_called_once()
                    cleanup_fn = mock_atexit.call_args[0][0]
                    cleanup_fn()
                finally:
                    sys.stdout = orig_stdout
                    sys.stderr = orig_stderr


if __name__ == "__main__":
    unittest.main()
