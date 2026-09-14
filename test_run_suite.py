import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

import run_suite


class TestRunSuiteConstants(unittest.TestCase):
    """Test suite constants and configuration definitions."""

    def test_quant_types_structure(self):
        self.assertIsInstance(run_suite.QUANT_TYPES, tuple)
        self.assertGreater(len(run_suite.QUANT_TYPES), 0)
        for item in run_suite.QUANT_TYPES:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            q_lower, q_orig = item
            self.assertIsInstance(q_lower, str)
            self.assertIsInstance(q_orig, str)
            self.assertEqual(q_lower, q_orig.lower())

        expected_quants = [
            "q4_k_s",
            "q4_k_m",
            "q4_k_l",
            "q4_k_xl",
            "q5_k_s",
            "q5_k_m",
            "q8_0",
            "f16",
        ]
        defined_quants = [q[0] for q in run_suite.QUANT_TYPES]
        for eq in expected_quants:
            self.assertIn(eq, defined_quants)

    def test_quant_priorities(self):
        self.assertIsInstance(run_suite.QUANT_PRIORITIES, tuple)
        self.assertEqual(run_suite.QUANT_PRIORITIES, ("q5_1", "q8_0", "q4_0", "f16"))


class TestGetModelSettings(unittest.TestCase):
    """Test model settings extraction and fallback logic in get_model_settings."""

    def test_base_quant_parsing_from_target_model_name(self):
        test_cases = [
            ("qwen3.6-27b-q4_k_m", "Q4_K_M"),
            ("llama-3-8b-Q8_0.gguf", "Q8_0"),
            ("phi-3-mini-f16", "f16"),
            ("q5_k_s-model", "Q5_K_S"),
            ("my-custom-model", "Unknown"),
        ]
        for model_name, expected_quant in test_cases:
            with self.subTest(model=model_name):
                with patch("run_suite.requests.get") as mock_get:
                    mock_get.side_effect = requests.exceptions.ConnectionError(
                        "Offline"
                    )
                    settings = run_suite.get_model_settings(
                        "http://127.0.0.1:8081", model_name
                    )
                    self.assertEqual(settings["base_quantization"], expected_quant)
                    self.assertEqual(settings["model_name"], model_name)

    def test_valid_endpoint_json_response_with_args(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "qwen3.6-27b-q4_k_m",
                    "status": {
                        "args": [
                            "--threads",
                            "8",
                            "--batch-size",
                            "512",
                            "--ubatch-size",
                            "128",
                            "--cache-type-k",
                            "q8_0",
                            "--cache-type-v",
                            "q4_0",
                            "--spec-draft",
                            "true",
                        ],
                        "preset": "",
                    },
                }
            ]
        }

        with patch("run_suite.requests.get", return_value=mock_response):
            settings = run_suite.get_model_settings(
                "http://127.0.0.1:8081", "qwen3.6-27b-q4_k_m"
            )

        self.assertEqual(settings["model_name"], "qwen3.6-27b-q4_k_m")
        self.assertEqual(settings["threads"], 8)
        self.assertEqual(settings["batch_size"], 512)
        self.assertEqual(settings["ubatch_size"], 128)
        self.assertEqual(settings["kv_cache_quant_k"], "q8_0")
        self.assertEqual(settings["kv_cache_quant_v"], "q4_0")
        self.assertEqual(settings["kv_cache_quant"], "q8_0")
        self.assertEqual(settings["base_quantization"], "Q4_K_M")
        self.assertEqual(settings["speculative_draft_type"], "ngram")

    def test_valid_endpoint_json_response_with_preset(self):
        preset_content = """
        threads = 16
        batch-size = 1024
        ubatch-size = 256
        cache-type-k = q5_1
        cache-type-v = q8_0
        draft = lookup
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen3.6-27b", "status": {"args": [], "preset": preset_content}}
            ]
        }

        with patch("run_suite.requests.get", return_value=mock_response):
            settings = run_suite.get_model_settings(
                "http://127.0.0.1:8081", "qwen3.6-27b"
            )

        self.assertEqual(settings["threads"], 16)
        self.assertEqual(settings["batch_size"], 1024)
        self.assertEqual(settings["ubatch_size"], 256)
        self.assertEqual(settings["kv_cache_quant_k"], "q5_1")
        self.assertEqual(settings["kv_cache_quant_v"], "q8_0")
        self.assertEqual(settings["kv_cache_quant"], "q5_1")
        self.assertEqual(settings["speculative_draft_type"], "ngram")

    def test_cache_type_v_fallback_when_k_absent(self):
        # Case 1: In CLI args
        mock_response_args = MagicMock()
        mock_response_args.status_code = 200
        mock_response_args.json.return_value = {
            "data": [
                {
                    "id": "custom-model",
                    "status": {"args": ["--cache-type-v", "q4_0"], "preset": ""},
                }
            ]
        }

        with patch("run_suite.requests.get", return_value=mock_response_args):
            settings = run_suite.get_model_settings(
                "http://127.0.0.1:8081", "custom-model"
            )
            self.assertEqual(settings["kv_cache_quant_v"], "q4_0")
            self.assertEqual(settings["kv_cache_quant_k"], "Unknown")
            self.assertEqual(settings["kv_cache_quant"], "q4_0")

        # Case 2: In preset
        mock_response_preset = MagicMock()
        mock_response_preset.status_code = 200
        mock_response_preset.json.return_value = {
            "data": [
                {
                    "id": "custom-model",
                    "status": {"args": [], "preset": "cache-type-v = q8_0"},
                }
            ]
        }

        with patch("run_suite.requests.get", return_value=mock_response_preset):
            settings = run_suite.get_model_settings(
                "http://127.0.0.1:8081", "custom-model"
            )
            self.assertEqual(settings["kv_cache_quant_v"], "q8_0")
            self.assertEqual(settings["kv_cache_quant_k"], "Unknown")
            self.assertEqual(settings["kv_cache_quant"], "q8_0")

    def test_speculative_draft_type_resolution(self):
        cases = [
            ("model-mtp-q8_0", [], "", "ngram"),
            ("plain-model", ["--lookup-cache", "true"], "", "ngram"),
            ("plain-model", [], "ngram = true", "ngram"),
            ("plain-model", ["--draft-model", "foo"], "", "ngram"),
            ("plain-model", [], "", "None"),
        ]
        for model_id, args, preset, expected_spec in cases:
            with self.subTest(model_id=model_id, args=args, preset=preset):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "data": [
                        {"id": model_id, "status": {"args": args, "preset": preset}}
                    ]
                }
                with patch("run_suite.requests.get", return_value=mock_resp):
                    settings = run_suite.get_model_settings(
                        "http://127.0.0.1:8081", model_id
                    )
                    self.assertEqual(settings["speculative_draft_type"], expected_spec)

    def test_int_parsing_fails_on_malformed_args_safe_fallback(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "model-malformed-args",
                    "status": {
                        "args": [
                            "--threads",
                            "not_an_int",
                            "--batch-size",
                            "invalid_batch",
                            "--ubatch-size",
                            "bad_ubatch",
                        ],
                        "preset": "",
                    },
                }
            ]
        }

        with patch("run_suite.requests.get", return_value=mock_response):
            settings = run_suite.get_model_settings(
                "http://127.0.0.1:8081", "model-malformed-args"
            )

        self.assertIsNone(settings["threads"])
        self.assertIsNone(settings["batch_size"])
        self.assertIsNone(settings["ubatch_size"])

    def test_int_parsing_fails_on_malformed_preset_safe_fallback(self):
        preset_content = """
        threads = not_an_int
        batch-size = invalid_val
        ubatch-size = malformed
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "model-malformed-preset",
                    "status": {"args": [], "preset": preset_content},
                }
            ]
        }

        with patch("run_suite.requests.get", return_value=mock_response):
            settings = run_suite.get_model_settings(
                "http://127.0.0.1:8081", "model-malformed-preset"
            )

        self.assertIsNone(settings["threads"])
        self.assertIsNone(settings["batch_size"])
        self.assertIsNone(settings["ubatch_size"])

    def test_args_boundary_missing_values(self):
        boundary_flags = [
            ("--threads", "threads", None),
            ("--batch-size", "batch_size", None),
            ("--ubatch-size", "ubatch_size", None),
            ("--cache-type-k", "kv_cache_quant_k", "Unknown"),
            ("--cache-type-v", "kv_cache_quant_v", "Unknown"),
        ]

        for flag, key, default_val in boundary_flags:
            with self.subTest(flag=flag):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "data": [
                        {
                            "id": "model-trailing-flag",
                            "status": {"args": [flag], "preset": "no_equals_line"},
                        }
                    ]
                }

                with patch("run_suite.requests.get", return_value=mock_response):
                    settings = run_suite.get_model_settings(
                        "http://127.0.0.1:8081", "model-trailing-flag"
                    )

                self.assertEqual(settings[key], default_val)
                self.assertEqual(settings["kv_cache_quant"], "Unknown")

    def test_connection_error_fallback(self):
        exceptions = [
            requests.exceptions.RequestException("Request failed"),
            requests.exceptions.ConnectionError("Connection refused"),
            requests.exceptions.Timeout("Read timeout"),
            ConnectionError("OS connection error"),
            ValueError("Malformed response JSON"),
        ]
        for exc in exceptions:
            with self.subTest(exc=type(exc).__name__):
                with patch("run_suite.requests.get") as mock_get:
                    if isinstance(exc, ValueError):
                        mock_resp = MagicMock()
                        mock_resp.status_code = 200
                        mock_resp.json.side_effect = exc
                        mock_get.return_value = mock_resp
                    else:
                        mock_get.side_effect = exc

                    settings = run_suite.get_model_settings(
                        "http://127.0.0.1:8081", "test-model-q4_k_s"
                    )

                    self.assertEqual(settings["model_name"], "test-model-q4_k_s")
                    self.assertEqual(settings["base_quantization"], "Q4_K_S")
                    self.assertIsNone(settings["threads"])
                    self.assertIsNone(settings["batch_size"])
                    self.assertIsNone(settings["ubatch_size"])
                    self.assertEqual(settings["kv_cache_quant"], "Unknown")
                    self.assertEqual(settings["speculative_draft_type"], "None")

    def test_target_model_not_found_or_empty_data(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "other-model"}]}

        with patch("run_suite.requests.get", return_value=mock_response):
            settings = run_suite.get_model_settings(
                "http://127.0.0.1:8081", "desired-model"
            )

        self.assertEqual(settings["model_name"], "desired-model")
        self.assertIsNone(settings["threads"])
        self.assertEqual(settings["kv_cache_quant"], "Unknown")

    def test_non_200_status_code(self):
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("run_suite.requests.get", return_value=mock_response):
            settings = run_suite.get_model_settings(
                "http://127.0.0.1:8081", "any-model"
            )

        self.assertIsNone(settings["threads"])
        self.assertEqual(settings["kv_cache_quant"], "Unknown")

    def test_advanced_benchmarks_integration(self):
        mock_adv = MagicMock()
        mock_adv.map_repo_to_preset_alias.return_value = "custom-alias"
        mock_adv.get_preset_metadata.return_value = {
            "flash_attn": "true",
            "parallel": "4",
        }

        with patch("run_suite.advanced_benchmarks", mock_adv):
            settings = run_suite.get_model_settings("http://127.0.0.1:8081", "my-model")
            self.assertEqual(settings["profile_alias"], "custom-alias")
            self.assertEqual(settings["flash_attn"], "true")
            self.assertEqual(settings["parallel"], "4")

    def test_advanced_benchmarks_none(self):
        with patch("run_suite.advanced_benchmarks", None):
            with patch("run_suite.requests.get") as mock_get:
                mock_get.side_effect = requests.exceptions.ConnectionError()
                settings = run_suite.get_model_settings(
                    "http://127.0.0.1:8081", "my-model"
                )
                self.assertNotIn("profile_alias", settings)


class TestRunThroughput(unittest.TestCase):
    """Test throughput benchmark execution and regex parsing."""

    @patch("run_suite.subprocess.run")
    def test_run_throughput_success(self, mock_run):
        stdout_content = """
        === Benchmark run ===
        Prompt Eval (p/s) : 120.0 tokens/sec (TTFT: 0.15s)
        Generation (t/s) : 40.0 tokens/sec
        Prompt Eval (p/s) : 140.0 tokens/sec (TTFT: 0.25s)
        Generation (t/s) : 60.0 tokens/sec
        """
        mock_run.return_value = MagicMock(
            returncode=0, stdout=stdout_content, stderr=""
        )

        result = run_suite.run_throughput("http://127.0.0.1:8081", "test-model")

        self.assertAlmostEqual(result["prefill_speed"], 130.0)
        self.assertAlmostEqual(result["decode_speed"], 50.0)
        self.assertAlmostEqual(result["ttft"], 0.20)

    @patch("run_suite.subprocess.run")
    def test_run_throughput_failure_exit_code(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Crash")

        result = run_suite.run_throughput("http://127.0.0.1:8081", "test-model")

        self.assertEqual(result["prefill_speed"], 0.0)
        self.assertEqual(result["decode_speed"], 0.0)
        self.assertEqual(result["ttft"], 0.0)

    @patch("run_suite.subprocess.run")
    def test_run_throughput_no_matches_in_stdout(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="No benchmarks here", stderr=""
        )

        result = run_suite.run_throughput("http://127.0.0.1:8081", "test-model")

        self.assertEqual(result["prefill_speed"], 0.0)
        self.assertEqual(result["decode_speed"], 0.0)
        self.assertEqual(result["ttft"], 0.0)

    @patch("run_suite.os.chmod")
    @patch("run_suite.os.access")
    @patch("run_suite.os.path.exists")
    @patch("run_suite.subprocess.run")
    def test_run_throughput_chmod_script_if_not_executable(
        self, mock_run, mock_exists, mock_access, mock_chmod
    ):
        mock_exists.return_value = True
        mock_access.return_value = False
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        run_suite.run_throughput("http://127.0.0.1:8081", "test-model")

        mock_chmod.assert_called_once()
        self.assertEqual(mock_chmod.call_args[0][1], 0o755)


class TestRunReasoning(unittest.TestCase):
    """Test reasoning benchmark execution and pass/fail mapping."""

    def test_run_reasoning_all_pass(self):
        mock_adv = MagicMock()
        mock_adv.run_needle_test.return_value = {"passed": True}
        mock_adv.run_ruler_test.return_value = {"passed": True}
        mock_adv.run_longbench_test.return_value = {"passed": True}
        mock_adv.run_swe_test.return_value = {"passed": True}

        with patch("run_suite.advanced_benchmarks", mock_adv):
            results = run_suite.run_reasoning(
                "http://127.0.0.1:8081", "test-model", tokens=1000
            )

        self.assertEqual(results["needle"], "Pass")
        self.assertEqual(results["ruler"], "Pass")
        self.assertEqual(results["longbench"], "Pass")
        self.assertEqual(results["swe_bench"], "Pass")

    def test_run_reasoning_all_fail(self):
        mock_adv = MagicMock()
        mock_adv.run_needle_test.return_value = {"passed": False}
        mock_adv.run_ruler_test.return_value = {"passed": False}
        mock_adv.run_longbench_test.return_value = None
        mock_adv.run_swe_test.return_value = {}

        with patch("run_suite.advanced_benchmarks", mock_adv):
            results = run_suite.run_reasoning(
                "http://127.0.0.1:8081", "test-model", tokens=1000
            )

        self.assertEqual(results["needle"], "Fail")
        self.assertEqual(results["ruler"], "Fail")
        self.assertEqual(results["longbench"], "Fail")
        self.assertEqual(results["swe_bench"], "Fail")

    def test_run_reasoning_exceptions_caught(self):
        mock_adv = MagicMock()
        mock_adv.run_needle_test.side_effect = RuntimeError("Needle error")
        mock_adv.run_ruler_test.side_effect = Exception("RULER timeout")
        mock_adv.run_longbench_test.side_effect = ValueError("Longbench bad val")
        mock_adv.run_swe_test.side_effect = KeyError("SWE key error")

        with patch("run_suite.advanced_benchmarks", mock_adv):
            results = run_suite.run_reasoning(
                "http://127.0.0.1:8081", "test-model", tokens=1000
            )

        self.assertEqual(results["needle"], "Fail")
        self.assertEqual(results["ruler"], "Fail")
        self.assertEqual(results["longbench"], "Fail")
        self.assertEqual(results["swe_bench"], "Fail")

    def test_run_reasoning_no_advanced_benchmarks(self):
        with patch("run_suite.advanced_benchmarks", None):
            results = run_suite.run_reasoning(
                "http://127.0.0.1:8081", "test-model", tokens=1000
            )

        self.assertEqual(
            results,
            {
                "needle": "Fail",
                "ruler": "Fail",
                "longbench": "Fail",
                "swe_bench": "Fail",
            },
        )


class TestRunAgentic(unittest.TestCase):
    """Test agentic benchmark execution and result forwarding."""

    def test_run_agentic_success(self):
        mock_agentic = MagicMock()
        mock_agentic.run_agentic_suite.return_value = {
            "suite": "standalone-agentic",
            "tasks_total": 2,
            "tasks_passed": 2,
            "average_turns": 1.5,
            "total_tool_calls": 4,
            "tasks": [],
            "token_breakdown": {
                "prompt_tokens": 500,
                "reasoning_tokens": 100,
                "completion_tokens": 200,
            },
        }

        with patch("run_suite.agentic_benchmarks", mock_agentic):
            res = run_suite.run_agentic(
                "http://127.0.0.1:8081",
                "test-model",
                task_filter="fix-syntax",
                api_key="secret-key",
                max_tokens=4096,
            )

        mock_agentic.run_agentic_suite.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            task_filter="fix-syntax",
            max_tokens=4096,
            api_key="secret-key",
        )
        self.assertEqual(res["tasks_passed"], 2)

    def test_run_agentic_exception_caught(self):
        mock_agentic = MagicMock()
        mock_agentic.run_agentic_suite.side_effect = RuntimeError("Harness exploded")

        with patch("run_suite.agentic_benchmarks", mock_agentic):
            res = run_suite.run_agentic("http://127.0.0.1:8081", "test-model")

        self.assertEqual(res["tasks_total"], 0)
        self.assertEqual(res["tasks_passed"], 0)

    def test_run_agentic_no_module(self):
        with patch("run_suite.agentic_benchmarks", None):
            res = run_suite.run_agentic("http://127.0.0.1:8081", "test-model")

        self.assertEqual(res["tasks_total"], 0)
        self.assertEqual(res["tasks_passed"], 0)

    def test_run_agentic_env_api_key(self):
        mock_agentic = MagicMock()
        mock_agentic.run_agentic_suite.return_value = {"tasks_passed": 1}

        with (
            patch("run_suite.agentic_benchmarks", mock_agentic),
            patch.dict(os.environ, {"API_KEY": "env-agentic-key"}, clear=True),
        ):
            run_suite.run_agentic("http://127.0.0.1:8081", "test-model")

        mock_agentic.run_agentic_suite.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            task_filter="all",
            max_tokens=16384,
            api_key="env-agentic-key",
        )


class TestRunKld(unittest.TestCase):
    """Test KLD benchmark execution and stdout parsing."""

    @patch("run_suite.subprocess.run")
    def test_run_kld_command_construction(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Both model and corpus
        run_suite.run_kld("path/to/model.gguf", "corpus.txt")
        mock_run.assert_called_with(
            [
                sys.executable,
                "kld_benchmark.py",
                "--model",
                "path/to/model.gguf",
                "--corpus",
                "corpus.txt",
            ],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(run_suite.__file__)),
        )

        # Neither model nor corpus
        mock_run.reset_mock()
        run_suite.run_kld(None, None)
        mock_run.assert_called_with(
            [sys.executable, "kld_benchmark.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(run_suite.__file__)),
        )

    @patch("run_suite.subprocess.run")
    def test_run_kld_table_parsing(self, mock_run):
        stdout_content = """
        =========================================================
        | KV Cache | Perplexity | Mean KLD | Same Top Match |
        |---|---|---|---|
        | q8_0 (Baseline) | 5.25 | 0.0000 | 100.0% |
        | q5_1 | 5.38 | 0.0125 | 98.2% |
        | q4_0 | N/A | N/A | N/A |
        | malformed | invalid_float | 0.05 | 95.0% |
        =========================================================
        """
        mock_run.return_value = MagicMock(
            returncode=0, stdout=stdout_content, stderr="Some stderr info"
        )

        kld_results = run_suite.run_kld("model.gguf", "corpus.txt")

        self.assertIn("q8_0", kld_results)
        self.assertEqual(kld_results["q8_0"]["perplexity"], 5.25)
        self.assertEqual(kld_results["q8_0"]["mean_kld"], 0.0)
        self.assertEqual(kld_results["q8_0"]["same_top_match_percent"], 100.0)

        self.assertIn("q5_1", kld_results)
        self.assertEqual(kld_results["q5_1"]["perplexity"], 5.38)
        self.assertEqual(kld_results["q5_1"]["mean_kld"], 0.0125)
        self.assertEqual(kld_results["q5_1"]["same_top_match_percent"], 98.2)

        self.assertIn("q4_0", kld_results)
        self.assertIsNone(kld_results["q4_0"]["perplexity"])
        self.assertIsNone(kld_results["q4_0"]["mean_kld"])
        self.assertIsNone(kld_results["q4_0"]["same_top_match_percent"])

        self.assertNotIn("malformed", kld_results)


class TestQuantizationPriorityAndMain(unittest.TestCase):
    """Test quantization priority resolution logic and unified suite main() orchestration."""

    def test_quantization_priority_resolution_exact_match(self):
        model_settings = {"kv_cache_quant": "q4_0"}
        kld_results = {
            "q5_1": {
                "perplexity": 5.2,
                "mean_kld": 0.01,
                "same_top_match_percent": 99.0,
            },
            "q4_0": {
                "perplexity": 5.5,
                "mean_kld": 0.03,
                "same_top_match_percent": 95.0,
            },
        }
        quantization_loss = {
            "perplexity": None,
            "mean_kld": None,
            "same_top_match_percent": None,
        }

        # Simulate resolution logic from main()
        kv_quant = model_settings.get("kv_cache_quant", "q5_1")
        if not kv_quant or kv_quant == "Unknown":
            kv_quant = "q5_1"
        match_quant = kv_quant
        if match_quant not in kld_results:
            for k in run_suite.QUANT_PRIORITIES:
                if k in kld_results:
                    match_quant = k
                    break
        if match_quant in kld_results:
            quantization_loss.update(kld_results[match_quant])

        self.assertEqual(match_quant, "q4_0")
        self.assertEqual(quantization_loss["perplexity"], 5.5)

    def test_quantization_priority_resolution_fallback_order(self):
        # Case 1: Unknown kv_cache_quant with both q5_1 and q8_0 available -> picks q5_1 (first priority)
        kld_results_all = {
            "q8_0": {
                "perplexity": 5.1,
                "mean_kld": 0.0,
                "same_top_match_percent": 100.0,
            },
            "q5_1": {
                "perplexity": 5.2,
                "mean_kld": 0.01,
                "same_top_match_percent": 99.0,
            },
            "f16": {
                "perplexity": 5.0,
                "mean_kld": 0.0,
                "same_top_match_percent": 100.0,
            },
        }
        model_settings = {"kv_cache_quant": "Unknown"}

        kv_quant = model_settings.get("kv_cache_quant", "q5_1")
        if not kv_quant or kv_quant == "Unknown":
            kv_quant = "q5_1"
        match_quant = kv_quant
        if match_quant not in kld_results_all:
            for k in run_suite.QUANT_PRIORITIES:
                if k in kld_results_all:
                    match_quant = k
                    break

        self.assertEqual(match_quant, "q5_1")

        # Case 2: Unknown kv_cache_quant with q5_1 missing -> falls back to q8_0
        kld_results_no_q5 = {
            "q8_0": {
                "perplexity": 5.1,
                "mean_kld": 0.0,
                "same_top_match_percent": 100.0,
            },
            "f16": {
                "perplexity": 5.0,
                "mean_kld": 0.0,
                "same_top_match_percent": 100.0,
            },
        }
        kv_quant = "q5_1"
        match_quant = kv_quant
        if match_quant not in kld_results_no_q5:
            for k in run_suite.QUANT_PRIORITIES:
                if k in kld_results_no_q5:
                    match_quant = k
                    break

        self.assertEqual(match_quant, "q8_0")

        # Case 3: Unknown kv_cache_quant with only f16 available -> falls back to f16
        kld_results_only_f16 = {
            "f16": {"perplexity": 5.0, "mean_kld": 0.0, "same_top_match_percent": 100.0}
        }
        match_quant = "q5_1"
        if match_quant not in kld_results_only_f16:
            for k in run_suite.QUANT_PRIORITIES:
                if k in kld_results_only_f16:
                    match_quant = k
                    break

        self.assertEqual(match_quant, "f16")

    def test_main_orchestration_mode_all(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_script_file = os.path.join(tmp_dir, "run_suite.py")

            mock_throughput = {
                "prefill_speed": 125.0,
                "decode_speed": 45.0,
                "ttft": 0.18,
            }
            mock_reasoning = {
                "needle": "Pass",
                "ruler": "Pass",
                "longbench": "Pass",
                "swe_bench": "Pass",
            }
            mock_kld = {
                "q5_1": {
                    "perplexity": 5.32,
                    "mean_kld": 0.015,
                    "same_top_match_percent": 98.1,
                }
            }
            mock_agentic = {
                "suite": "standalone-agentic",
                "tasks_total": 1,
                "tasks_passed": 1,
                "average_turns": 2.0,
                "total_tool_calls": 3,
                "tasks": [],
                "token_breakdown": {
                    "prompt_tokens": 100,
                    "reasoning_tokens": 20,
                    "completion_tokens": 50,
                },
            }
            mock_settings = {
                "model_name": "Qwen3.6-27B",
                "base_quantization": "Q4_K_M",
                "kv_cache_quant": "Unknown",
                "kv_cache_quant_k": "Unknown",
                "kv_cache_quant_v": "Unknown",
                "threads": 8,
                "ubatch_size": 128,
                "batch_size": 512,
                "speculative_draft_type": "None",
            }

            test_args = [
                "run_suite.py",
                "--mode",
                "all",
                "--endpoint",
                "http://127.0.0.1:8081",
                "--model",
                "Qwen3.6-27B",
                "--tokens",
                "1000",
                "--gguf-path",
                "model.gguf",
                "--corpus",
                "corpus.txt",
            ]

            with (
                patch("sys.argv", test_args),
                patch.object(run_suite, "__file__", fake_script_file),
                patch(
                    "run_suite.run_throughput", return_value=mock_throughput
                ) as mock_tp_fn,
                patch(
                    "run_suite.run_reasoning", return_value=mock_reasoning
                ) as mock_reas_fn,
                patch("run_suite.run_agentic", return_value=mock_agentic) as mock_ag_fn,
                patch("run_suite.run_kld", return_value=mock_kld) as mock_kld_fn,
                patch(
                    "run_suite.get_model_settings", return_value=mock_settings
                ) as mock_settings_fn,
            ):
                run_suite.main()

                mock_tp_fn.assert_called_once_with(
                    "http://127.0.0.1:8081", "Qwen3.6-27B"
                )
                mock_reas_fn.assert_called_once_with(
                    "http://127.0.0.1:8081", "Qwen3.6-27B", 1000, max_tokens=16384
                )
                mock_ag_fn.assert_called_once_with(
                    "http://127.0.0.1:8081",
                    "Qwen3.6-27B",
                    task_filter="all",
                    max_tokens=16384,
                )
                mock_kld_fn.assert_called_once_with("model.gguf", "corpus.txt")
                mock_settings_fn.assert_called_once_with(
                    "http://127.0.0.1:8081", "Qwen3.6-27B"
                )

                history_dir = os.path.join(tmp_dir, "history")
                self.assertTrue(os.path.isdir(history_dir))
                files = os.listdir(history_dir)
                self.assertEqual(len(files), 1)
                self.assertTrue(
                    files[0].startswith("run_") and files[0].endswith(".json")
                )

                with open(os.path.join(history_dir, files[0])) as f:
                    data = json.load(f)

                self.assertIn("run_metadata", data)
                self.assertEqual(
                    data["run_metadata"]["target_endpoint"], "http://127.0.0.1:8081"
                )
                self.assertEqual(data["throughput_metrics"]["prefill_speed"], 125.0)
                self.assertEqual(data["reasoning_accuracy"]["needle"], "Pass")
                self.assertEqual(data["agentic_metrics"]["tasks_passed"], 1)
                self.assertEqual(data["token_breakdown"]["prompt_tokens"], 100)
                self.assertEqual(data["quantization_loss"]["perplexity"], 5.32)
                # Check that Unknown kv_cache_quant got updated to matched quant
                self.assertEqual(data["model_settings"]["kv_cache_quant"], "q5_1")

    def test_main_quantization_fallback_resolution_in_orchestration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_script_file = os.path.join(tmp_dir, "run_suite.py")

            # Model kv_cache_quant is a non-existent quant, fallback should resolve to q8_0
            mock_kld = {
                "q8_0": {
                    "perplexity": 5.10,
                    "mean_kld": 0.000,
                    "same_top_match_percent": 100.0,
                }
            }
            mock_settings = {
                "model_name": "TestModel",
                "base_quantization": "Q8_0",
                "kv_cache_quant": "non_existent_quant",
            }

            test_args = ["run_suite.py", "--mode", "kld"]

            with (
                patch("sys.argv", test_args),
                patch.object(run_suite, "__file__", fake_script_file),
                patch("run_suite.run_kld", return_value=mock_kld),
                patch("run_suite.get_model_settings", return_value=mock_settings),
            ):
                run_suite.main()

                history_dir = os.path.join(tmp_dir, "history")
                files = os.listdir(history_dir)
                with open(os.path.join(history_dir, files[0])) as f:
                    data = json.load(f)

                self.assertEqual(data["quantization_loss"]["perplexity"], 5.10)

    def test_main_kld_results_no_matching_quant(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_script_file = os.path.join(tmp_dir, "run_suite.py")
            mock_kld = {"completely_unknown_quant": {"perplexity": 5.0}}
            mock_settings = {
                "model_name": "TestModel",
                "kv_cache_quant": "some_other_quant",
            }
            test_args = ["run_suite.py", "--mode", "kld"]

            with (
                patch("sys.argv", test_args),
                patch.object(run_suite, "__file__", fake_script_file),
                patch("run_suite.run_kld", return_value=mock_kld),
                patch("run_suite.get_model_settings", return_value=mock_settings),
            ):
                run_suite.main()

                history_dir = os.path.join(tmp_dir, "history")
                files = os.listdir(history_dir)
                with open(os.path.join(history_dir, files[0])) as f:
                    data = json.load(f)

                self.assertIsNone(data["quantization_loss"]["perplexity"])

    def test_main_orchestration_mode_subsets(self):
        modes = [
            ("throughput", True, False, False, False),
            ("reasoning", False, True, False, False),
            ("agentic", False, False, True, False),
            ("kld", False, False, False, True),
        ]
        for mode, expect_tp, expect_reas, expect_ag, expect_kld in modes:
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    fake_script_file = os.path.join(tmp_dir, "run_suite.py")
                    test_args = ["run_suite.py", "--mode", mode]

                    with (
                        patch("sys.argv", test_args),
                        patch.object(run_suite, "__file__", fake_script_file),
                        patch(
                            "run_suite.run_throughput", return_value={}
                        ) as mock_tp_fn,
                        patch(
                            "run_suite.run_reasoning", return_value={}
                        ) as mock_reas_fn,
                        patch("run_suite.run_agentic", return_value={}) as mock_ag_fn,
                        patch("run_suite.run_kld", return_value={}) as mock_kld_fn,
                        patch(
                            "run_suite.get_model_settings",
                            return_value={"kv_cache_quant": "q5_1"},
                        ),
                    ):
                        run_suite.main()

                        self.assertEqual(mock_tp_fn.called, expect_tp)
                        self.assertEqual(mock_reas_fn.called, expect_reas)
                        self.assertEqual(mock_ag_fn.called, expect_ag)
                        self.assertEqual(mock_kld_fn.called, expect_kld)

    def test_main_cli_agentic_tasks_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_script_file = os.path.join(tmp_dir, "run_suite.py")
            test_args = [
                "run_suite.py",
                "--mode",
                "agentic",
                "--agentic-tasks",
                "fix-syntax,git-repair",
            ]
            with (
                patch("sys.argv", test_args),
                patch.object(run_suite, "__file__", fake_script_file),
                patch(
                    "run_suite.run_agentic",
                    return_value={"suite": "standalone-agentic", "token_breakdown": {}},
                ) as mock_ag,
                patch(
                    "run_suite.get_model_settings",
                    return_value={"kv_cache_quant": "q5_1"},
                ),
            ):
                run_suite.main()
                mock_ag.assert_called_once_with(
                    "http://127.0.0.1:8083",
                    "Qwen3.6-27B",
                    task_filter="fix-syntax,git-repair",
                    max_tokens=16384,
                )

    def test_run_kld_line_with_fewer_than_four_parts(self):
        fake_stdout = "| Part1 | Part2 |\n"
        with patch("run_suite.subprocess.run") as mock_subproc:
            mock_subproc.return_value = MagicMock(
                returncode=0, stdout=fake_stdout, stderr=""
            )
            res = run_suite.run_kld(None, None)
            self.assertEqual(res, {})

    def test_main_default_endpoint_8083(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_script_file = os.path.join(tmp_dir, "run_suite.py")
            mock_settings = {"model_name": "TestModel", "kv_cache_quant": "q8_0"}
            test_args = ["run_suite.py", "--mode", "kld"]

            with (
                patch("sys.argv", test_args),
                patch.object(run_suite, "__file__", fake_script_file),
                patch("run_suite.run_kld", return_value={}),
                patch(
                    "run_suite.get_model_settings", return_value=mock_settings
                ) as mock_get_settings,
            ):
                run_suite.main()
                mock_get_settings.assert_called_once_with(
                    "http://127.0.0.1:8083", "Qwen3.6-27B"
                )

                history_dir = os.path.join(tmp_dir, "history")
                files = os.listdir(history_dir)
                with open(os.path.join(history_dir, files[0])) as f:
                    data = json.load(f)
                self.assertEqual(
                    data["run_metadata"]["target_endpoint"], "http://127.0.0.1:8083"
                )


class TestRunSuiteApiKeyAuth(unittest.TestCase):
    """Test API key authentication in get_model_settings, run_throughput, run_reasoning, and main."""

    @patch("run_suite.requests.get")
    def test_get_model_settings_with_api_key_bearer_header(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "my-model", "status": {}}]}
        mock_get.return_value = mock_resp

        settings = run_suite.get_model_settings(
            "http://127.0.0.1:8081", "my-model", api_key="secret-suite-key"
        )
        self.assertEqual(settings["model_name"], "my-model")
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertEqual(
            kwargs.get("headers"), {"Authorization": "Bearer secret-suite-key"}
        )

    @patch("run_suite.subprocess.run")
    def test_run_throughput_passes_api_key_in_env(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        run_suite.run_throughput(
            "http://127.0.0.1:8081", "test-model", api_key="throughput-secret"
        )
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertIn("env", kwargs)
        self.assertEqual(kwargs["env"].get("OPENAI_API_KEY"), "throughput-secret")

    def test_run_reasoning_forwards_api_key_to_advanced_benchmarks(self):
        mock_adv = MagicMock()
        mock_adv.run_needle_test.return_value = {"passed": True}
        mock_adv.run_ruler_test.return_value = {"passed": True}
        mock_adv.run_longbench_test.return_value = {"passed": True}
        mock_adv.run_swe_test.return_value = {"passed": True}

        with patch("run_suite.advanced_benchmarks", mock_adv):
            results = run_suite.run_reasoning(
                "http://127.0.0.1:8081",
                "test-model",
                tokens=1000,
                api_key="reasoning-secret",
            )

        self.assertEqual(results["needle"], "Pass")
        mock_adv.run_needle_test.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            tokens=1000,
            max_tokens=16384,
            api_key="reasoning-secret",
        )
        mock_adv.run_ruler_test.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            tokens=1000,
            max_tokens=16384,
            api_key="reasoning-secret",
        )
        mock_adv.run_longbench_test.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            tokens=1000,
            max_tokens=16384,
            api_key="reasoning-secret",
        )
        mock_adv.run_swe_test.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            max_tokens=16384,
            api_key="reasoning-secret",
        )

    def test_run_reasoning_forwards_custom_max_tokens(self):
        mock_adv = MagicMock()
        mock_adv.run_needle_test.return_value = {"passed": True}
        mock_adv.run_ruler_test.return_value = {"passed": True}
        mock_adv.run_longbench_test.return_value = {"passed": True}
        mock_adv.run_swe_test.return_value = {"passed": True}

        with patch("run_suite.advanced_benchmarks", mock_adv):
            results = run_suite.run_reasoning(
                "http://127.0.0.1:8081",
                "test-model",
                tokens=1000,
                api_key="reasoning-secret",
                max_tokens=32768,
            )

        self.assertEqual(results["swe_bench"], "Pass")
        mock_adv.run_needle_test.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            tokens=1000,
            max_tokens=32768,
            api_key="reasoning-secret",
        )
        mock_adv.run_ruler_test.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            tokens=1000,
            max_tokens=32768,
            api_key="reasoning-secret",
        )
        mock_adv.run_longbench_test.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            tokens=1000,
            max_tokens=32768,
            api_key="reasoning-secret",
        )
        mock_adv.run_swe_test.assert_called_once_with(
            "http://127.0.0.1:8081",
            "test-model",
            max_tokens=32768,
            api_key="reasoning-secret",
        )

    def test_main_cli_api_key_argument(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_script_file = os.path.join(tmp_dir, "run_suite.py")
            mock_settings = {"model_name": "TestModel", "kv_cache_quant": "q8_0"}
            test_args = [
                "run_suite.py",
                "--mode",
                "all",
                "--api-key",
                "cli-test-bearer-key",
            ]

            with (
                patch("sys.argv", test_args),
                patch.object(run_suite, "__file__", fake_script_file),
                patch("run_suite.run_throughput", return_value={}) as mock_tp,
                patch("run_suite.run_reasoning", return_value={}) as mock_reas,
                patch("run_suite.run_kld", return_value={}),
                patch(
                    "run_suite.get_model_settings", return_value=mock_settings
                ) as mock_settings_fn,
            ):
                run_suite.main()

                mock_tp.assert_called_once()
                self.assertEqual(
                    mock_tp.call_args.kwargs.get("api_key"), "cli-test-bearer-key"
                )
                mock_reas.assert_called_once()
                self.assertEqual(
                    mock_reas.call_args.kwargs.get("api_key"), "cli-test-bearer-key"
                )
                mock_settings_fn.assert_called_once()
                self.assertEqual(
                    mock_settings_fn.call_args.kwargs.get("api_key"),
                    "cli-test-bearer-key",
                )

    @patch("run_suite.requests.get")
    def test_get_model_settings_env_fallback(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "m1"}]}
        mock_get.return_value = mock_resp

        # API_KEY priority
        with patch.dict(
            os.environ, {"API_KEY": "env-api-val", "OPENAI_API_KEY": "env-openai-val"}
        ):
            run_suite.get_model_settings("http://127.0.0.1:8081", "m1")
            _, kwargs = mock_get.call_args
            self.assertEqual(
                kwargs.get("headers"), {"Authorization": "Bearer env-api-val"}
            )

        # OPENAI_API_KEY fallback
        mock_get.reset_mock()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-val"}, clear=True):
            run_suite.get_model_settings("http://127.0.0.1:8081", "m1")
            _, kwargs = mock_get.call_args
            self.assertEqual(
                kwargs.get("headers"), {"Authorization": "Bearer env-openai-val"}
            )

    @patch("run_suite.subprocess.run")
    def test_run_throughput_env_fallback(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with patch.dict(os.environ, {"API_KEY": "env-tp-key"}, clear=True):
            run_suite.run_throughput("http://127.0.0.1:8081", "m1")
            _, kwargs = mock_run.call_args
            self.assertEqual(kwargs["env"].get("OPENAI_API_KEY"), "env-tp-key")

    def test_run_reasoning_env_fallback(self):
        mock_adv = MagicMock()
        mock_adv.run_needle_test.return_value = {"passed": True}
        mock_adv.run_ruler_test.return_value = {"passed": True}
        mock_adv.run_longbench_test.return_value = {"passed": True}
        mock_adv.run_swe_test.return_value = {"passed": True}

        with patch("run_suite.advanced_benchmarks", mock_adv):
            with patch.dict(os.environ, {"API_KEY": "env-reas-key"}, clear=True):
                run_suite.run_reasoning("http://127.0.0.1:8081", "m1", tokens=1000)
                mock_adv.run_needle_test.assert_called_once_with(
                    "http://127.0.0.1:8081",
                    "m1",
                    tokens=1000,
                    max_tokens=16384,
                    api_key="env-reas-key",
                )

    def test_main_cli_arguments_redaction_in_saved_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_script_file = os.path.join(tmp_dir, "run_suite.py")
            mock_settings = {"model_name": "TestModel", "kv_cache_quant": "q8_0"}
            test_args = [
                "run_suite.py",
                "--mode",
                "throughput",
                "--api-key",
                "my-confidential-suite-key",
            ]

            with (
                patch("sys.argv", test_args),
                patch.object(run_suite, "__file__", fake_script_file),
                patch(
                    "run_suite.run_throughput",
                    return_value={
                        "prefill_speed": 10.0,
                        "decode_speed": 5.0,
                        "ttft": 0.1,
                    },
                ),
                patch("run_suite.get_model_settings", return_value=mock_settings),
            ):
                run_suite.main()

                # Find written history json
                hist_dir = Path(tmp_dir) / "history"
                files = list(hist_dir.glob("run_*.json"))
                self.assertEqual(len(files), 1)
                with open(files[0], "r", encoding="utf-8") as f:
                    saved_data = json.load(f)

                saved_args = saved_data["run_metadata"]["cli_arguments"]
                self.assertIn("--api-key", saved_args)
                self.assertIn("********", saved_args)
                self.assertNotIn("my-confidential-suite-key", saved_args)


class TestRunSuiteContextTiersAndMaxTokens(unittest.TestCase):
    def test_run_suite_context_tiers_constant(self):
        self.assertEqual(run_suite.CONTEXT_TIERS["8k"], 8192)
        self.assertEqual(run_suite.CONTEXT_TIERS["32k"], 32768)
        self.assertEqual(run_suite.CONTEXT_TIERS["64k"], 65536)
        self.assertEqual(run_suite.CONTEXT_TIERS["128k"], 131072)
        self.assertEqual(run_suite.CONTEXT_TIERS["240k"], 240000)

    def test_run_suite_parse_context_tokens(self):
        self.assertEqual(run_suite.parse_context_tokens("8k"), 8192)
        self.assertEqual(run_suite.parse_context_tokens("32k"), 32768)
        self.assertEqual(run_suite.parse_context_tokens("64k"), 65536)
        self.assertEqual(run_suite.parse_context_tokens("128k"), 131072)
        self.assertEqual(run_suite.parse_context_tokens("240k"), 240000)
        self.assertEqual(run_suite.parse_context_tokens(" 32K "), 32768)
        self.assertEqual(run_suite.parse_context_tokens(5000), 5000)
        self.assertEqual(run_suite.parse_context_tokens("5000"), 5000)

        with self.assertRaises(ValueError):
            run_suite.parse_context_tokens("invalid")
        with self.assertRaises(TypeError):
            run_suite.parse_context_tokens(None)

    def test_main_cli_tier_string_and_max_tokens_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_script_file = os.path.join(tmp_dir, "run_suite.py")
            mock_settings = {"model_name": "TestModel", "kv_cache_quant": "q8_0"}
            test_args = [
                "run_suite.py",
                "--mode",
                "reasoning",
                "--tokens",
                "32k",
                "--max-tokens",
                "4096",
            ]

            with (
                patch("sys.argv", test_args),
                patch.object(run_suite, "__file__", fake_script_file),
                patch("run_suite.run_reasoning", return_value={}) as mock_reas_fn,
                patch("run_suite.get_model_settings", return_value=mock_settings),
            ):
                run_suite.main()
                mock_reas_fn.assert_called_once_with(
                    "http://127.0.0.1:8083", "Qwen3.6-27B", 32768, max_tokens=4096
                )

    def test_main_cli_240k_tier(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_script_file = os.path.join(tmp_dir, "run_suite.py")
            mock_settings = {"model_name": "TestModel", "kv_cache_quant": "q8_0"}
            test_args = [
                "run_suite.py",
                "--mode",
                "reasoning",
                "--tokens",
                "240k",
            ]

            with (
                patch("sys.argv", test_args),
                patch.object(run_suite, "__file__", fake_script_file),
                patch("run_suite.run_reasoning", return_value={}) as mock_reas_fn,
                patch("run_suite.get_model_settings", return_value=mock_settings),
            ):
                run_suite.main()
                mock_reas_fn.assert_called_once_with(
                    "http://127.0.0.1:8083", "Qwen3.6-27B", 240000, max_tokens=16384
                )

    def test_run_reasoning_forwards_tokens_and_max_tokens_all(self):
        mock_adv = MagicMock()
        mock_adv.run_needle_test.return_value = {"passed": True}
        mock_adv.run_ruler_test.return_value = {"passed": True}
        mock_adv.run_longbench_test.return_value = {"passed": True}
        mock_adv.run_swe_test.return_value = {"passed": True}

        with patch("run_suite.advanced_benchmarks", mock_adv):
            results = run_suite.run_reasoning(
                "http://127.0.0.1:8081",
                "m1",
                tokens="64k",
                max_tokens=8192,
            )

        self.assertEqual(results["needle"], "Pass")
        mock_adv.run_needle_test.assert_called_once_with(
            "http://127.0.0.1:8081", "m1", tokens=65536, max_tokens=8192
        )
        mock_adv.run_ruler_test.assert_called_once_with(
            "http://127.0.0.1:8081", "m1", tokens=65536, max_tokens=8192
        )
        mock_adv.run_longbench_test.assert_called_once_with(
            "http://127.0.0.1:8081", "m1", tokens=65536, max_tokens=8192
        )
        mock_adv.run_swe_test.assert_called_once_with(
            "http://127.0.0.1:8081", "m1", max_tokens=8192
        )


if __name__ == "__main__":
    unittest.main()
