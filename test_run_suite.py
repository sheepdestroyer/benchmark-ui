import unittest
from unittest.mock import MagicMock, patch
import requests

from run_suite import (
    QUANT_TYPES,
    QUANT_PRIORITIES,
    get_model_settings,
    parse_args,
)


class TestConstants(unittest.TestCase):
    def test_quant_types_structure(self):
        self.assertIsInstance(QUANT_TYPES, tuple)
        self.assertGreater(len(QUANT_TYPES), 0)
        for entry in QUANT_TYPES:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)
            self.assertIsInstance(entry[0], str)
            self.assertIsInstance(entry[1], str)
            self.assertEqual(entry[0], entry[0].lower())

    def test_quant_types_expected_values(self):
        quant_dict = dict(QUANT_TYPES)
        self.assertIn("q4_k_s", quant_dict)
        self.assertEqual(quant_dict["q4_k_s"], "Q4_K_S")
        self.assertIn("q8_0", quant_dict)
        self.assertEqual(quant_dict["q8_0"], "Q8_0")
        self.assertIn("f16", quant_dict)
        self.assertEqual(quant_dict["f16"], "f16")

    def test_quant_priorities_structure(self):
        self.assertIsInstance(QUANT_PRIORITIES, tuple)
        self.assertEqual(QUANT_PRIORITIES, ("q5_1", "q8_0", "q4_0", "f16"))
        for item in QUANT_PRIORITIES:
            self.assertIsInstance(item, str)


class TestGetModelSettings(unittest.TestCase):
    @patch("run_suite.requests.get")
    def test_successful_api_parsing_from_args(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "target-model-q4_k_m",
                    "status": {
                        "args": [
                            "--threads", "8",
                            "--batch-size", "1024",
                            "--ubatch-size", "256",
                            "--cache-type-k", "q8_0",
                            "--cache-type-v", "q8_0",
                            "--spec-type", "draft"
                        ],
                        "preset": ""
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        settings = get_model_settings("http://127.0.0.1:8081", "target-model-q4_k_m")

        self.assertEqual(settings["model_name"], "target-model-q4_k_m")
        self.assertEqual(settings["base_quantization"], "Q4_K_M")
        self.assertEqual(settings["threads"], 8)
        self.assertEqual(settings["batch_size"], 1024)
        self.assertEqual(settings["ubatch_size"], 256)
        self.assertEqual(settings["kv_cache_quant_k"], "q8_0")
        self.assertEqual(settings["kv_cache_quant_v"], "q8_0")
        self.assertEqual(settings["kv_cache_quant"], "q8_0")
        self.assertEqual(settings["speculative_draft_type"], "ngram")

    @patch("run_suite.requests.get")
    def test_successful_api_parsing_from_preset(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "target-model-q8_0",
                    "status": {
                        "args": [],
                        "preset": "threads = 16\nbatch-size = 2048\nubatch-size = 512\ncache-type-k = q4_0\ncache-type-v = q4_0"
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        settings = get_model_settings("http://127.0.0.1:8081", "target-model-q8_0")

        self.assertEqual(settings["model_name"], "target-model-q8_0")
        self.assertEqual(settings["base_quantization"], "Q8_0")
        self.assertEqual(settings["threads"], 16)
        self.assertEqual(settings["batch_size"], 2048)
        self.assertEqual(settings["ubatch_size"], 512)
        self.assertEqual(settings["kv_cache_quant_k"], "q4_0")
        self.assertEqual(settings["kv_cache_quant_v"], "q4_0")
        self.assertEqual(settings["kv_cache_quant"], "q4_0")
        self.assertEqual(settings["speculative_draft_type"], "None")

    @patch("run_suite.requests.get")
    def test_fallback_on_connection_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("Connection refused")

        settings = get_model_settings("http://unreachable:8081", "fallback-model-q5_k_m")

        self.assertEqual(settings["model_name"], "fallback-model-q5_k_m")
        self.assertEqual(settings["base_quantization"], "Q5_K_M")
        self.assertIsNone(settings["threads"])
        self.assertIsNone(settings["batch_size"])
        self.assertIsNone(settings["ubatch_size"])

    @patch("run_suite.requests.get")
    def test_malformed_integer_values_in_args(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test-model",
                    "status": {
                        "args": [
                            "--threads", "not_a_number",
                            "--batch-size", "invalid",
                            "--ubatch-size", "12.34"
                        ],
                        "preset": ""
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        settings = get_model_settings("http://127.0.0.1:8081", "test-model")

        self.assertIsNone(settings["threads"])
        self.assertIsNone(settings["batch_size"])
        self.assertIsNone(settings["ubatch_size"])

    @patch("run_suite.requests.get")
    def test_malformed_integer_values_in_preset(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test-model",
                    "status": {
                        "args": [],
                        "preset": "threads = bad_int\nbatch-size = invalid_val\nubatch-size = "
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        settings = get_model_settings("http://127.0.0.1:8081", "test-model")

        self.assertIsNone(settings["threads"])
        self.assertIsNone(settings["batch_size"])
        self.assertIsNone(settings["ubatch_size"])


class TestCLIArgumentParser(unittest.TestCase):
    def test_default_cli_arguments(self):
        args = parse_args([])
        self.assertEqual(args.mode, "all")
        self.assertEqual(args.endpoint, "http://127.0.0.1:8081")
        self.assertEqual(args.model, "Qwen3.6-27B")
        self.assertEqual(args.tokens, 200000)
        self.assertIsNone(args.gguf_path)
        self.assertEqual(args.corpus, "kld_corpus.txt")

    def test_custom_cli_arguments(self):
        cli_args = [
            "--mode", "throughput",
            "--endpoint", "http://localhost:9000",
            "--model", "llama-3-8b-q4_k_m",
            "--tokens", "10000",
            "--gguf-path", "/models/llama-3.gguf",
            "--corpus", "custom_corpus.txt"
        ]
        args = parse_args(cli_args)
        self.assertEqual(args.mode, "throughput")
        self.assertEqual(args.endpoint, "http://localhost:9000")
        self.assertEqual(args.model, "llama-3-8b-q4_k_m")
        self.assertEqual(args.tokens, 10000)
        self.assertEqual(args.gguf_path, "/models/llama-3.gguf")
        self.assertEqual(args.corpus, "custom_corpus.txt")

    def test_invalid_mode_choice(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_args(["--mode", "invalid_mode"])


if __name__ == "__main__":
    unittest.main()
