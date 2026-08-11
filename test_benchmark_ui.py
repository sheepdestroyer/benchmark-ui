import unittest
import sys
import importlib
from unittest.mock import MagicMock, patch

class TestBenchmarkUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules_patcher = patch.dict(
            sys.modules,
            {
                "streamlit": MagicMock(),
                "pandas": MagicMock(),
                "plotly.express": MagicMock(),
                "plotly": MagicMock(),
            }
        )
        cls.modules_patcher.start()
        cls.benchmark_ui = importlib.import_module("benchmark_ui")

    @classmethod
    def tearDownClass(cls):
        if "benchmark_ui" in sys.modules:
            del sys.modules["benchmark_ui"]
        cls.modules_patcher.stop()

    def test_validate_endpoint_url_valid(self):
        self.assertEqual(self.benchmark_ui.validate_endpoint_url("http://localhost"), "http://localhost")
        self.assertEqual(self.benchmark_ui.validate_endpoint_url("http://127.0.0.1"), "http://127.0.0.1")
        self.assertEqual(self.benchmark_ui.validate_endpoint_url("https://api.openai.com"), "https://api.openai.com")
        self.assertEqual(self.benchmark_ui.validate_endpoint_url("http://example.com:8080"), "http://example.com:8080")
        self.assertEqual(self.benchmark_ui.validate_endpoint_url("http://8.8.8.8"), "http://8.8.8.8")

    def test_validate_endpoint_url_invalid(self):
        with self.assertRaisesRegex(ValueError, "Endpoint URL cannot be empty"):
            self.benchmark_ui.validate_endpoint_url("")

        with self.assertRaisesRegex(ValueError, "Invalid URL scheme"):
            self.benchmark_ui.validate_endpoint_url("ftp://example.com")

        with self.assertRaisesRegex(ValueError, "Invalid URL: missing hostname"):
            self.benchmark_ui.validate_endpoint_url("http://")

        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            self.benchmark_ui.validate_endpoint_url("http://192.168.1.1")
        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            self.benchmark_ui.validate_endpoint_url("http://10.0.0.1")
        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            self.benchmark_ui.validate_endpoint_url("http://172.16.0.1")

    def test_validate_model_name_valid(self):
        self.assertEqual(self.benchmark_ui.validate_model_name("llama-3-8b"), "llama-3-8b")
        self.assertEqual(self.benchmark_ui.validate_model_name("gpt-4"), "gpt-4")
        self.assertEqual(self.benchmark_ui.validate_model_name("claude-3-opus-20240229"), "claude-3-opus-20240229")
        self.assertEqual(self.benchmark_ui.validate_model_name("Qwen/Qwen1.5-72B-Chat"), "Qwen/Qwen1.5-72B-Chat")
        self.assertEqual(self.benchmark_ui.validate_model_name("meta-llama/Llama-2-7b-chat-hf"), "meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(self.benchmark_ui.validate_model_name("my_model:v1"), "my_model:v1")
        self.assertEqual(self.benchmark_ui.validate_model_name("model.name.with.dots"), "model.name.with.dots")

    def test_validate_model_name_invalid(self):
        with self.assertRaisesRegex(ValueError, "Invalid model name"):
            self.benchmark_ui.validate_model_name("")

        with self.assertRaisesRegex(ValueError, "Invalid model name"):
            self.benchmark_ui.validate_model_name("model with spaces")

        with self.assertRaisesRegex(ValueError, "Invalid model name"):
            self.benchmark_ui.validate_model_name("model_with_!@#")

        with self.assertRaisesRegex(ValueError, "Invalid model name"):
            self.benchmark_ui.validate_model_name("model_with_;")
