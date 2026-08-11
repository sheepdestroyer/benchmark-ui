import sys
import unittest
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
                "plotly": MagicMock(),
                "plotly.express": MagicMock(),
                "plotly.graph_objects": MagicMock(),
            },
        )
        cls.modules_patcher.start()
        cls.benchmark_ui = importlib.import_module("benchmark_ui")

    @classmethod
    def tearDownClass(cls):
        if "benchmark_ui" in sys.modules:
            del sys.modules["benchmark_ui"]
        cls.modules_patcher.stop()

    def test_validate_endpoint_url_valid(self):
        valid_urls = [
            "http://localhost",
            "http://127.0.0.1",
            "http://127.0.0.1:8080",
            "https://example.com",
            "https://api.openai.com/v1"
        ]
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertEqual(self.benchmark_ui.validate_endpoint_url(url), url)

    def test_validate_endpoint_url_invalid(self):
        cases = [
            ("", "Endpoint URL cannot be empty"),
            ("ftp://example.com", "Invalid URL scheme"),
            ("http://", "Invalid URL: missing hostname"),
            ("http://192.168.1.1", "Forbidden IP address range"),
            ("http://10.0.0.1", "Forbidden IP address range")
        ]
        for url, expected_error in cases:
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, expected_error):
                    self.benchmark_ui.validate_endpoint_url(url)

    def test_validate_model_name_valid(self):
        valid_models = [
            "Qwen3.6-35B-A3B",
            "meta-llama/Llama-3-8b-instruct",
            "my_model:latest",
            "model.name",
            "model-name"
        ]
        for model in valid_models:
            with self.subTest(model=model):
                self.assertEqual(self.benchmark_ui.validate_model_name(model), model)

    def test_validate_model_name_invalid(self):
        invalid_models = [
            "",
            "my model",
            "model$name",
            "model<name>"
        ]
        for model in invalid_models:
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "Invalid model name"):
                    self.benchmark_ui.validate_model_name(model)

    def test_parse_benchmark_output_happy_path(self):
        output = """
---> Running Turn 1 (test)...
Prompt Tokens   : 100
Completion Tokens : 50
Prompt Eval (p/s) : 20.5
TTFT: 0.5
Generation (t/s) : 10.0
Decode: 5.0
---> Running Turn 2 (test2)...
Prompt Tokens   : 200
Completion Tokens : 100
Prompt Eval (p/s) : 15.5
TTFT: 1.0
Generation (t/s) : 8.0
Decode: 10.0
"""
        turns = self.benchmark_ui.parse_benchmark_output(output)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["Turn"], "Turn 1 (test)")
        self.assertEqual(turns[0]["Prompt Tokens"], 100)
        self.assertEqual(turns[0]["Completion Tokens"], 50)
        self.assertEqual(turns[0]["Prompt Eval (p/s)"], 20.5)
        self.assertEqual(turns[0]["TTFT (s)"], 0.5)
        self.assertEqual(turns[0]["Generation (t/s)"], 10.0)
        self.assertEqual(turns[0]["Decode Time (s)"], 5.0)

        self.assertEqual(turns[1]["Turn"], "Turn 2 (test2)")
        self.assertEqual(turns[1]["Prompt Tokens"], 200)
        self.assertEqual(turns[1]["Completion Tokens"], 100)
        self.assertEqual(turns[1]["Prompt Eval (p/s)"], 15.5)
        self.assertEqual(turns[1]["TTFT (s)"], 1.0)
        self.assertEqual(turns[1]["Generation (t/s)"], 8.0)
        self.assertEqual(turns[1]["Decode Time (s)"], 10.0)

    def test_parse_benchmark_output_missing_matches(self):
        output = """
---> Running Turn 1 (test)...
Prompt Tokens   : 100
Prompt Eval (p/s) : 20.5
"""
        turns = self.benchmark_ui.parse_benchmark_output(output)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["Turn"], "Turn 1 (test)")
        self.assertEqual(turns[0]["Prompt Tokens"], 100)
        self.assertEqual(turns[0]["Completion Tokens"], 0)
        self.assertEqual(turns[0]["Prompt Eval (p/s)"], 20.5)
        self.assertEqual(turns[0]["TTFT (s)"], 0.0)
        self.assertEqual(turns[0]["Generation (t/s)"], 0.0)
        self.assertEqual(turns[0]["Decode Time (s)"], 0.0)

    def test_parse_benchmark_output_no_turns(self):
        output = "Just some random output without turn indicators."
        turns = self.benchmark_ui.parse_benchmark_output(output)
        self.assertEqual(len(turns), 0)

if __name__ == "__main__":
    unittest.main()
