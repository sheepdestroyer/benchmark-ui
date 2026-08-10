import unittest
import sys
from unittest.mock import MagicMock

# Mock Streamlit, Pandas, and Plotly to prevent import errors and st.set_page_config issues
sys.modules['streamlit'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()

# Now we can safely import benchmark_ui
import benchmark_ui


class TestBenchmarkUI_Functions(unittest.TestCase):

    def test_validate_endpoint_url_valid(self):
        valid_urls = [
            "http://localhost",
            "http://127.0.0.1",
            "http://127.0.0.1:8080",
            "https://example.com",
            "https://api.openai.com/v1"
        ]
        for url in valid_urls:
            self.assertEqual(benchmark_ui.validate_endpoint_url(url), url)

    def test_validate_endpoint_url_invalid(self):
        # Empty url
        with self.assertRaisesRegex(ValueError, "Endpoint URL cannot be empty"):
            benchmark_ui.validate_endpoint_url("")

        # Invalid scheme
        with self.assertRaisesRegex(ValueError, "Invalid URL scheme"):
            benchmark_ui.validate_endpoint_url("ftp://example.com")

        # Missing hostname
        with self.assertRaisesRegex(ValueError, "Invalid URL: missing hostname"):
            benchmark_ui.validate_endpoint_url("http://")

        # Forbidden IPs (private)
        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            benchmark_ui.validate_endpoint_url("http://192.168.1.1")

        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            benchmark_ui.validate_endpoint_url("http://10.0.0.1")

    def test_validate_model_name_valid(self):
        valid_models = [
            "Qwen3.6-35B-A3B",
            "meta-llama/Llama-3-8b-instruct",
            "my_model:latest",
            "model.name",
            "model-name"
        ]
        for model in valid_models:
            self.assertEqual(benchmark_ui.validate_model_name(model), model)

    def test_validate_model_name_invalid(self):
        invalid_models = [
            "",
            "my model",
            "model$name",
            "model<name>"
        ]
        for model in invalid_models:
            with self.assertRaisesRegex(ValueError, "Invalid model name"):
                benchmark_ui.validate_model_name(model)

    def test_parse_benchmark_output(self):
        sample_output = """
Some initial text that should be ignored.
---> Running Turn 1 (Cold Start)...
Prompt Tokens   : 150
Completion Tokens : 50
Prompt Eval (p/s) : 120.5
TTFT: 0.85
Generation (t/s) : 45.2
Decode: 1.1

---> Running Turn 2 (KV Cache Hit)...
Prompt Tokens   : 150
Completion Tokens : 100
Prompt Eval (p/s) : 2500.0
TTFT: 0.15
Generation (t/s) : 48.5
Decode: 2.05
"""
        parsed = benchmark_ui.parse_benchmark_output(sample_output)

        self.assertEqual(len(parsed), 2)

        # Turn 1 Checks
        self.assertEqual(parsed[0]["Turn"], "Turn 1 (Cold Start)")
        self.assertEqual(parsed[0]["Prompt Tokens"], 150)
        self.assertEqual(parsed[0]["Completion Tokens"], 50)
        self.assertEqual(parsed[0]["Prompt Eval (p/s)"], 120.5)
        self.assertEqual(parsed[0]["TTFT (s)"], 0.85)
        self.assertEqual(parsed[0]["Generation (t/s)"], 45.2)
        self.assertEqual(parsed[0]["Decode Time (s)"], 1.1)

        # Turn 2 Checks
        self.assertEqual(parsed[1]["Turn"], "Turn 2 (KV Cache Hit)")
        self.assertEqual(parsed[1]["Prompt Tokens"], 150)
        self.assertEqual(parsed[1]["Completion Tokens"], 100)
        self.assertEqual(parsed[1]["Prompt Eval (p/s)"], 2500.0)
        self.assertEqual(parsed[1]["TTFT (s)"], 0.15)
        self.assertEqual(parsed[1]["Generation (t/s)"], 48.5)
        self.assertEqual(parsed[1]["Decode Time (s)"], 2.05)


if __name__ == '__main__':
    unittest.main()
