import unittest
import sys
from unittest.mock import MagicMock

# Mock heavy/problematic dependencies before importing the module
sys.modules['streamlit'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly'] = MagicMock()

# Now we can safely import the functions
from benchmark_ui import validate_endpoint_url, validate_model_name

class TestBenchmarkUI(unittest.TestCase):
    def test_validate_endpoint_url_valid(self):
        self.assertEqual(validate_endpoint_url("http://localhost"), "http://localhost")
        self.assertEqual(validate_endpoint_url("http://127.0.0.1"), "http://127.0.0.1")
        self.assertEqual(validate_endpoint_url("https://api.openai.com"), "https://api.openai.com")
        self.assertEqual(validate_endpoint_url("http://example.com:8080"), "http://example.com:8080")
        # Global IP
        self.assertEqual(validate_endpoint_url("http://8.8.8.8"), "http://8.8.8.8")

    def test_validate_endpoint_url_invalid(self):
        with self.assertRaisesRegex(ValueError, "Endpoint URL cannot be empty"):
            validate_endpoint_url("")

        with self.assertRaisesRegex(ValueError, "Invalid URL scheme"):
            validate_endpoint_url("ftp://example.com")

        with self.assertRaisesRegex(ValueError, "Invalid URL: missing hostname"):
            validate_endpoint_url("http://")

        # Private IPs (forbidden)
        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            validate_endpoint_url("http://192.168.1.1")
        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            validate_endpoint_url("http://10.0.0.1")
        with self.assertRaisesRegex(ValueError, "Forbidden IP address range"):
            validate_endpoint_url("http://172.16.0.1")

    def test_validate_model_name_valid(self):
        self.assertEqual(validate_model_name("llama-3-8b"), "llama-3-8b")
        self.assertEqual(validate_model_name("gpt-4"), "gpt-4")
        self.assertEqual(validate_model_name("claude-3-opus-20240229"), "claude-3-opus-20240229")
        self.assertEqual(validate_model_name("Qwen/Qwen1.5-72B-Chat"), "Qwen/Qwen1.5-72B-Chat")
        self.assertEqual(validate_model_name("meta-llama/Llama-2-7b-chat-hf"), "meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(validate_model_name("my_model:v1"), "my_model:v1")
        self.assertEqual(validate_model_name("model.name.with.dots"), "model.name.with.dots")

    def test_validate_model_name_invalid(self):
        with self.assertRaises(ValueError):
            validate_model_name("")

        with self.assertRaises(ValueError):
            validate_model_name("model with spaces")

        with self.assertRaises(ValueError):
            validate_model_name("model_with_!@#")

        with self.assertRaises(ValueError):
            validate_model_name("model_with_;")

if __name__ == '__main__':
    unittest.main()
