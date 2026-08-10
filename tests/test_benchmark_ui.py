import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock dependencies before importing benchmark_ui
sys.modules['streamlit'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()

import benchmark_ui

class TestParseBenchmarkOutput(unittest.TestCase):

    @patch('benchmark_ui.st')
    def test_parse_happy_path(self, mock_st):
        output_text = """
Some initial logging...
---> Running Turn 1 (Cold Start)...
llama_perf_context_print:        load time =    1000.00 ms
llama_perf_context_print: prompt eval time =    2000.00 ms /   100 tokens (   20.00 ms per token,    50.00 tokens per second)
llama_perf_context_print:        eval time =    5000.00 ms /    50 runs   (  100.00 ms per token,    10.00 tokens per second)
llama_perf_context_print:       total time =    7000.00 ms /   150 tokens
Prompt Tokens : 100
Completion Tokens : 50
Prompt Eval (p/s) : 25.5
TTFT: 1.2
Generation (t/s) : 10.5
Decode: 5.0
"""
        expected = [{
            "Turn": "Turn 1 (Cold Start)",
            "Prompt Tokens": 100,
            "Completion Tokens": 50,
            "Prompt Eval (p/s)": 25.5,
            "TTFT (s)": 1.2,
            "Generation (t/s)": 10.5,
            "Decode Time (s)": 5.0
        }]
        result = benchmark_ui.parse_benchmark_output(output_text)
        self.assertEqual(result, expected)
        mock_st.warning.assert_not_called()

    @patch('benchmark_ui.st')
    def test_parse_missing_metrics(self, mock_st):
        output_text = """
---> Running Turn 2 (KV Cache Hit)...
Prompt Tokens : 100
"""
        expected = [{
            "Turn": "Turn 2 (KV Cache Hit)",
            "Prompt Tokens": 100,
            "Completion Tokens": 0,
            "Prompt Eval (p/s)": 0.0,
            "TTFT (s)": 0.0,
            "Generation (t/s)": 0.0,
            "Decode Time (s)": 0.0
        }]
        result = benchmark_ui.parse_benchmark_output(output_text)
        self.assertEqual(result, expected)
        mock_st.warning.assert_not_called()

    @patch('benchmark_ui.st')
    def test_parse_all_defaults(self, mock_st):
        output_text = "---> Running Turn 3 (JSON Tool Calls)..."
        expected = [{
            "Turn": "Turn 3 (JSON Tool Calls)",
            "Prompt Tokens": 0,
            "Completion Tokens": 0,
            "Prompt Eval (p/s)": 0.0,
            "TTFT (s)": 0.0,
            "Generation (t/s)": 0.0,
            "Decode Time (s)": 0.0
        }]
        result = benchmark_ui.parse_benchmark_output(output_text)
        self.assertEqual(result, expected)
        mock_st.warning.assert_called_once()

    @patch('benchmark_ui.st')
    def test_parse_empty_output(self, mock_st):
        output_text = ""
        expected = []
        result = benchmark_ui.parse_benchmark_output(output_text)
        self.assertEqual(result, expected)
        mock_st.warning.assert_called_once()

    @patch('benchmark_ui.st')
    def test_parse_multiple_turns(self, mock_st):
        output_text = """
---> Running Turn 1 (Cold Start)...
Prompt Tokens : 100
Completion Tokens : 50
Prompt Eval (p/s) : 25.5
TTFT: 1.2
Generation (t/s) : 10.5
Decode: 5.0
---> Running Turn 2 (KV Cache Hit)...
Prompt Tokens : 200
Completion Tokens : 100
Prompt Eval (p/s) : 50.0
TTFT: 0.5
Generation (t/s) : 20.0
Decode: 2.5
"""
        expected = [
            {
                "Turn": "Turn 1 (Cold Start)",
                "Prompt Tokens": 100,
                "Completion Tokens": 50,
                "Prompt Eval (p/s)": 25.5,
                "TTFT (s)": 1.2,
                "Generation (t/s)": 10.5,
                "Decode Time (s)": 5.0
            },
            {
                "Turn": "Turn 2 (KV Cache Hit)",
                "Prompt Tokens": 200,
                "Completion Tokens": 100,
                "Prompt Eval (p/s)": 50.0,
                "TTFT (s)": 0.5,
                "Generation (t/s)": 20.0,
                "Decode Time (s)": 2.5
            }
        ]
        result = benchmark_ui.parse_benchmark_output(output_text)
        self.assertEqual(result, expected)
        mock_st.warning.assert_not_called()

if __name__ == '__main__':
    unittest.main()
