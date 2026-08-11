import unittest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure root dir is in sys.path
root_dir = str(Path(__file__).parent.parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def make_turn_dict(turn_name, prompt_tokens=0, completion_tokens=0, prompt_eval=0.0, ttft=0.0, generation=0.0, decode=0.0):
    return {
        "Turn": turn_name,
        "Prompt Tokens": prompt_tokens,
        "Completion Tokens": completion_tokens,
        "Prompt Eval (p/s)": prompt_eval,
        "TTFT (s)": ttft,
        "Generation (t/s)": generation,
        "Decode Time (s)": decode,
    }

class TestParseBenchmarkOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.modules_patcher = patch.dict(sys.modules, {
            'streamlit': MagicMock(),
            'pandas': MagicMock(),
            'plotly': MagicMock(),
            'plotly.express': MagicMock(),
        })
        cls.modules_patcher.start()
        global benchmark_ui
        import benchmark_ui

    @classmethod
    def tearDownClass(cls):
        cls.modules_patcher.stop()

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
        expected = [make_turn_dict("Turn 1 (Cold Start)", prompt_tokens=100, completion_tokens=50, prompt_eval=25.5, ttft=1.2, generation=10.5, decode=5.0)]
        result = benchmark_ui.parse_benchmark_output(output_text)
        self.assertEqual(result, expected)
        mock_st.warning.assert_not_called()

    @patch('benchmark_ui.st')
    def test_parse_missing_metrics(self, mock_st):
        output_text = """
---> Running Turn 2 (KV Cache Hit)...
Prompt Tokens : 100
"""
        expected = [make_turn_dict("Turn 2 (KV Cache Hit)", prompt_tokens=100)]
        result = benchmark_ui.parse_benchmark_output(output_text)
        self.assertEqual(result, expected)
        mock_st.warning.assert_not_called()

    @patch('benchmark_ui.st')
    def test_parse_all_defaults(self, mock_st):
        output_text = "---> Running Turn 3 (JSON Tool Calls)..."
        expected = [make_turn_dict("Turn 3 (JSON Tool Calls)")]
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
            make_turn_dict("Turn 1 (Cold Start)", prompt_tokens=100, completion_tokens=50, prompt_eval=25.5, ttft=1.2, generation=10.5, decode=5.0),
            make_turn_dict("Turn 2 (KV Cache Hit)", prompt_tokens=200, completion_tokens=100, prompt_eval=50.0, ttft=0.5, generation=20.0, decode=2.5),
        ]
        result = benchmark_ui.parse_benchmark_output(output_text)
        self.assertEqual(result, expected)
        mock_st.warning.assert_not_called()

if __name__ == '__main__':
    unittest.main()
