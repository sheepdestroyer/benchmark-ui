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
                "plotly": MagicMock(),
                "plotly.express": MagicMock(),
            }
        )
        cls.modules_patcher.start()
        cls.benchmark_ui = importlib.import_module("benchmark_ui")

    @classmethod
    def tearDownClass(cls):
        if "benchmark_ui" in sys.modules:
            del sys.modules["benchmark_ui"]
        cls.modules_patcher.stop()

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
