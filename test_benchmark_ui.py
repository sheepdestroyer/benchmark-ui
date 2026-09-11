import unittest
import sys
import subprocess
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


class TestListModels(unittest.TestCase):
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

    def test_invalid_endpoint(self):
        res = self.benchmark_ui.list_models("ftp://invalid.url")
        self.assertTrue(res.startswith("Error: Invalid endpoint URL:"))

    @patch("subprocess.run")
    def test_script_not_found(self, mock_run):
        mock_script = MagicMock()
        mock_script.exists.return_value = False
        with patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertTrue(res.startswith("Error: Benchmark script not found at "))
            mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_success_without_stderr(self, mock_run):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "Model-1\nModel-2"
        mock_res.stderr = ""
        mock_run.return_value = mock_res

        mock_script = MagicMock()
        mock_script.exists.return_value = True
        mock_script.__str__.return_value = "/path/to/benchmark.sh"

        with patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertEqual(res, "Model-1\nModel-2")
            mock_run.assert_called_once_with(
                ["/path/to/benchmark.sh", "--list", "http://127.0.0.1:8081"],
                cwd=str(self.benchmark_ui.WORK_DIR),
                capture_output=True,
                text=True,
                timeout=30
            )

    @patch("subprocess.run")
    def test_success_with_stderr(self, mock_run):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "Model-1"
        mock_res.stderr = "Warning: low memory"
        mock_run.return_value = mock_res

        mock_script = MagicMock()
        mock_script.exists.return_value = True
        mock_script.__str__.return_value = "/path/to/benchmark.sh"

        with patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertIn("Model-1", res)
            self.assertIn("STDERR:\nWarning: low memory", res)

    @patch("subprocess.run")
    def test_non_zero_exit_code(self, mock_run):
        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stdout = ""
        mock_res.stderr = "Connection refused"
        mock_run.return_value = mock_res

        mock_script = MagicMock()
        mock_script.exists.return_value = True
        mock_script.__str__.return_value = "/path/to/benchmark.sh"

        with patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertTrue(res.startswith("Error (exit code 1):"))
            self.assertIn("Connection refused", res)

    @patch("subprocess.run")
    def test_timeout_expired(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=30)

        mock_script = MagicMock()
        mock_script.exists.return_value = True
        mock_script.__str__.return_value = "/path/to/benchmark.sh"

        with patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertEqual(res, "Error: List models timed out")

    @patch("subprocess.run")
    def test_generic_exception(self, mock_run):
        mock_run.side_effect = RuntimeError("Unexpected process failure")

        mock_script = MagicMock()
        mock_script.exists.return_value = True
        mock_script.__str__.return_value = "/path/to/benchmark.sh"

        with patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertEqual(res, "Error listing models: Unexpected process failure")


class TestRunBenchmarkStream(unittest.TestCase):
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

    def test_validation_failure(self):
        res = list(self.benchmark_ui.run_benchmark_stream("invalid_model!", "http://127.0.0.1:8081"))
        self.assertEqual(len(res), 1)
        self.assertTrue(res[0].startswith("Error: Invalid input:"))

        res_endpoint = list(self.benchmark_ui.run_benchmark_stream("valid-model", "ftp://invalid"))
        self.assertEqual(len(res_endpoint), 1)
        self.assertTrue(res_endpoint[0].startswith("Error: Invalid input:"))

    @patch("subprocess.Popen")
    def test_script_not_found(self, mock_popen):
        mock_script = MagicMock()
        mock_script.exists.return_value = False
        with patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = list(self.benchmark_ui.run_benchmark_stream("valid-model", "http://127.0.0.1:8081"))
            self.assertEqual(len(res), 1)
            self.assertTrue(res[0].startswith("Error: Benchmark script not found at "))
            mock_popen.assert_not_called()

    @patch("os.chmod")
    @patch("os.access", return_value=False)
    def test_chmod_failure(self, mock_access, mock_chmod):
        mock_chmod.side_effect = PermissionError("Permission denied")
        mock_script = MagicMock()
        mock_script.exists.return_value = True

        with patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = list(self.benchmark_ui.run_benchmark_stream("valid-model", "http://127.0.0.1:8081"))
            self.assertEqual(len(res), 1)
            self.assertTrue(res[0].startswith("Error: Cannot make script executable:"))

    @patch("subprocess.Popen")
    def test_successful_streaming(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.readline.side_effect = ["line 1\n", "line 2\n", ""]
        mock_proc.stdout.closed = False
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        mock_script = MagicMock()
        mock_script.exists.return_value = True
        mock_script.__str__.return_value = "/path/to/benchmark.sh"

        with patch("os.access", return_value=True), patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = list(self.benchmark_ui.run_benchmark_stream("gpt-4", "http://127.0.0.1:8081"))
            self.assertEqual(res, ["line 1\n", "line 2\n"])
            mock_popen.assert_called_once_with(
                ["/path/to/benchmark.sh", "gpt-4", "http://127.0.0.1:8081"],
                cwd=str(self.benchmark_ui.WORK_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.assertIsNone(self.benchmark_ui.st.session_state.current_proc)

    @patch("subprocess.Popen")
    def test_nonzero_returncode(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout.readline.side_effect = ["processing...\n", ""]
        mock_proc.stdout.closed = False
        mock_proc.poll.return_value = 1
        mock_popen.return_value = mock_proc

        mock_script = MagicMock()
        mock_script.exists.return_value = True
        mock_script.__str__.return_value = "/path/to/benchmark.sh"

        with patch("os.access", return_value=True), patch.object(self.benchmark_ui, "BENCH_SCRIPT", mock_script):
            res = list(self.benchmark_ui.run_benchmark_stream("gpt-4", "http://127.0.0.1:8081"))
            self.assertEqual(res, ["processing...\n", "\n[EXIT CODE: 1]"])


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

    def test_parse_happy_path(self):
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
        with patch.object(self.benchmark_ui, "st") as mock_st:
            expected = [make_turn_dict("Turn 1 (Cold Start)", prompt_tokens=100, completion_tokens=50, prompt_eval=25.5, ttft=1.2, generation=10.5, decode=5.0)]
            result = self.benchmark_ui.parse_benchmark_output(output_text)
            self.assertEqual(result, expected)
            mock_st.warning.assert_not_called()

    def test_parse_missing_metrics(self):
        output_text = """
---> Running Turn 2 (KV Cache Hit)...
Prompt Tokens : 100
"""
        with patch.object(self.benchmark_ui, "st") as mock_st:
            expected = [make_turn_dict("Turn 2 (KV Cache Hit)", prompt_tokens=100)]
            result = self.benchmark_ui.parse_benchmark_output(output_text)
            self.assertEqual(result, expected)
            mock_st.warning.assert_not_called()

    def test_parse_all_defaults(self):
        output_text = "---> Running Turn 3 (JSON Tool Calls)..."
        with patch.object(self.benchmark_ui, "st") as mock_st:
            expected = [make_turn_dict("Turn 3 (JSON Tool Calls)")]
            result = self.benchmark_ui.parse_benchmark_output(output_text)
            self.assertEqual(result, expected)
            mock_st.warning.assert_called_once()

    def test_parse_empty_output(self):
        output_text = ""
        with patch.object(self.benchmark_ui, "st") as mock_st:
            expected = []
            result = self.benchmark_ui.parse_benchmark_output(output_text)
            self.assertEqual(result, expected)
            mock_st.warning.assert_called_once()

    def test_parse_multiple_turns(self):
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
        with patch.object(self.benchmark_ui, "st") as mock_st:
            expected = [
                make_turn_dict("Turn 1 (Cold Start)", prompt_tokens=100, completion_tokens=50, prompt_eval=25.5, ttft=1.2, generation=10.5, decode=5.0),
                make_turn_dict("Turn 2 (KV Cache Hit)", prompt_tokens=200, completion_tokens=100, prompt_eval=50.0, ttft=0.5, generation=20.0, decode=2.5),
            ]
            result = self.benchmark_ui.parse_benchmark_output(output_text)
            self.assertEqual(result, expected)
            mock_st.warning.assert_not_called()
