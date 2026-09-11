import unittest
import sys
import os
import importlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import requests

class MockSessionState(dict):
    """Dictionary subclass supporting attribute-style access like Streamlit's session_state."""
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value


class BaseBenchmarkUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_st = MagicMock()
        cls.mock_pd = MagicMock()
        cls.mock_px = MagicMock()
        cls.modules_patcher = patch.dict(
            sys.modules,
            {
                "streamlit": cls.mock_st,
                "pandas": cls.mock_pd,
                "plotly.express": cls.mock_px,
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

    def setUp(self):
        self.session_state = MockSessionState({
            'list_models': False,
            'run_benchmark': False,
            'benchmark_history': [],
            'benchmark_running': False,
            'current_proc': None,
            'benchmark_start': 0.0,
        })
        self.benchmark_ui.st.session_state = self.session_state


class TestBenchmarkUI(BaseBenchmarkUITest):
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


class TestParseBenchmarkOutput(BaseBenchmarkUITest):
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


class TestCleanupCurrentProc(BaseBenchmarkUITest):
    def test_cleanup_proc_none(self):
        self.session_state.current_proc = None
        self.benchmark_ui.cleanup_current_proc()
        self.assertIsNone(self.session_state.current_proc)

    def test_cleanup_proc_active_terminated(self):
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.closed = False
        mock_proc.poll.return_value = None
        self.session_state.current_proc = mock_proc

        self.benchmark_ui.cleanup_current_proc()

        mock_proc.stdout.close.assert_called_once()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=2)
        self.assertIsNone(self.session_state.current_proc)

    def test_cleanup_proc_timeout_killed(self):
        mock_proc = MagicMock()
        mock_proc.stdout = None
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="test", timeout=2), None]
        self.session_state.current_proc = mock_proc

        self.benchmark_ui.cleanup_current_proc()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        self.assertEqual(mock_proc.wait.call_count, 2)
        mock_proc.wait.assert_has_calls([call(timeout=2), call(timeout=1)])
        self.assertIsNone(self.session_state.current_proc)

    def test_cleanup_proc_already_exited(self):
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.closed = True
        mock_proc.poll.return_value = 0
        self.session_state.current_proc = mock_proc

        self.benchmark_ui.cleanup_current_proc()

        mock_proc.stdout.close.assert_not_called()
        mock_proc.terminate.assert_not_called()
        self.assertIsNone(self.session_state.current_proc)

    def test_cleanup_proc_stdout_close_exception(self):
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.closed = False
        mock_proc.stdout.close.side_effect = OSError("close error")
        mock_proc.poll.return_value = 0
        self.session_state.current_proc = mock_proc

        self.benchmark_ui.cleanup_current_proc()

        mock_proc.stdout.close.assert_called_once()
        self.assertIsNone(self.session_state.current_proc)


class TestListModels(BaseBenchmarkUITest):
    def test_list_models_invalid_endpoint(self):
        res = self.benchmark_ui.list_models("")
        self.assertTrue(res.startswith("Error: Invalid endpoint URL:"))

        res_forbidden = self.benchmark_ui.list_models("http://192.168.1.1")
        self.assertTrue(res_forbidden.startswith("Error: Invalid endpoint URL:"))

    def test_list_models_script_not_found(self):
        with patch("pathlib.Path.exists", return_value=False):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertTrue(res.startswith("Error: Benchmark script not found at"))

    def test_list_models_success(self):
        mock_proc_res = subprocess.CompletedProcess(
            args=["benchmark.sh", "--list", "http://127.0.0.1:8081"],
            returncode=0,
            stdout="Qwen3.6-27B\nQwen3.6-35B-A3B\n",
            stderr=""
        )
        with patch("pathlib.Path.exists", return_value=True), \
             patch("subprocess.run", return_value=mock_proc_res) as mock_run:
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertEqual(res, "Qwen3.6-27B\nQwen3.6-35B-A3B\n")
            mock_run.assert_called_once_with(
                [str(self.benchmark_ui.BENCH_SCRIPT), "--list", "http://127.0.0.1:8081"],
                cwd=str(self.benchmark_ui.WORK_DIR),
                capture_output=True,
                text=True,
                timeout=30
            )

    def test_list_models_empty_data(self):
        mock_proc_res = subprocess.CompletedProcess(
            args=["benchmark.sh", "--list", "http://127.0.0.1:8081"],
            returncode=0,
            stdout="",
            stderr=""
        )
        with patch("pathlib.Path.exists", return_value=True), \
             patch("subprocess.run", return_value=mock_proc_res):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertEqual(res, "")

    def test_list_models_with_stderr(self):
        mock_proc_res = subprocess.CompletedProcess(
            args=["benchmark.sh", "--list", "http://127.0.0.1:8081"],
            returncode=0,
            stdout="Qwen3.6-27B\n",
            stderr="Warning: slow endpoint response"
        )
        with patch("pathlib.Path.exists", return_value=True), \
             patch("subprocess.run", return_value=mock_proc_res):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertIn("Qwen3.6-27B\n", res)
            self.assertIn("\n\nSTDERR:\nWarning: slow endpoint response", res)

    def test_list_models_non_zero_exit_status(self):
        mock_proc_res = subprocess.CompletedProcess(
            args=["benchmark.sh", "--list", "http://127.0.0.1:8081"],
            returncode=1,
            stdout="Error: Could not fetch models.",
            stderr="curl: (7) Failed to connect to 127.0.0.1:8081"
        )
        with patch("pathlib.Path.exists", return_value=True), \
             patch("subprocess.run", return_value=mock_proc_res):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertTrue(res.startswith("Error (exit code 1):"))
            self.assertIn("Error: Could not fetch models.", res)
            self.assertIn("curl: (7) Failed to connect to 127.0.0.1:8081", res)

    def test_list_models_timeout(self):
        with patch("pathlib.Path.exists", return_value=True), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="benchmark.sh", timeout=30)):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertEqual(res, "Error: List models timed out")

    def test_list_models_request_exception(self):
        with patch("pathlib.Path.exists", return_value=True), \
             patch("subprocess.run", side_effect=requests.exceptions.RequestException("Network unreachable")):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertEqual(res, "Error listing models: Network unreachable")

    def test_list_models_generic_os_exception(self):
        with patch("pathlib.Path.exists", return_value=True), \
             patch("subprocess.run", side_effect=OSError("Exec format error")):
            res = self.benchmark_ui.list_models("http://127.0.0.1:8081")
            self.assertEqual(res, "Error listing models: Exec format error")


class TestRunBenchmarkStream(BaseBenchmarkUITest):
    def test_run_benchmark_stream_invalid_endpoint_url(self):
        lines = list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "ftp://example.com"))
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("Error: Invalid input:"))

        lines_empty = list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", ""))
        self.assertEqual(len(lines_empty), 1)
        self.assertTrue(lines_empty[0].startswith("Error: Invalid input:"))

    def test_run_benchmark_stream_invalid_model_name(self):
        lines = list(self.benchmark_ui.run_benchmark_stream("model with spaces", "http://127.0.0.1:8081"))
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("Error: Invalid input:"))

        lines_empty = list(self.benchmark_ui.run_benchmark_stream("", "http://127.0.0.1:8081"))
        self.assertEqual(len(lines_empty), 1)
        self.assertTrue(lines_empty[0].startswith("Error: Invalid input:"))

    def test_run_benchmark_stream_script_not_found(self):
        with patch("pathlib.Path.exists", return_value=False):
            lines = list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "http://127.0.0.1:8081"))
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].startswith("Error: Benchmark script not found at"))

    def test_run_benchmark_stream_chmod_handling(self):
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = ["Starting benchmark...\n", ""]
        mock_proc.stdout.closed = False
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0

        with patch("pathlib.Path.exists", return_value=True), \
             patch("os.access", return_value=False), \
             patch("os.chmod") as mock_chmod, \
             patch("subprocess.Popen", return_value=mock_proc):
            lines = list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "http://127.0.0.1:8081"))
            mock_chmod.assert_called_once_with(self.benchmark_ui.BENCH_SCRIPT, 0o755)
            self.assertEqual(lines, ["Starting benchmark...\n"])

    def test_run_benchmark_stream_chmod_failure(self):
        with patch("pathlib.Path.exists", return_value=True), \
             patch("os.access", return_value=False), \
             patch("os.chmod", side_effect=PermissionError("chmod not permitted")):
            lines = list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "http://127.0.0.1:8081"))
            self.assertEqual(lines, ["Error: Cannot make script executable: chmod not permitted"])

    def test_run_benchmark_stream_success(self):
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            "---> Running Turn 1 (Cold Start)...\n",
            "Prompt Tokens : 100\n",
            "Completion Tokens : 50\n",
            ""
        ]
        mock_proc.stdout.closed = False
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0

        with patch("pathlib.Path.exists", return_value=True), \
             patch("os.access", return_value=True), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            gen = self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "http://127.0.0.1:8081")
            first_line = next(gen)
            self.assertEqual(first_line, "---> Running Turn 1 (Cold Start)...\n")
            # Verify current_proc was populated in session state during execution
            self.assertIs(self.session_state.current_proc, mock_proc)

            remaining_lines = list(gen)
            self.assertEqual(remaining_lines, ["Prompt Tokens : 100\n", "Completion Tokens : 50\n"])
            # Verify finally block cleaned up current_proc
            self.assertIsNone(self.session_state.current_proc)
            mock_proc.stdout.close.assert_called_once()
            mock_popen.assert_called_once_with(
                [str(self.benchmark_ui.BENCH_SCRIPT), "Qwen3.6-27B", "http://127.0.0.1:8081"],
                cwd=str(self.benchmark_ui.WORK_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

    def test_run_benchmark_stream_failure_nonzero_returncode(self):
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = ["Failed to connect to endpoint\n", ""]
        mock_proc.stdout.closed = False
        mock_proc.wait.return_value = 1
        mock_proc.returncode = 1
        mock_proc.poll.return_value = 1

        with patch("pathlib.Path.exists", return_value=True), \
             patch("os.access", return_value=True), \
             patch("subprocess.Popen", return_value=mock_proc):
            lines = list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "http://127.0.0.1:8081"))
            self.assertEqual(lines, ["Failed to connect to endpoint\n", "\n[EXIT CODE: 1]"])
            self.assertIsNone(self.session_state.current_proc)

    def test_run_benchmark_stream_exception_during_execution(self):
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = RuntimeError("Broken pipe")
        mock_proc.stdout.closed = False
        mock_proc.poll.return_value = None

        with patch("pathlib.Path.exists", return_value=True), \
             patch("os.access", return_value=True), \
             patch("subprocess.Popen", return_value=mock_proc):
            with self.assertRaises(RuntimeError):
                list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "http://127.0.0.1:8081"))

            mock_proc.terminate.assert_called_once()
            mock_proc.wait.assert_called_once_with(timeout=2)
            self.assertIsNone(self.session_state.current_proc)

    def test_run_benchmark_stream_timeout_during_terminate(self):
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = KeyboardInterrupt("User abort")
        mock_proc.stdout.closed = False
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="terminate", timeout=2), None]

        with patch("pathlib.Path.exists", return_value=True), \
             patch("os.access", return_value=True), \
             patch("subprocess.Popen", return_value=mock_proc):
            with self.assertRaises(KeyboardInterrupt):
                list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "http://127.0.0.1:8081"))

            mock_proc.terminate.assert_called_once()
            mock_proc.kill.assert_called_once()
            self.assertIsNone(self.session_state.current_proc)

    def test_run_benchmark_stream_no_stdout(self):
        mock_proc = MagicMock()
        mock_proc.stdout = None
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0

        with patch("pathlib.Path.exists", return_value=True), \
             patch("os.access", return_value=True), \
             patch("subprocess.Popen", return_value=mock_proc):
            lines = list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "http://127.0.0.1:8081"))
            self.assertEqual(lines, [])
            self.assertIsNone(self.session_state.current_proc)

    def test_run_benchmark_stream_stdout_close_exception(self):
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = ["line\n", ""]
        mock_proc.stdout.closed = False
        mock_proc.stdout.close.side_effect = OSError("close error")
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0

        with patch("pathlib.Path.exists", return_value=True), \
             patch("os.access", return_value=True), \
             patch("subprocess.Popen", return_value=mock_proc):
            lines = list(self.benchmark_ui.run_benchmark_stream("Qwen3.6-27B", "http://127.0.0.1:8081"))
            self.assertEqual(lines, ["line\n"])
            mock_proc.stdout.close.assert_called_once()


class TestMainUI(BaseBenchmarkUITest):
    def setUp(self):
        super().setUp()
        self.benchmark_ui.st.reset_mock()
        self.mock_pd.reset_mock()
        self.mock_px.reset_mock()
        self.mock_pd.DataFrame.side_effect = None
        self.mock_tab1 = MagicMock()
        self.mock_tab2 = MagicMock()
        self.benchmark_ui.st.tabs.return_value = (self.mock_tab1, self.mock_tab2)
        self.benchmark_ui.st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
        self.benchmark_ui.st.sidebar = MagicMock()
        self.benchmark_ui.st.button.side_effect = None
        self.benchmark_ui.st.button.return_value = False
        self.benchmark_ui.st.text_input.side_effect = lambda label, value="", **kwargs: value

    def test_main_default_view(self):
        self.session_state.list_models = False
        self.session_state.run_benchmark = False

        self.benchmark_ui.main()

        self.benchmark_ui.st.title.assert_called_with("🚀 LLM Benchmark UI")
        self.benchmark_ui.st.info.assert_any_call("👈 Configure your model and endpoint in the sidebar, then choose an action.")

    def test_main_list_models_view(self):
        self.session_state.list_models = True
        self.session_state.run_benchmark = False

        with patch.object(self.benchmark_ui, "list_models", return_value="Qwen3.6-27B\nQwen3.6-35B\n") as mock_lm:
            self.benchmark_ui.main()
            mock_lm.assert_called_once_with("http://127.0.0.1:8081")
            self.benchmark_ui.st.subheader.assert_any_call("Available Models")
            self.benchmark_ui.st.code.assert_any_call("Qwen3.6-27B\nQwen3.6-35B\n", language="bash")

    def test_main_run_benchmark_streaming_and_completed(self):
        self.session_state.run_benchmark = True
        self.session_state.benchmark_running = True
        self.session_state.benchmark_history = [MagicMock() for _ in range(52)]
        stream_lines = [
            "---> Running Turn 1 (Cold Start)...\n",
            "Prompt Tokens : 100\n",
            "Completion Tokens : 50\n",
            "Prompt Eval (p/s) : 25.0\n",
            "TTFT: 1.0\n",
            "Generation (t/s) : 10.0\n",
            "Decode: 5.0\n"
        ]

        with patch.object(self.benchmark_ui, "run_benchmark_stream", return_value=stream_lines), \
             patch("time.monotonic", side_effect=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]):
            self.benchmark_ui.main()

            self.assertFalse(self.session_state.benchmark_running)
            self.assertEqual(len(self.session_state.benchmark_history), 50)
            self.assertIn("Benchmark results saved to history!", self.benchmark_ui.st.success.call_args[0][0])

    def test_main_run_benchmark_already_completed(self):
        self.session_state.run_benchmark = True
        self.session_state.benchmark_running = False
        self.session_state.last_benchmark_output = "Done benchmark output"

        self.benchmark_ui.main()

        self.benchmark_ui.st.download_button.assert_called_once()
        call_kwargs = self.benchmark_ui.st.download_button.call_args[1]
        self.assertEqual(call_kwargs["data"], "Done benchmark output")

    def test_main_sidebar_button_list_models_click(self):
        def mock_button(label, **kwargs):
            return label == "📋 List Models"

        self.benchmark_ui.st.button.side_effect = mock_button
        self.benchmark_ui.main()

        self.assertTrue(self.session_state.list_models)
        self.assertFalse(self.session_state.run_benchmark)

    def test_main_sidebar_button_run_benchmark_click(self):
        def mock_button(label, **kwargs):
            return label == "▶️ Run Benchmark"

        self.benchmark_ui.st.button.side_effect = mock_button
        with patch.object(self.benchmark_ui, "run_benchmark_stream", return_value=["line\n"]):
            self.benchmark_ui.main()

        self.assertFalse(self.session_state.list_models)
        self.assertTrue(self.session_state.run_benchmark)
        self.assertFalse(self.session_state.benchmark_running)
        self.assertEqual(len(self.session_state.benchmark_history), 1)

    def test_main_visualizations_tab_empty(self):
        self.session_state.benchmark_history = []
        self.benchmark_ui.main()
        self.benchmark_ui.st.info.assert_any_call("No benchmark history yet. Run a benchmark first to see visualizations.")

    def test_main_visualizations_tab_no_runs_selected(self):
        self.session_state.benchmark_history = [
            {
                "id": "run_1",
                "model": "Qwen3.6-27B",
                "endpoint": "http://127.0.0.1:8081",
                "timestamp": "2026-09-11 12:00:00",
                "duration": "10.00s",
                "raw_output": "test",
                "turns": [make_turn_dict("Turn 1 (Cold Start)")]
            }
        ]
        self.benchmark_ui.st.multiselect.return_value = []
        self.benchmark_ui.main()
        self.benchmark_ui.st.warning.assert_any_call("Please select at least one benchmark run.")

    def test_main_visualizations_tab_with_history(self):
        self.session_state.benchmark_history = [
            {
                "id": "run_1",
                "model": "Qwen3.6-27B",
                "endpoint": "http://127.0.0.1:8081",
                "timestamp": "2026-09-11 12:00:00",
                "duration": "10.00s",
                "raw_output": "test",
                "turns": [make_turn_dict("Turn 1 (Cold Start)", prompt_tokens=100, completion_tokens=50)]
            },
            {
                "id": "run_2",
                "model": "Qwen3.6-35B",
                "endpoint": "http://127.0.0.1:8081",
                "timestamp": "2026-09-11 12:05:00",
                "duration": "12.00s",
                "raw_output": "test2",
                "turns": [make_turn_dict("Turn 1 (Cold Start)", prompt_tokens=120, completion_tokens=60)]
            }
        ]
        self.benchmark_ui.st.multiselect.return_value = ["#1: 2026-09-11 12:00:00 - Qwen3.6-27B", "#2: 2026-09-11 12:05:00 - Qwen3.6-35B"]
        self.benchmark_ui.st.radio.return_value = "Bar Chart"
        self.benchmark_ui.st.selectbox.return_value = "Generation (t/s)"

        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.columns = ["Turn", "Model", "Run ID", "Generation (t/s)"]
        mock_df.__getitem__.return_value.sum.return_value = 150
        self.mock_pd.DataFrame.return_value = mock_df

        self.benchmark_ui.main()

        self.benchmark_ui.st.plotly_chart.assert_called()

        # Test Line Chart
        self.benchmark_ui.st.radio.return_value = "Line Chart"
        self.benchmark_ui.main()

        # Test Pie Chart
        self.benchmark_ui.st.radio.return_value = "Pie Chart"
        self.benchmark_ui.main()

        # Test empty pie chart data
        mock_pie_empty = MagicMock()
        mock_pie_empty.empty = True
        self.mock_pd.DataFrame.side_effect = [mock_df, mock_pie_empty]
        self.benchmark_ui.main()
        self.benchmark_ui.st.warning.assert_any_call("No token data available for Pie Chart.")

        # Test empty turns df
        mock_empty_df = MagicMock()
        mock_empty_df.empty = True
        mock_empty_df.columns = []
        self.mock_pd.DataFrame.side_effect = None
        self.mock_pd.DataFrame.return_value = mock_empty_df
        self.benchmark_ui.st.radio.return_value = "Bar Chart"
        self.benchmark_ui.main()
        self.benchmark_ui.st.warning.assert_any_call("No valid turn data available for visualization.")

    def test_main_visualizations_clear_history(self):
        self.session_state.benchmark_history = [
            {
                "id": "run_1",
                "model": "Qwen3.6-27B",
                "endpoint": "http://127.0.0.1:8081",
                "timestamp": "2026-09-11 12:00:00",
                "duration": "10.00s",
                "raw_output": "test",
                "turns": [make_turn_dict("Turn 1 (Cold Start)")]
            }
        ]
        def mock_button(label, **kwargs):
            return label == "🗑️ Clear History"

        self.benchmark_ui.st.button.side_effect = mock_button
        self.benchmark_ui.main()

        self.assertEqual(self.session_state.benchmark_history, [])
        self.benchmark_ui.st.rerun.assert_called_once()


if __name__ == "__main__":
    unittest.main()
