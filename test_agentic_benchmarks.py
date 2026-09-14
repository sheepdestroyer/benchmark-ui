"""
Unit tests for Harbor and standalone agentic benchmark harness (agentic_benchmarks.py).
"""

import configparser
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import agentic_benchmarks


class TestAgenticBenchmarks(unittest.TestCase):
    def test_get_podman_socket_path_env_override(self):
        with patch.dict(os.environ, {"DOCKER_HOST": "unix:///custom/docker.sock"}):
            path = agentic_benchmarks.get_podman_socket_path()
            self.assertEqual(path, "unix:///custom/docker.sock")

    def test_get_podman_socket_path_default_exists(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", return_value=True),
            patch("os.getuid", return_value=1002, create=True),
        ):
            path = agentic_benchmarks.get_podman_socket_path()
            self.assertEqual(path, "unix:///run/user/1002/podman/podman.sock")

    def test_get_podman_socket_path_missing_raises(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", return_value=False),
            patch("os.getuid", return_value=1002, create=True),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                agentic_benchmarks.get_podman_socket_path()
            self.assertIn("Podman socket not found at", str(ctx.exception))

    def test_is_harbor_available(self):
        with patch("shutil.which", return_value="/usr/local/bin/harbor"):
            self.assertTrue(agentic_benchmarks.is_harbor_available())

        with patch("shutil.which", return_value=None):
            self.assertFalse(agentic_benchmarks.is_harbor_available())

    def test_safe_resolve_path_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = agentic_benchmarks.safe_resolve_path(tmp_dir, "foo/bar.txt")
            self.assertTrue(res.is_relative_to(Path(tmp_dir).resolve()))

    def test_safe_resolve_path_traversal_denied(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(PermissionError):
                agentic_benchmarks.safe_resolve_path(tmp_dir, "../../../etc/passwd")

    def test_execute_sandbox_tool_read_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir, "test.txt")
            p.write_text("hello sandbox", encoding="utf-8")

            # Normal read
            out = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir, "read_file", {"path": "test.txt"}
            )
            self.assertEqual(out, "hello sandbox")

            # Missing file
            out_missing = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir, "read_file", {"path": "missing.txt"}
            )
            self.assertIn("Error: File not found", out_missing)

            # Directory read error
            d = Path(tmp_dir, "subdir")
            d.mkdir()
            out_dir = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir, "read_file", {"path": "subdir"}
            )
            self.assertIn("Error: Path is a directory", out_dir)

            # Traversal read error
            out_trav = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir, "read_file", {"path": "../secret"}
            )
            self.assertIn("Error: Access denied", out_trav)

    def test_execute_sandbox_tool_write_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Normal write
            out = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir,
                "write_file",
                {"path": "sub/out.txt", "content": "written content"},
            )
            self.assertIn("File written successfully", out)
            self.assertEqual(
                Path(tmp_dir, "sub/out.txt").read_text(encoding="utf-8"),
                "written content",
            )

            # Traversal write error
            out_trav = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir, "write_file", {"path": "../bad.txt", "content": "bad"}
            )
            self.assertIn("Error: Access denied", out_trav)

    def test_execute_sandbox_tool_bash(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Normal command
            out = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir, "bash", {"command": "echo 'running bash'"}
            )
            self.assertIn("running bash", out)

            # Command with stderr and non-zero exit
            out_err = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir, "bash", {"command": "echo 'error msg' >&2; exit 2"}
            )
            self.assertIn("error msg", out_err)
            self.assertIn("exit code 2", out_err)

            # Empty command
            out_empty = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir, "bash", {"command": "   "}
            )
            self.assertEqual(out_empty, "Error: Empty command")

            # Command producing no output
            out_noop = agentic_benchmarks.execute_sandbox_tool(
                tmp_dir, "bash", {"command": "true"}
            )
            self.assertEqual(out_noop, "(command produced no output)")

            # Command timeout
            with patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="sleep", timeout=30),
            ):
                out_timeout = agentic_benchmarks.execute_sandbox_tool(
                    tmp_dir, "bash", {"command": "sleep 100"}
                )
                self.assertIn("timed out after 30 seconds", out_timeout)

    def test_execute_sandbox_tool_unknown(self):
        out = agentic_benchmarks.execute_sandbox_tool("/tmp", "unknown_tool", {})
        self.assertIn("Error: Unknown tool", out)

    def test_parse_harbor_output_json_lines(self):
        stdout = (
            '{"task_name": "fix-git", "passed": true, "turns_taken": 3, "tool_calls": 4, "prompt_tokens": 500, "reasoning_tokens": 100, "completion_tokens": 200, "duration_seconds": 12.3}\n'
            '{"task_name": "fix-calc", "status": "passed", "turns_taken": 2, "tool_calls": 2, "prompt_tokens": 300, "reasoning_tokens": 50, "completion_tokens": 100, "duration_seconds": 8.1}\n'
        )
        tasks = agentic_benchmarks.parse_harbor_output(
            stdout, stderr="", returncode=0, duration=20.4
        )
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["task_name"], "fix-git")
        self.assertTrue(tasks[0]["passed"])
        self.assertEqual(tasks[0]["turns_taken"], 3)
        self.assertEqual(tasks[0]["tool_calls"], 4)
        self.assertEqual(tasks[0]["prompt_tokens"], 500)
        self.assertEqual(tasks[0]["reasoning_tokens"], 100)
        self.assertEqual(tasks[0]["completion_tokens"], 200)
        self.assertEqual(tasks[0]["duration_seconds"], 12.3)

    def test_parse_harbor_output_regex_lines(self):
        stdout = (
            "Task: terminal-bench/fix-git | Status: PASS | Turns: 4 | Tool calls: 6 | Tokens: prompt=1200, completion=400, reasoning=150 | Duration: 15.5s\n"
            "Task: terminal-bench/repair-db | Status: FAIL | Turns: 10 | Tool calls: 12 | Tokens: prompt=3000, completion=800, reasoning=400 | Duration: 35.0s\n"
        )
        tasks = agentic_benchmarks.parse_harbor_output(
            stdout, stderr="", returncode=0, duration=50.5
        )
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["task_name"], "terminal-bench/fix-git")
        self.assertTrue(tasks[0]["passed"])
        self.assertEqual(tasks[0]["turns_taken"], 4)
        self.assertEqual(tasks[0]["tool_calls"], 6)
        self.assertEqual(tasks[0]["prompt_tokens"], 1200)
        self.assertEqual(tasks[0]["completion_tokens"], 400)
        self.assertEqual(tasks[0]["reasoning_tokens"], 150)
        self.assertEqual(tasks[0]["duration_seconds"], 15.5)

        self.assertEqual(tasks[1]["task_name"], "terminal-bench/repair-db")
        self.assertFalse(tasks[1]["passed"])

    def test_parse_harbor_output_fallback_overall(self):
        tasks = agentic_benchmarks.parse_harbor_output(
            stdout="Running harbor with no tasks parsed...",
            stderr="warning: cache cold",
            returncode=0,
            duration=14.2,
            default_task_name="my-dataset",
        )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_name"], "my-dataset")
        self.assertTrue(tasks[0]["passed"])
        self.assertEqual(tasks[0]["duration_seconds"], 14.2)

    def test_run_harbor_benchmark_success(self):
        mock_proc = MagicMock()
        mock_proc.stdout = '{"task_name": "fix-git", "passed": true, "turns_taken": 2, "tool_calls": 3, "prompt_tokens": 400, "reasoning_tokens": 100, "completion_tokens": 150, "duration_seconds": 10.0}'
        mock_proc.stderr = ""
        mock_proc.returncode = 0

        with (
            patch(
                "agentic_benchmarks.get_podman_socket_path",
                return_value="unix:///run/user/1000/podman/podman.sock",
            ),
            patch("subprocess.run", return_value=mock_proc) as mock_subproc,
        ):
            tasks = agentic_benchmarks.run_harbor_benchmark(
                endpoint="http://127.0.0.1:8081",
                model="local-llama/Qwen3.6-35B",
                task_filter="fix-git",
                num_tasks=1,
                api_key="custom-key",
            )
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["task_name"], "fix-git")
            self.assertTrue(tasks[0]["passed"])

            # Verify subprocess call args
            cmd = mock_subproc.call_args[0][0]
            self.assertIn("harbor", cmd)
            self.assertIn("-i", cmd)
            self.assertIn("fix-git", cmd)
            self.assertIn("--ae", cmd)
            self.assertIn("OPENAI_BASE_URL=http://127.0.0.1:8081", cmd)
            self.assertIn("OPENAI_API_KEY=custom-key", cmd)

    def test_run_harbor_benchmark_subprocess_exception(self):
        with (
            patch(
                "agentic_benchmarks.get_podman_socket_path",
                return_value="unix:///run/user/1000/podman/podman.sock",
            ),
            patch("subprocess.run", side_effect=RuntimeError("Harbor binary failed")),
        ):
            tasks = agentic_benchmarks.run_harbor_benchmark(
                endpoint="http://127.0.0.1:8081",
                model="local-llama/Qwen3.6-35B",
                task_filter="all",
            )
            self.assertEqual(len(tasks), 1)
            self.assertFalse(tasks[0]["passed"])
            self.assertEqual(tasks[0]["task_name"], "terminal-bench/terminal-bench-2")

    def test_standalone_tasks_setup_and_verify(self):
        # 1. fix-syntax
        with tempfile.TemporaryDirectory() as tmp_dir:
            agentic_benchmarks.setup_fix_syntax(tmp_dir)
            self.assertFalse(agentic_benchmarks.verify_fix_syntax(tmp_dir))
            # Fix code
            fixed_code = (
                "def calculate_average(numbers):\n"
                "    if not numbers:\n"
                "        return 0\n"
                "    total = sum(numbers)\n"
                "    return total / len(numbers)\n\n"
                'if __name__ == "__main__":\n'
                "    print(calculate_average([10, 20, 30]))\n"
            )
            Path(tmp_dir, "app.py").write_text(fixed_code, encoding="utf-8")
            self.assertTrue(agentic_benchmarks.verify_fix_syntax(tmp_dir))

        # 2. log-analysis
        with tempfile.TemporaryDirectory() as tmp_dir:
            agentic_benchmarks.setup_log_analysis(tmp_dir)
            self.assertFalse(agentic_benchmarks.verify_log_analysis(tmp_dir))
            # Create valid report.json
            valid_report = {
                "errors": [
                    {"code": "ERR_DB_TIMEOUT", "message": "Database timeout"},
                    {"code": "ERR_DISK_FULL", "message": "Disk full"},
                    {"code": "ERR_AUTH_FAIL", "message": "Auth fail"},
                ],
                "total_errors": 3,
            }
            Path(tmp_dir, "report.json").write_text(
                json.dumps(valid_report), encoding="utf-8"
            )
            self.assertTrue(agentic_benchmarks.verify_log_analysis(tmp_dir))

        # 3. git-repair
        with tempfile.TemporaryDirectory() as tmp_dir:
            agentic_benchmarks.setup_git_repair(tmp_dir)
            self.assertFalse(agentic_benchmarks.verify_git_repair(tmp_dir))
            # Restore branch main to include detached commit
            subprocess.run(
                ["git", "branch", "-f", "main", "HEAD@{1}"], cwd=tmp_dir, check=True
            )
            subprocess.run(["git", "checkout", "main"], cwd=tmp_dir, check=True)
            self.assertTrue(agentic_benchmarks.verify_git_repair(tmp_dir))

        # 4. env-config
        with tempfile.TemporaryDirectory() as tmp_dir:
            agentic_benchmarks.setup_env_config(tmp_dir)
            self.assertFalse(agentic_benchmarks.verify_env_config(tmp_dir))
            # Update config.ini
            cp = configparser.ConfigParser()
            cp.read(Path(tmp_dir, "config.ini"), encoding="utf-8")
            cp.set("database", "port", "5433")
            cp.set("database", "ssl_mode", "require")
            cp.set("server", "debug", "false")
            cp.set("server", "workers", "4")
            with open(Path(tmp_dir, "config.ini"), "w", encoding="utf-8") as f:
                cp.write(f)
            self.assertTrue(agentic_benchmarks.verify_env_config(tmp_dir))

    def test_parse_tool_calls_from_text(self):
        text_with_bash = "I will run bash:\n```bash\nls -la\n```"
        calls = agentic_benchmarks.parse_tool_calls_from_text(text_with_bash)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")
        self.assertIn("ls -la", calls[0]["function"]["arguments"])

        text_with_json = '```json\n{"tool": "read_file", "path": "app.py"}\n```'
        calls2 = agentic_benchmarks.parse_tool_calls_from_text(text_with_json)
        self.assertEqual(len(calls2), 1)
        self.assertEqual(calls2[0]["function"]["name"], "read_file")
        self.assertIn("app.py", calls2[0]["function"]["arguments"])

    def test_run_standalone_agentic_benchmark_success_loop(self):
        # Mock requests.post to simulate multi-turn interaction solving fix-syntax
        mock_resp_1 = MagicMock()
        mock_resp_1.status_code = 200
        mock_resp_1.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "app.py"}),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        }

        fixed_code = (
            "def calculate_average(numbers):\n"
            "    if not numbers:\n"
            "        return 0\n"
            "    total = sum(numbers)\n"
            "    return total / len(numbers)\n\n"
            'if __name__ == "__main__":\n'
            "    print(calculate_average([10, 20, 30]))\n"
        )

        mock_resp_2 = MagicMock()
        mock_resp_2.status_code = 200
        mock_resp_2.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps(
                                        {"path": "app.py", "content": fixed_code}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }

        with patch("requests.post", side_effect=[mock_resp_1, mock_resp_2]):
            tasks = agentic_benchmarks.run_standalone_agentic_benchmark(
                endpoint="http://127.0.0.1:8081",
                model="local-model",
                task_filter="fix-syntax",
                max_turns=5,
            )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_name"], "fix-syntax")
        self.assertTrue(tasks[0]["passed"])
        self.assertEqual(tasks[0]["turns_taken"], 2)
        self.assertEqual(tasks[0]["tool_calls"], 2)
        self.assertEqual(tasks[0]["prompt_tokens"], 350)
        self.assertEqual(tasks[0]["reasoning_tokens"], 30)
        self.assertEqual(tasks[0]["completion_tokens"], 80)

    def test_run_standalone_agentic_benchmark_non_200_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("requests.post", return_value=mock_resp):
            tasks = agentic_benchmarks.run_standalone_agentic_benchmark(
                endpoint="http://127.0.0.1:8081",
                model="local-model",
                task_filter="fix-syntax",
            )

        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0]["passed"])
        self.assertEqual(tasks[0]["turns_taken"], 0)

    def test_run_standalone_agentic_benchmark_no_tool_calls_no_pass(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": "I cannot fix this"}}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

        with patch("requests.post", return_value=mock_resp):
            tasks = agentic_benchmarks.run_standalone_agentic_benchmark(
                endpoint="http://127.0.0.1:8081",
                model="local-model",
                task_filter="fix-syntax",
            )

        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0]["passed"])
        self.assertEqual(tasks[0]["turns_taken"], 1)
        self.assertEqual(tasks[0]["tool_calls"], 0)

    def test_run_agentic_suite_dispatches_harbor_when_available(self):
        mock_harbor_tasks = [
            {
                "task_name": "task-1",
                "passed": True,
                "turns_taken": 2,
                "tool_calls": 3,
                "prompt_tokens": 400,
                "reasoning_tokens": 100,
                "completion_tokens": 150,
                "duration_seconds": 5.2,
            }
        ]

        with (
            patch("agentic_benchmarks.is_harbor_available", return_value=True),
            patch(
                "agentic_benchmarks.run_harbor_benchmark",
                return_value=mock_harbor_tasks,
            ) as mock_hb,
        ):
            summary = agentic_benchmarks.run_agentic_suite(
                endpoint="http://127.0.0.1:8081",
                model="model-1",
                prefer_harbor=True,
            )
            mock_hb.assert_called_once()
            self.assertEqual(summary["suite"], "terminal-bench-2")
            self.assertEqual(summary["tasks_total"], 1)
            self.assertEqual(summary["tasks_passed"], 1)
            self.assertEqual(summary["average_turns"], 2.0)
            self.assertEqual(summary["total_tool_calls"], 3)
            self.assertEqual(summary["token_breakdown"]["prompt_tokens"], 400)
            self.assertEqual(summary["token_breakdown"]["reasoning_tokens"], 100)
            self.assertEqual(summary["token_breakdown"]["completion_tokens"], 150)

    def test_run_agentic_suite_harbor_exception_falls_back_to_standalone(self):
        mock_standalone_tasks = [
            {
                "task_name": "fix-syntax",
                "passed": True,
                "turns_taken": 1,
                "tool_calls": 1,
                "prompt_tokens": 200,
                "reasoning_tokens": 50,
                "completion_tokens": 80,
                "duration_seconds": 3.1,
            }
        ]

        with (
            patch("agentic_benchmarks.is_harbor_available", return_value=True),
            patch(
                "agentic_benchmarks.run_harbor_benchmark",
                side_effect=RuntimeError("Harbor crash"),
            ),
            patch(
                "agentic_benchmarks.run_standalone_agentic_benchmark",
                return_value=mock_standalone_tasks,
            ) as mock_sa,
        ):
            summary = agentic_benchmarks.run_agentic_suite(
                endpoint="http://127.0.0.1:8081",
                model="model-1",
                prefer_harbor=True,
            )
            mock_sa.assert_called_once()
            self.assertEqual(summary["suite"], "standalone-agentic")
            self.assertEqual(summary["tasks_total"], 1)
            self.assertEqual(summary["tasks_passed"], 1)

    def test_run_agentic_suite_standalone_when_harbor_unavailable(self):
        mock_standalone_tasks = [
            {
                "task_name": "fix-syntax",
                "passed": False,
                "turns_taken": 3,
                "tool_calls": 2,
                "prompt_tokens": 100,
                "reasoning_tokens": 0,
                "completion_tokens": 50,
                "duration_seconds": 2.0,
            }
        ]

        with (
            patch("agentic_benchmarks.is_harbor_available", return_value=False),
            patch(
                "agentic_benchmarks.run_standalone_agentic_benchmark",
                return_value=mock_standalone_tasks,
            ) as mock_sa,
        ):
            summary = agentic_benchmarks.run_agentic_suite(
                endpoint="http://127.0.0.1:8081",
                model="model-1",
                prefer_harbor=True,
            )
            mock_sa.assert_called_once()
            self.assertEqual(summary["suite"], "standalone-agentic")
            self.assertEqual(summary["tasks_total"], 1)
            self.assertEqual(summary["tasks_passed"], 0)
            self.assertEqual(summary["average_turns"], 3.0)

    def test_parse_harbor_output_corrupted_json_lines(self):
        stdout = '{"task_name": broken}\n{"foo": "task_name"}\n'
        tasks = agentic_benchmarks.parse_harbor_output(
            stdout, stderr="", returncode=1, duration=5.0
        )
        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0]["passed"])

    def test_execute_sandbox_tool_io_exceptions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir, "file.txt")
            p.write_text("content", encoding="utf-8")

            with patch.object(Path, "read_text", side_effect=OSError("Read error")):
                out_read = agentic_benchmarks.execute_sandbox_tool(
                    tmp_dir, "read_file", {"path": "file.txt"}
                )
                self.assertIn("Error reading file", out_read)

            with patch.object(Path, "write_text", side_effect=OSError("Write error")):
                out_write = agentic_benchmarks.execute_sandbox_tool(
                    tmp_dir, "write_file", {"path": "file.txt", "content": "data"}
                )
                self.assertIn("Error writing file", out_write)

            with patch("subprocess.run", side_effect=OSError("Exec error")):
                out_bash = agentic_benchmarks.execute_sandbox_tool(
                    tmp_dir, "bash", {"command": "ls"}
                )
                self.assertIn("Error executing command", out_bash)

    def test_verify_fix_syntax_exception(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("subprocess.run", side_effect=OSError("Failed exec")):
                self.assertFalse(agentic_benchmarks.verify_fix_syntax(tmp_dir))

    def test_verify_log_analysis_branches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Missing file
            self.assertFalse(agentic_benchmarks.verify_log_analysis(tmp_dir))

            # 2. Corrupted JSON
            p = Path(tmp_dir, "report.json")
            p.write_text("corrupted json", encoding="utf-8")
            self.assertFalse(agentic_benchmarks.verify_log_analysis(tmp_dir))

            # 3. total_errors != 3
            p.write_text(
                json.dumps({"total_errors": 1, "errors": []}), encoding="utf-8"
            )
            self.assertFalse(agentic_benchmarks.verify_log_analysis(tmp_dir))

            # 4. Missing required error code
            p.write_text(
                json.dumps(
                    {
                        "total_errors": 3,
                        "errors": [
                            {"code": "ERR_DB_TIMEOUT"},
                            {"code": "ERR_DISK_FULL"},
                            {"code": "OTHER"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(agentic_benchmarks.verify_log_analysis(tmp_dir))

    def test_verify_git_repair_branches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Branch not main
            with patch("subprocess.run") as mock_subproc:
                mock_subproc.return_value = MagicMock(stdout="detached\n", returncode=0)
                self.assertFalse(agentic_benchmarks.verify_git_repair(tmp_dir))

            # 2. Dirty working tree
            with patch("subprocess.run") as mock_subproc:
                mock_subproc.side_effect = [
                    MagicMock(stdout="main\n", returncode=0),
                    MagicMock(stdout=" M dirty.txt\n", returncode=0),
                ]
                self.assertFalse(agentic_benchmarks.verify_git_repair(tmp_dir))

            # 3. Exception in git
            with patch("subprocess.run", side_effect=OSError("Git crashed")):
                self.assertFalse(agentic_benchmarks.verify_git_repair(tmp_dir))

    def test_verify_env_config_branches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Missing config.ini
            self.assertFalse(agentic_benchmarks.verify_env_config(tmp_dir))

            cfg_file = Path(tmp_dir, "config.ini")

            # 2. ssl_mode != require
            cfg_file.write_text(
                "[database]\nport=5433\nssl_mode=disabled\n[server]\ndebug=false\nworkers=4\n",
                encoding="utf-8",
            )
            self.assertFalse(agentic_benchmarks.verify_env_config(tmp_dir))

            # 3. port != 5433
            cfg_file.write_text(
                "[database]\nport=5432\nssl_mode=require\n[server]\ndebug=false\nworkers=4\n",
                encoding="utf-8",
            )
            self.assertFalse(agentic_benchmarks.verify_env_config(tmp_dir))

            # 4. debug != false
            cfg_file.write_text(
                "[database]\nport=5433\nssl_mode=require\n[server]\ndebug=true\nworkers=4\n",
                encoding="utf-8",
            )
            self.assertFalse(agentic_benchmarks.verify_env_config(tmp_dir))

            # 5. workers != 4
            cfg_file.write_text(
                "[database]\nport=5433\nssl_mode=require\n[server]\ndebug=false\nworkers=1\n",
                encoding="utf-8",
            )
            self.assertFalse(agentic_benchmarks.verify_env_config(tmp_dir))

            # 6. Corrupted file causing ConfigParser exception
            cfg_file.write_text(
                "invalid ini content without section header\n", encoding="utf-8"
            )
            self.assertFalse(agentic_benchmarks.verify_env_config(tmp_dir))

    def test_parse_tool_calls_from_text_edge_cases(self):
        # Invalid JSON inside ```json
        calls_bad_json = agentic_benchmarks.parse_tool_calls_from_text(
            "```json\nnot a json\n```"
        )
        self.assertEqual(calls_bad_json, [])

        # Valid JSON but unknown tool
        calls_unknown = agentic_benchmarks.parse_tool_calls_from_text(
            '```json\n{"tool": "unknown_tool", "args": {}}\n```'
        )
        self.assertEqual(calls_unknown, [])

    def test_run_standalone_agentic_benchmark_filters_and_arguments(self):
        # Test comma-separated filter with env OPENAI_API_KEY
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}),
            patch("requests.post", side_effect=Exception("Connection refused")),
        ):
            tasks = agentic_benchmarks.run_standalone_agentic_benchmark(
                endpoint="http://127.0.0.1:8081",
                model="model-1",
                task_filter="fix-syntax,log-analysis",
            )
            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0]["task_name"], "fix-syntax")
            self.assertEqual(tasks[1]["task_name"], "log-analysis")

        # Test unknown filter fallback
        with patch("requests.post", side_effect=Exception("Failed")):
            tasks_unknown = agentic_benchmarks.run_standalone_agentic_benchmark(
                endpoint="http://127.0.0.1:8081",
                model="model-1",
                task_filter="custom-unknown",
            )
            self.assertEqual(len(tasks_unknown), 0)

    def test_run_standalone_agentic_benchmark_max_turns_and_tool_args(self):
        # Mock responses where model calls tools with dict arguments and invalid json string arguments
        # and exhausts max_turns without passing
        mock_resp_dict_args = MagicMock()
        mock_resp_dict_args.status_code = 200
        mock_resp_dict_args.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_d1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"path": "app.py"},
                                },
                            },
                            {
                                "id": "call_d2",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "invalid-json-string {",
                                },
                            },
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        with patch("requests.post", return_value=mock_resp_dict_args):
            tasks = agentic_benchmarks.run_standalone_agentic_benchmark(
                endpoint="http://127.0.0.1:8081",
                model="model-1",
                task_filter="fix-syntax",
                max_turns=2,
            )
            self.assertEqual(len(tasks), 1)
            self.assertFalse(tasks[0]["passed"])
            self.assertEqual(tasks[0]["turns_taken"], 2)

    def test_parse_tool_calls_from_text_json_array(self):
        calls_array = agentic_benchmarks.parse_tool_calls_from_text(
            "```json\n[1, 2, 3]\n```"
        )
        self.assertEqual(calls_array, [])

    def test_run_standalone_agentic_benchmark_all_tasks(self):
        with patch("requests.post", side_effect=Exception("Connection refused")):
            tasks = agentic_benchmarks.run_standalone_agentic_benchmark(
                endpoint="http://127.0.0.1:8081",
                model="model-1",
                task_filter="all",
            )
            self.assertEqual(len(tasks), 4)
            task_names = [t["task_name"] for t in tasks]
            self.assertIn("fix-syntax", task_names)
            self.assertIn("log-analysis", task_names)
            self.assertIn("git-repair", task_names)
            self.assertIn("env-config", task_names)


if __name__ == "__main__":
    unittest.main()
