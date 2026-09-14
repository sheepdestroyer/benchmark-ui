"""
Agentic benchmark harness for Harbor CLI / Terminal-Bench 2.0 and standalone multi-turn agent tasks.
"""

import configparser
import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

STANDALONE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash shell command inside the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text contents of a file inside the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to read",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file inside the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write into the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


def get_podman_socket_path() -> str:
    """
    Detect user-level Podman socket path or DOCKER_HOST override.
    Returns path prefixed with 'unix://', e.g. 'unix:///run/user/<uid>/podman/podman.sock'.
    Raises FileNotFoundError if no DOCKER_HOST is set and the user socket doesn't exist.
    """
    docker_host = os.environ.get("DOCKER_HOST")
    if docker_host and docker_host.strip():
        return docker_host.strip()

    uid = os.getuid() if hasattr(os, "getuid") else 1000
    candidate = f"/run/user/{uid}/podman/podman.sock"
    if os.path.exists(candidate):
        return f"unix://{candidate}"

    raise FileNotFoundError(f"Podman socket not found at {candidate}")


def is_harbor_available() -> bool:
    """Check if 'harbor' CLI binary is available on PATH."""
    return shutil.which("harbor") is not None


def parse_harbor_output(
    stdout: str,
    stderr: str,
    returncode: int,
    duration: float,
    default_task_name: str = "terminal-bench-2",
) -> List[Dict[str, Any]]:
    """
    Parse Harbor CLI stdout/stderr or JSON blocks to extract task results.
    Returns a list of structured task result dictionaries.
    """
    tasks: List[Dict[str, Any]] = []

    # 1. Try parsing JSON lines or embedded JSON blocks
    for line in stdout.splitlines():
        line_clean = line.strip()
        if (
            line_clean.startswith("{")
            and line_clean.endswith("}")
            and "task_name" in line_clean
        ):
            try:
                data = json.loads(line_clean)
                if isinstance(data, dict) and "task_name" in data:
                    tasks.append(
                        {
                            "task_name": str(data.get("task_name")),
                            "passed": bool(
                                data.get("passed", data.get("status") == "passed")
                            ),
                            "turns_taken": int(data.get("turns_taken", 0)),
                            "tool_calls": int(data.get("tool_calls", 0)),
                            "prompt_tokens": int(data.get("prompt_tokens", 0)),
                            "reasoning_tokens": int(data.get("reasoning_tokens", 0)),
                            "completion_tokens": int(data.get("completion_tokens", 0)),
                            "duration_seconds": round(
                                float(data.get("duration_seconds", duration)), 3
                            ),
                        }
                    )
            except (json.JSONDecodeError, ValueError):
                pass

    if tasks:
        return tasks

    # 2. Try regex parsing line-by-line formatted output
    # e.g.: Task: fix-git | Status: PASS | Turns: 3 | Tool calls: 4 | Tokens: prompt=100, completion=50, reasoning=20 | Duration: 4.5s
    task_re = re.compile(
        r"Task:\s*(?P<task>[^\s|]+)\s*\|\s*Status:\s*(?P<status>PASS|FAIL|PASSED|FAILED)"
        r"(?:\s*\|\s*Turns:\s*(?P<turns>\d+))?"
        r"(?:\s*\|\s*Tool calls:\s*(?P<tools>\d+))?"
        r"(?:\s*\|\s*Tokens:\s*prompt=(?P<prompt>\d+),\s*completion=(?P<comp>\d+)(?:,\s*reasoning=(?P<reas>\d+))?)?"
        r"(?:\s*\|\s*Duration:\s*(?P<dur>[0-9.]+)s?)?",
        re.IGNORECASE,
    )
    for line in stdout.splitlines():
        match = task_re.search(line)
        if match:
            status = match.group("status").upper()
            passed = status in ("PASS", "PASSED")
            tasks.append(
                {
                    "task_name": match.group("task"),
                    "passed": passed,
                    "turns_taken": int(match.group("turns") or 0),
                    "tool_calls": int(match.group("tools") or 0),
                    "prompt_tokens": int(match.group("prompt") or 0),
                    "reasoning_tokens": int(match.group("reas") or 0),
                    "completion_tokens": int(match.group("comp") or 0),
                    "duration_seconds": round(float(match.group("dur") or duration), 3),
                }
            )

    if tasks:
        return tasks

    # 3. Fallback: summarize overall run
    overall_passed = returncode == 0
    return [
        {
            "task_name": default_task_name,
            "passed": overall_passed,
            "turns_taken": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "reasoning_tokens": 0,
            "completion_tokens": 0,
            "duration_seconds": round(duration, 3),
        }
    ]


def normalize_harbor_endpoint(endpoint: str) -> str:
    """Normalize endpoint for container networking (host.containers.internal)."""
    ep = endpoint.strip().rstrip("/")
    ep = ep.replace("://127.0.0.1", "://host.containers.internal")
    ep = ep.replace("://localhost", "://host.containers.internal")
    if not ep.endswith("/v1"):
        ep = f"{ep}/v1"
    return ep


def run_harbor_benchmark(
    endpoint: str,
    model: str,
    task_filter: str = "all",
    num_tasks: int = 1,
    agent: str = "pi",
    dataset: str = "terminal-bench/terminal-bench-2",
    api_key: Optional[str] = None,
    timeout: int = 1800,
) -> List[Dict[str, Any]]:
    """
    Build and execute Harbor CLI command matching run-tb-pi.sh.
    Parses Harbor output to extract structured task results.
    """
    effective_endpoint = normalize_harbor_endpoint(endpoint)
    effective_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or "dummy"
    )

    cmd = [
        "harbor",
        "run",
        "-d",
        dataset,
        "-a",
        agent,
        "-m",
        model,
        "-l",
        str(num_tasks),
    ]
    if task_filter and task_filter != "all":
        cmd.extend(["-i", task_filter])

    cmd.extend(
        [
            "--ae",
            f"OPENAI_BASE_URL={effective_endpoint}",
            "--ae",
            f"OPENAI_API_KEY={effective_key}",
        ]
    )

    start_time = time.time()
    default_task = task_filter if task_filter != "all" else dataset
    try:
        docker_host = get_podman_socket_path()
        env = os.environ.copy()
        env["DOCKER_HOST"] = docker_host
        env["OPENAI_BASE_URL"] = effective_endpoint
        env["OPENAI_API_KEY"] = effective_key

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
        duration = time.time() - start_time
        return parse_harbor_output(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            duration=duration,
            default_task_name=default_task,
        )
    except Exception:
        duration = time.time() - start_time
        return [
            {
                "task_name": default_task,
                "passed": False,
                "turns_taken": 0,
                "tool_calls": 0,
                "prompt_tokens": 0,
                "reasoning_tokens": 0,
                "completion_tokens": 0,
                "duration_seconds": round(duration, 3),
            }
        ]


def safe_resolve_path(sandbox_dir: str, rel_path: str) -> Path:
    """Resolve a relative path safely within sandbox_dir, preventing path traversal."""
    base = Path(sandbox_dir).resolve()
    target = (base / rel_path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise PermissionError(f"Access denied: path '{rel_path}' is outside sandbox.")
    return target


def execute_sandbox_tool(
    sandbox_dir: str, tool_name: str, arguments: Dict[str, Any]
) -> str:
    """Execute bash, read_file, or write_file safely inside sandbox_dir."""
    if tool_name == "read_file":
        path_str = arguments.get("path", "")
        try:
            target = safe_resolve_path(sandbox_dir, path_str)
            if not target.exists():
                return f"Error: File not found: {path_str}"
            if target.is_dir():
                return f"Error: Path is a directory: {path_str}"
            return target.read_text(encoding="utf-8")
        except PermissionError as pe:
            return f"Error: {pe}"
        except Exception as e:
            return f"Error reading file: {e}"

    elif tool_name == "write_file":
        path_str = arguments.get("path", "")
        content = arguments.get("content", "")
        try:
            target = safe_resolve_path(sandbox_dir, path_str)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"File written successfully: {path_str}"
        except PermissionError as pe:
            return f"Error: {pe}"
        except Exception as e:
            return f"Error writing file: {e}"

    elif tool_name == "bash":
        cmd = arguments.get("command", "")
        if not cmd.strip():
            return "Error: Empty command"
        sanitized_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": sandbox_dir,
            "TMPDIR": sandbox_dir,
            "LANG": "C.UTF-8",
        }
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=sandbox_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=sanitized_env,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    proc.communicate(timeout=5)
                except Exception:  # pragma: no cover
                    pass
                return "Error: Command timed out after 30 seconds"

            out = stdout
            if stderr:
                out = (out + "\n" if out else "") + stderr
            if proc.returncode != 0:
                out = (
                    out + f"\n(exit code {proc.returncode})"
                    if out
                    else f"(exit code {proc.returncode})"
                )
            return out if out else "(command produced no output)"
        except Exception as e:
            return f"Error executing command: {e}"

    return f"Error: Unknown tool '{tool_name}'"


# ---------------------------------------------------------------------------
# Standalone Agentic Tasks Definition
# ---------------------------------------------------------------------------


def setup_fix_syntax(sandbox_dir: str) -> str:
    """Setup task 'fix-syntax': broken python script needing syntax repair."""
    broken_code = (
        "def calculate_average(numbers):\n"
        "    if not numbers:\n"
        "        return 0\n"
        "    total = sum(numbers\n"
        "    return total / len(numbers)\n\n"
        'if __name__ == "__main__":\n'
        "    print(calculate_average([10, 20, 30]))\n"
    )
    Path(sandbox_dir, "app.py").write_text(broken_code, encoding="utf-8")
    return (
        "The script 'app.py' contains a Python syntax error. "
        "Inspect 'app.py', repair the syntax error, and ensure running 'python3 app.py' prints the average 20.0."
    )


def verify_fix_syntax(sandbox_dir: str) -> bool:
    """Verify task 'fix-syntax' passes."""
    try:
        res = subprocess.run(
            ["python3", "app.py"],
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return res.returncode == 0 and "20.0" in res.stdout.strip()
    except Exception:
        return False


def setup_log_analysis(sandbox_dir: str) -> str:
    """Setup task 'log-analysis': extract failure root causes into report.json."""
    logs = (
        "2026-09-14 10:00:01 [INFO] Server started on port 8080\n"
        "2026-09-14 10:05:23 [WARNING] High memory usage: 85%\n"
        "2026-09-14 10:12:45 [ERROR] [ERR_DB_TIMEOUT] Database connection timed out after 30000ms\n"
        "2026-09-14 10:15:10 [INFO] Worker health check OK\n"
        "2026-09-14 10:20:00 [ERROR] [ERR_DISK_FULL] No space left on device /mnt/data\n"
        "2026-09-14 10:22:15 [ERROR] [ERR_AUTH_FAIL] Unauthorized API token provided\n"
        "2026-09-14 10:25:00 [INFO] Periodic flush complete\n"
    )
    Path(sandbox_dir, "server.log").write_text(logs, encoding="utf-8")
    return (
        "Analyze 'server.log'. Extract all unique error codes (e.g. ERR_DB_TIMEOUT) and their failure messages "
        "into a JSON file named 'report.json'. The JSON must have the schema: "
        '{"errors": [{"code": "...", "message": "..."}], "total_errors": 3}.'
    )


def verify_log_analysis(sandbox_dir: str) -> bool:
    """Verify task 'log-analysis' passes."""
    report_file = Path(sandbox_dir, "report.json")
    if not report_file.exists():
        return False
    try:
        data = json.loads(report_file.read_text(encoding="utf-8"))
        if data.get("total_errors") != 3:
            return False
        errors = data.get("errors", [])
        codes = {e.get("code") for e in errors}
        return {"ERR_DB_TIMEOUT", "ERR_DISK_FULL", "ERR_AUTH_FAIL"}.issubset(codes)
    except Exception:
        return False


def setup_git_repair(sandbox_dir: str) -> str:
    """Setup task 'git-repair': git repository with detached HEAD needing recovery."""
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Test User"
    env["GIT_AUTHOR_EMAIL"] = "test@example.com"
    env["GIT_COMMITTER_NAME"] = "Test User"
    env["GIT_COMMITTER_EMAIL"] = "test@example.com"

    def run_git(args: List[str]):
        subprocess.run(
            ["git"] + args,
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    run_git(["init", "-b", "main"])
    run_git(["config", "user.name", "Test User"])
    run_git(["config", "user.email", "test@example.com"])

    Path(sandbox_dir, "main.txt").write_text("initial main content\n", encoding="utf-8")
    run_git(["add", "main.txt"])
    run_git(["commit", "-m", "Initial commit on main"])

    Path(sandbox_dir, "feature.txt").write_text(
        "detached feature work\n", encoding="utf-8"
    )
    run_git(["add", "feature.txt"])
    run_git(["commit", "-m", "Detached feature commit"])

    # Checkout previous commit to create detached HEAD state
    run_git(["checkout", "HEAD~1"])

    return (
        "The git repository in this workspace is in a detached HEAD state. "
        "Recover the detached commit containing 'feature.txt' onto branch 'main' so that 'main' "
        "is checked out with a clean working tree and contains 'feature.txt'."
    )


def verify_git_repair(sandbox_dir: str) -> bool:
    """Verify task 'git-repair' passes."""
    try:
        branch_res = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        current_branch = branch_res.stdout.strip()
        if current_branch != "main":
            return False

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if status_res.stdout.strip() != "":
            return False

        return Path(sandbox_dir, "feature.txt").exists()
    except Exception:
        return False


def setup_env_config(sandbox_dir: str) -> str:
    """Setup task 'env-config': INI configuration file updates."""
    ini_content = (
        "[database]\n"
        "host = localhost\n"
        "port = 5432\n"
        "username = dev_user\n"
        "password = change_me\n"
        "ssl_mode = disabled\n\n"
        "[server]\n"
        "port = 8080\n"
        "debug = true\n"
        "workers = 1\n"
    )
    Path(sandbox_dir, "config.ini").write_text(ini_content, encoding="utf-8")
    return (
        "Update 'config.ini' for production hardening: "
        "under [database], set port to 5433 and ssl_mode to 'require'. "
        "Under [server], set debug to 'false' and workers to 4. "
        "Ensure standard INI formatting is preserved."
    )


def verify_env_config(sandbox_dir: str) -> bool:
    """Verify task 'env-config' passes."""
    cfg_file = Path(sandbox_dir, "config.ini")
    if not cfg_file.exists():
        return False
    try:
        cp = configparser.ConfigParser()
        cp.read(cfg_file, encoding="utf-8")
        if cp.get("database", "ssl_mode") != "require":
            return False
        if cp.getint("database", "port") != 5433:
            return False
        if cp.get("server", "debug").lower() != "false":
            return False
        if cp.getint("server", "workers") != 4:
            return False
        return True
    except Exception:
        return False


STANDALONE_TASKS: Dict[str, Dict[str, Callable[[str], Any]]] = {
    "fix-syntax": {"setup": setup_fix_syntax, "verify": verify_fix_syntax},
    "log-analysis": {"setup": setup_log_analysis, "verify": verify_log_analysis},
    "git-repair": {"setup": setup_git_repair, "verify": verify_git_repair},
    "env-config": {"setup": setup_env_config, "verify": verify_env_config},
}


def parse_tool_calls_from_text(content: str) -> List[Dict[str, Any]]:
    """Fallback parser for models that emit tool calls in markdown or JSON text."""
    tool_calls: List[Dict[str, Any]] = []

    # Check for ```bash ... ``` code blocks
    bash_blocks = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)
    for block in bash_blocks:
        tool_calls.append(
            {
                "id": f"call_text_{len(tool_calls)}",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": block.strip()}),
                },
            }
        )

    # Check for ```json {"tool": ..., ...} ```
    json_blocks = re.findall(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block.strip())
            if isinstance(data, dict):
                t_name = data.get("tool") or data.get("name")
                t_args = data.get("arguments") or data.get("args") or data
                if t_name in ("bash", "read_file", "write_file"):
                    tool_calls.append(
                        {
                            "id": f"call_text_{len(tool_calls)}",
                            "function": {
                                "name": t_name,
                                "arguments": json.dumps(t_args),
                            },
                        }
                    )
        except Exception:
            pass

    return tool_calls


def run_standalone_agentic_benchmark(
    endpoint: str,
    model: str,
    task_filter: str = "all",
    api_key: Optional[str] = None,
    max_turns: int = 10,
    max_tokens: int = 16384,
) -> List[Dict[str, Any]]:
    """
    Standalone multi-turn sandboxed agentic harness.
    Executes an autonomous tool-calling loop over selected tasks.
    """
    effective_key = (
        api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or ""
    )
    headers = {"Content-Type": "application/json"}
    if effective_key:
        headers["Authorization"] = f"Bearer {effective_key}"

    # Select tasks
    if task_filter == "all":
        selected_tasks = list(STANDALONE_TASKS.keys())
    else:
        filters = [f.strip() for f in task_filter.split(",") if f.strip()]
        selected_tasks = [
            t
            for t in STANDALONE_TASKS
            if any(t == f or fnmatch.fnmatch(t, f) for f in filters)
        ]
        if not selected_tasks:
            # Fallback to exact match or empty
            selected_tasks = [t for t in STANDALONE_TASKS if t == task_filter]

    chat_url = endpoint.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"
    results: List[Dict[str, Any]] = []

    for task_name in selected_tasks:
        task_def = STANDALONE_TASKS[task_name]
        start_time = time.time()
        turns_taken = 0
        tool_calls_count = 0
        prompt_tokens_total = 0
        reasoning_tokens_total = 0
        completion_tokens_total = 0
        task_passed = False

        with tempfile.TemporaryDirectory() as sandbox_dir:
            task_prompt = task_def["setup"](sandbox_dir)
            system_prompt = (
                "You are an expert autonomous software engineer. "
                "You have access to tools: 'bash', 'read_file', 'write_file'. "
                "The current working directory is your workspace. "
                "Inspect, solve the user request, and verify your solution before finishing."
            )

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt},
            ]

            for _ in range(max_turns):
                payload = {
                    "model": model,
                    "messages": messages,
                    "tools": STANDALONE_TOOLS,
                    "tool_choice": "auto",
                    "max_tokens": min(max_tokens, 4096),
                    "temperature": 0.0,
                }

                try:
                    resp = requests.post(
                        chat_url,
                        json=payload,
                        headers=headers,
                        timeout=60,
                    )
                    if resp.status_code != 200:
                        break
                    resp_json = resp.json()
                except Exception:
                    break

                usage = resp_json.get("usage", {})
                prompt_tokens_total += usage.get("prompt_tokens", 0)
                completion_tokens_total += usage.get("completion_tokens", 0)
                reasoning_tokens_total += usage.get(
                    "completion_tokens_details", {}
                ).get("reasoning_tokens", 0)

                choice = resp_json.get("choices", [{}])[0]
                assistant_msg = choice.get("message", {})
                tool_calls = assistant_msg.get("tool_calls") or []

                # Fallback to text parsing if no native tool calls were returned
                content = assistant_msg.get("content") or ""
                if not tool_calls and content:
                    tool_calls = parse_tool_calls_from_text(content)
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls

                turns_taken += 1
                messages.append(assistant_msg)

                if tool_calls:
                    tool_calls_count += len(tool_calls)
                    for tc in tool_calls:
                        tc_id = tc.get("id", f"call_{tool_calls_count}")
                        fn = tc.get("function", {})
                        fn_name = fn.get("name", "")
                        fn_args_raw = fn.get("arguments", {})
                        if isinstance(fn_args_raw, str):
                            try:
                                fn_args = json.loads(fn_args_raw)
                            except Exception:
                                fn_args = {}
                        else:
                            fn_args = fn_args_raw

                        tool_output = execute_sandbox_tool(
                            sandbox_dir, fn_name, fn_args
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": fn_name,
                                "content": tool_output,
                            }
                        )

                # Check task verification condition
                if task_def["verify"](sandbox_dir):
                    task_passed = True
                    break

                # If model finished without calling tools and didn't pass, break
                if not tool_calls:
                    break

            duration = time.time() - start_time
            results.append(
                {
                    "task_name": task_name,
                    "passed": task_passed,
                    "turns_taken": turns_taken,
                    "tool_calls": tool_calls_count,
                    "prompt_tokens": prompt_tokens_total,
                    "reasoning_tokens": reasoning_tokens_total,
                    "completion_tokens": completion_tokens_total,
                    "duration_seconds": round(duration, 3),
                }
            )

    return results


def run_agentic_suite(
    endpoint: str,
    model: str,
    task_filter: str = "all",
    prefer_harbor: bool = True,
    api_key: Optional[str] = None,
    max_tokens: int = 16384,
) -> Dict[str, Any]:
    """
    Unified entrypoint for agentic benchmarking.
    Dispatches to Harbor if available and requested, otherwise falls back to standalone harness.
    """
    harbor_used = False
    tasks: List[Dict[str, Any]] = []

    if prefer_harbor and is_harbor_available():
        try:
            tasks = run_harbor_benchmark(
                endpoint=endpoint,
                model=model,
                task_filter=task_filter,
                api_key=api_key,
            )
            is_execution_failure = not tasks or (
                len(tasks) == 1
                and not tasks[0].get("passed", False)
                and tasks[0].get("turns_taken", 0) == 0
                and tasks[0].get("tool_calls", 0) == 0
                and tasks[0].get("prompt_tokens", 0) == 0
            )
            if is_execution_failure:
                tasks = run_standalone_agentic_benchmark(
                    endpoint=endpoint,
                    model=model,
                    task_filter=task_filter,
                    api_key=api_key,
                    max_tokens=max_tokens,
                )
            else:
                harbor_used = True
        except Exception:
            tasks = run_standalone_agentic_benchmark(
                endpoint=endpoint,
                model=model,
                task_filter=task_filter,
                api_key=api_key,
                max_tokens=max_tokens,
            )
    else:
        tasks = run_standalone_agentic_benchmark(
            endpoint=endpoint,
            model=model,
            task_filter=task_filter,
            api_key=api_key,
            max_tokens=max_tokens,
        )

    tasks_total = len(tasks)
    tasks_passed = sum(1 for t in tasks if t.get("passed", False))
    average_turns = (
        round(sum(t.get("turns_taken", 0) for t in tasks) / tasks_total, 2)
        if tasks_total > 0
        else 0.0
    )
    total_tool_calls = sum(t.get("tool_calls", 0) for t in tasks)
    token_breakdown = {
        "prompt_tokens": sum(t.get("prompt_tokens", 0) for t in tasks),
        "reasoning_tokens": sum(t.get("reasoning_tokens", 0) for t in tasks),
        "completion_tokens": sum(t.get("completion_tokens", 0) for t in tasks),
    }

    return {
        "suite": "terminal-bench-2" if harbor_used else "standalone-agentic",
        "tasks_total": tasks_total,
        "tasks_passed": tasks_passed,
        "average_turns": average_turns,
        "total_tool_calls": total_tool_calls,
        "tasks": tasks,
        "token_breakdown": token_breakdown,
    }
