#!/usr/bin/env python3
import os
import sys
import json
import datetime
import subprocess
import time
import atexit
from pathlib import Path

BENCH_DIR = Path(__file__).parent.resolve()
HISTORY_DIR = (BENCH_DIR / "history").resolve()
ERROR_LOG = (BENCH_DIR / "matrix_errors.log").resolve()
RUN_LOG = (BENCH_DIR / "matrix_run.log").resolve()

# Tee logger class to automatically redirect output to console and file
class TeeLogger:
    def __init__(self, original_stream, log_file_or_path, mode="a"):
        self.original_stream = original_stream
        if isinstance(log_file_or_path, (str, Path)):
            self.log_file = open(log_file_or_path, mode, encoding="utf-8")
            self._owns_file = True
        else:
            self.log_file = log_file_or_path
            self._owns_file = False
        
    def write(self, message):
        self.original_stream.write(message)
        if hasattr(self, "log_file") and self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()
        
    def flush(self):
        self.original_stream.flush()
        if hasattr(self, "log_file") and self.log_file and not self.log_file.closed:
            self.log_file.flush()

    def close(self):
        if hasattr(self, "log_file") and self.log_file and not self.log_file.closed:
            if getattr(self, "_owns_file", True):
                self.log_file.close()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()

    @property
    def encoding(self):
        return getattr(self.original_stream, "encoding", "utf-8")

HOME_DIR = os.environ.get("HOME", os.path.expanduser("~"))
DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", os.path.join(HOME_DIR, ".cache", "huggingface"))

def resolve_latest_snapshot(cache_dir, repo_folder, gguf_filename, fallback_hash):
    snapshots_dir = Path(cache_dir) / "hub" / repo_folder / "snapshots"
    if snapshots_dir.exists() and snapshots_dir.is_dir():
        subdirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if subdirs:
            latest = max(subdirs, key=lambda d: d.stat().st_mtime)
            return str(latest / gguf_filename)
    return str(Path(cache_dir) / "hub" / repo_folder / "snapshots" / fallback_hash / gguf_filename)

def get_default_gguf_paths(cache_dir=None):
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    return {
        "Qwen3.6-27B": os.environ.get("GGUF_PATH_QWEN27B", resolve_latest_snapshot(cache_dir, "models--unsloth--Qwen3.6-27B-GGUF", "Qwen3.6-27B-Q4_K_S.gguf", "82d411acf4a06cfb8d9b073a5211bf410bfc29bf")),
        "Qwen3.6-27B-spec3": os.environ.get("GGUF_PATH_QWEN27B_SPEC3", resolve_latest_snapshot(cache_dir, "models--unsloth--Qwen3.6-27B-MTP-GGUF", "Qwen3.6-27B-Q4_K_S.gguf", "b3a58239d8d40b953e34936c9afeb28baa518230")),
        "Qwen3.6-27B-spec4": os.environ.get("GGUF_PATH_QWEN27B_SPEC4", resolve_latest_snapshot(cache_dir, "models--unsloth--Qwen3.6-27B-MTP-GGUF", "Qwen3.6-27B-UD-Q4_K_XL.gguf", "b3a58239d8d40b953e34936c9afeb28baa518230")),
        "Qwen3.6-35B-A3B-spec": os.environ.get("GGUF_PATH_QWEN35B_SPEC", resolve_latest_snapshot(cache_dir, "models--unsloth--Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", "a483e9e6cbd595906af30beda3187c2663a1118c"))
    }

def restart_router_service(model="N/A", context=0):
    res = subprocess.run(["systemctl", "--user", "restart", "llama-router"], capture_output=True, text=True)
    if res.returncode != 0:
        err_msg = f"systemctl restart llama-router failed with exit code {res.returncode}"
        print(f"    [!] Warning: {err_msg}")
        if res.stderr:
            print(f"        STDERR: {res.stderr.strip()}")
        log_error(model, context, err_msg, stdout=res.stdout, stderr=res.stderr)
    return res.returncode == 0

def wait_for_endpoint_health(endpoint="http://127.0.0.1:8081", timeout=60, poll_interval=2):
    import urllib.request
    url = f"{endpoint.rstrip("/")}/v1/models"
    start_time = time.time()
    print(f"    [*] Polling health of {endpoint}...")
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    print(f"    [+] Server endpoint {endpoint} is ready!")
                    return True
        except Exception:
            pass
        time.sleep(poll_interval)
    print(f"    [-] Server endpoint {endpoint} did not respond within {timeout}s.")
    return False

def log_error(model, context, error_msg, stdout="", stderr=""):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TIMESTAMP: {datetime.datetime.now().isoformat()}\n")
        f.write(f"RUN CONFIGURATION: Model={model} | Context={context}\n")
        f.write(f"ERROR: {error_msg}\n")
        if stdout:
            f.write(f"STDOUT:\n{stdout}\n")
        if stderr:
            f.write(f"STDERR:\n{stderr}\n")
        f.write("=" * 80 + "\n\n")

def get_completed_runs(presets_file=None):
    completed = set()
    if not HISTORY_DIR.exists():
        return completed
    
    # Read model_presets to support mapping if needed
    if presets_file is None:
        presets_file = os.environ.get("PRESETS_FILE", os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini")))
    presets_sections = set()
    if os.path.exists(presets_file):
        try:
            import configparser
            config = configparser.ConfigParser(strict=False)
            config.read(presets_file)
            presets_sections = set(config.sections())
        except Exception:
            pass

    for filepath in HISTORY_DIR.glob("run_*.json"):
        # Filter out individual KLD files (which end in quantization formats, e.g. _f16.json)
        parts = filepath.stem.split("_")
        if len(parts) > 1 and parts[-1] in {"f16", "q8_0", "q5_1", "q4_0"}:
            continue
            
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            settings = data.get("model_settings", {})
            
            # Find profile from CLI arguments first as the source of truth
            metadata = data.get("run_metadata", {})
            args = metadata.get("cli_arguments", [])
            profile = None
            ctx = 0
            for i, arg in enumerate(args):
                if arg == "--model" and i + 1 < len(args):
                    profile = args[i+1]
                elif arg == "--tokens" and i + 1 < len(args):
                    try:
                        ctx = int(args[i+1])
                    except (ValueError, TypeError):
                        ctx = 0
                    
            # Fallback to settings profile_alias or model_name if CLI args not parsed
            if not profile:
                profile = settings.get("profile_alias")
            if not profile:
                profile = settings.get("model_name", "")
                
            # If the profile name contains unsloth repo prefix, try mapping it
            if profile and ("/" in profile or ":" in profile):
                matched = False
                for section in presets_sections:
                    if section.lower() == profile.lower():
                        profile = section
                        matched = True
                        break
                if not matched:
                    for section in presets_sections:
                        if section.lower() in profile.lower() or profile.lower() in section.lower():
                            profile = section
                            break
                    
            if profile and ctx:
                completed.add((profile, ctx))
        except Exception:
            pass
            
    return completed

def run_matrix(endpoint="http://127.0.0.1:8081", presets_file=None, cache_dir=None):
    # Make sure history directory exists
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not (HISTORY_DIR / ".gitkeep").exists():
        with open(HISTORY_DIR / ".gitkeep", "w") as f:
            f.write("")
            
    # Get previously completed runs for resuming
    completed_runs = get_completed_runs(presets_file=presets_file)
    gguf_paths = get_default_gguf_paths(cache_dir=cache_dir)
    
    models = ["Qwen3.6-27B", "Qwen3.6-27B-spec3", "Qwen3.6-27B-spec4", "Qwen3.6-35B-A3B-spec"]
    contexts = [1024, 8192, 32000, 64000, 128000, 228000]
    
    total_runs = len(models) * len(contexts)
    count = 0
    failed_models = set()
    
    print("\n" + "=" * 80)
    print(f" LLM MATRIX RUNNER - START TIME: {datetime.datetime.now().isoformat()}")
    print("=" * 80)
    print(f"[+] Loaded {len(completed_runs)} already completed runs from history/.")
    print(f"[+] Output is automatically copied to: {RUN_LOG}")
    print(f"[+] Failures/Errors are logged to: {ERROR_LOG}\n")
    
    for m in models:
        for c in contexts:
            count += 1
            
            # Check if this model has failed previously in this matrix loop
            if m in failed_models:
                print(f"[{count}/{total_runs}] Skipping (Model Broken/Missing): Model={m} | Context={c}")
                continue
            
            # Check if this run is already completed
            if (m, c) in completed_runs:
                print(f"[{count}/{total_runs}] Skipping (Already Completed): Model={m} | Context={c}")
                continue
                
            print(f"[{count}/{total_runs}] Launching Real Run: Model={m} | Context={c} tokens...")
            
            # Check endpoint health before starting benchmark
            if not wait_for_endpoint_health(endpoint=endpoint, timeout=15, poll_interval=2):
                err_msg = f"Endpoint {endpoint} is unhealthy before starting benchmark for model {m}"
                print(f"    [-] Unhealthy Endpoint: {err_msg}. Skipping model {m}...")
                log_error(m, c, err_msg)
                failed_models.add(m)
                continue
            
            cmd = [
                sys.executable, "run_suite.py",
                "--mode", "all",
                "--model", m,
                "--tokens", str(c),
                "--endpoint", endpoint
            ]
            if m in gguf_paths and os.path.exists(gguf_paths[m]):
                cmd.extend(["--gguf-path", gguf_paths[m]])
                
            # Scale timeout based on context size (larger contexts need more time for multiple needle/ruler runs)
            if c <= 8192:
                run_timeout = 300    # 5 minutes
            elif c <= 32000:
                run_timeout = 600    # 10 minutes
            elif c <= 64000:
                run_timeout = 1000   # ~16 minutes
            else:
                run_timeout = 2400   # 40 minutes
                
            try:
                # Limit each run dynamically to prevent blocking forever on real hangs
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=run_timeout)
                if result.returncode == 0:
                    print("    [+] Success! Run saved in history/.")
                    completed_runs.add((m, c))
                else:
                    err_msg = f"run_suite.py returned exit code {result.returncode}"
                    print(f"    [-] Failed: {err_msg}. Logging error, marking model as broken, and restarting llama-router...")
                    log_error(m, c, err_msg, result.stdout, result.stderr)
                    failed_models.add(m)  # Mark model as failed to skip subsequent context sizes
                    
                    # Restart server router service to clear locks
                    restart_router_service(m, c)
                    wait_for_endpoint_health(endpoint=endpoint)
            except subprocess.TimeoutExpired as e:
                err_msg = f"Subprocess timed out (exceeded {run_timeout} seconds limit)"
                print(f"    [-] Timeout: {err_msg}. Marking model as broken, and restarting llama-router...")
                log_error(m, c, err_msg, str(e.stdout or ""), str(e.stderr or ""))
                failed_models.add(m)  # Mark model as failed to skip subsequent context sizes
                
                # Restart server router service to clear locks
                subprocess.run(["systemctl", "--user", "restart", "llama-router"])
                wait_for_endpoint_health(endpoint=endpoint)
            except Exception as e:
                err_msg = f"Subprocess exception: {e}"
                print(f"    [-] Execution error: {err_msg}. Marking model as broken, and restarting llama-router...")
                log_error(m, c, err_msg)
                failed_models.add(m)
                
                # Restart server router service to clear locks
                subprocess.run(["systemctl", "--user", "restart", "llama-router"])
                wait_for_endpoint_health(endpoint=endpoint)
                
    print("\n" + "=" * 80)
    print(f" LLM MATRIX RUNNER COMPLETE - END TIME: {datetime.datetime.now().isoformat()}")
    print(f" Successfully evaluated: {len(completed_runs)} | Total matrix: {total_runs} runs.")
    print("=" * 80 + "\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Matrix Runner")
    parser.add_argument("--endpoint", default=os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8081"), help="LLM server API endpoint")
    parser.add_argument("--presets-file", default=os.environ.get("PRESETS_FILE", None), help="Path to model_presets.ini")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", None), help="Cache directory for HuggingFace models")
    args = parser.parse_args()
    
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(RUN_LOG, "a", encoding="utf-8")
    stdout_logger = TeeLogger(sys.stdout, log_file)
    stderr_logger = TeeLogger(sys.stderr, log_file)
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    def _cleanup_loggers():
        if not log_file.closed:
            log_file.close()

    atexit.register(_cleanup_loggers)

    run_matrix(endpoint=args.endpoint, presets_file=args.presets_file, cache_dir=args.cache_dir)

if __name__ == "__main__":
    main()
