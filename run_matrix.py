#!/usr/bin/env python3
import os
import sys
import json
import datetime
import subprocess
import time
from pathlib import Path

BENCH_DIR = Path(__file__).parent.resolve()
HISTORY_DIR = BENCH_DIR / "history"
ERROR_LOG = BENCH_DIR / "matrix_errors.log"
RUN_LOG = BENCH_DIR / "matrix_run.log"

# Tee logger class to automatically redirect output to console and file
class TeeLogger:
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log_file = open(filename, mode, encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Setup Tee Logging
sys.stdout = TeeLogger(RUN_LOG, "a")
sys.stderr = sys.stdout  # Redirect stderr to the same logger

# Local GGUF model paths for perplexity testing
gguf_paths = {
    "Qwen3.6-27B": "/home/sheepdestroyer/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-GGUF/snapshots/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q4_K_S.gguf",
    "Qwen3.6-27B-spec3": "/home/sheepdestroyer/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/b3a58239d8d40b953e34936c9afeb28baa518230/Qwen3.6-27B-Q4_K_S.gguf",
    "Qwen3.6-27B-spec4": "/home/sheepdestroyer/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/b3a58239d8d40b953e34936c9afeb28baa518230/Qwen3.6-27B-UD-Q4_K_XL.gguf",
    "Qwen3.6-35B-A3B-spec": "/home/sheepdestroyer/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
}

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

def get_completed_runs():
    completed = set()
    if not HISTORY_DIR.exists():
        return completed
    
    # Read model_presets to support mapping if needed
    presets_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini"))
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
        if len(parts) > 1 and parts[-1] in ["f16", "q8_0", "q5_1", "q4_0"]:
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
                    ctx = int(args[i+1])
                    
            # Fallback to settings profile_alias or model_name if CLI args not parsed
            if not profile:
                profile = settings.get("profile_alias")
            if not profile:
                profile = settings.get("model_name", "")
                
            # If the profile name contains unsloth repo prefix, try mapping it
            if profile and ("/" in profile or ":" in profile):
                for section in presets_sections:
                    if section.lower() in profile.lower() or profile.lower() in section.lower():
                        profile = section
                        break
                    
            if profile and ctx:
                completed.add((profile, ctx))
        except Exception:
            pass
            
    return completed

def run_matrix():
    # Make sure history directory exists
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not (HISTORY_DIR / ".gitkeep").exists():
        with open(HISTORY_DIR / ".gitkeep", "w") as f:
            f.write("")
            
    # Get previously completed runs for resuming
    completed_runs = get_completed_runs()
    
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
            
            cmd = [
                sys.executable, "run_suite.py",
                "--mode", "all",
                "--model", m,
                "--tokens", str(c)
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
                    print(f"    [+] Success! Run saved in history/.")
                    completed_runs.add((m, c))
                else:
                    err_msg = f"run_suite.py returned exit code {result.returncode}"
                    print(f"    [-] Failed: {err_msg}. Logging error, marking model as broken, and restarting llama-router...")
                    log_error(m, c, err_msg, result.stdout, result.stderr)
                    failed_models.add(m)  # Mark model as failed to skip subsequent context sizes
                    
                    # Restart server router service to clear locks
                    subprocess.run(["systemctl", "--user", "restart", "llama-router"])
                    time.sleep(20)
            except subprocess.TimeoutExpired as e:
                err_msg = f"Subprocess timed out (exceeded {run_timeout} seconds limit)"
                print(f"    [-] Timeout: {err_msg}. Marking model as broken, and restarting llama-router...")
                log_error(m, c, err_msg, str(e.stdout or ""), str(e.stderr or ""))
                failed_models.add(m)  # Mark model as failed to skip subsequent context sizes
                
                # Restart server router service to clear locks
                subprocess.run(["systemctl", "--user", "restart", "llama-router"])
                time.sleep(20)
            except Exception as e:
                err_msg = f"Subprocess exception: {e}"
                print(f"    [-] Execution error: {err_msg}. Marking model as broken, and restarting llama-router...")
                log_error(m, c, err_msg)
                failed_models.add(m)
                
                # Restart server router service to clear locks
                subprocess.run(["systemctl", "--user", "restart", "llama-router"])
                time.sleep(20)
                
    print("\n" + "=" * 80)
    print(f" LLM MATRIX RUNNER COMPLETE - END TIME: {datetime.datetime.now().isoformat()}")
    print(f" Successfully evaluated: {len(completed_runs)} | Total matrix: {total_runs} runs.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_matrix()
