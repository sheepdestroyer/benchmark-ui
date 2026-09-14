#!/usr/bin/env python3

# Tuple of (lowercase_quant, original_quant)
QUANT_TYPES = (
    ("q4_k_s", "Q4_K_S"),
    ("q4_k_m", "Q4_K_M"),
    ("q4_k_l", "Q4_K_L"),
    ("q4_k_xl", "Q4_K_XL"),
    ("q5_k_s", "Q5_K_S"),
    ("q5_k_m", "Q5_K_M"),
    ("q8_0", "Q8_0"),
    ("f16", "f16"),
)

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

from utils import CONTEXT_TIERS as CONTEXT_TIERS, parse_context_tokens, redact_cli_args

# Import advanced_benchmarks functions if possible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import advanced_benchmarks
except ImportError:  # pragma: no cover
    advanced_benchmarks = None

try:
    import agentic_benchmarks
except ImportError:  # pragma: no cover
    agentic_benchmarks = None


QUANT_PRIORITIES = ("q5_1", "q8_0", "q4_0", "f16")


def get_model_settings(endpoint, target_model, api_key=None):
    settings = {
        "model_name": target_model,
        "base_quantization": "Unknown",
        "kv_cache_quant": "Unknown",
        "kv_cache_quant_k": "Unknown",
        "kv_cache_quant_v": "Unknown",
        "threads": None,
        "ubatch_size": None,
        "batch_size": None,
        "speculative_draft_type": "None",
    }

    # Try parsing base quantization from target_model name if it's there
    target_model_lower = target_model.lower()
    for q_lower, q_orig in QUANT_TYPES:
        if q_lower in target_model_lower:
            settings["base_quantization"] = q_orig
            break

    try:
        url = f"{endpoint}/v1/models"
        effective_key = (
            api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        )
        headers = {"Authorization": f"Bearer {effective_key}"} if effective_key else {}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            model_info = None
            for item in data.get("data", []):
                if item.get("id") == target_model:
                    model_info = item
                    break

            if model_info:
                status = model_info.get("status", {})
                args = status.get("args", [])
                preset = status.get("preset", "")

                # Check args
                for i, arg in enumerate(args):
                    if arg == "--threads" and i + 1 < len(args):
                        try:
                            settings["threads"] = int(args[i + 1])
                        except (ValueError, TypeError):
                            settings["threads"] = None
                    elif arg == "--batch-size" and i + 1 < len(args):
                        try:
                            settings["batch_size"] = int(args[i + 1])
                        except (ValueError, TypeError):
                            settings["batch_size"] = None
                    elif arg == "--ubatch-size" and i + 1 < len(args):
                        try:
                            settings["ubatch_size"] = int(args[i + 1])
                        except (ValueError, TypeError):
                            settings["ubatch_size"] = None
                    elif arg == "--cache-type-k" and i + 1 < len(args):
                        settings["kv_cache_quant_k"] = args[i + 1]
                        settings["kv_cache_quant"] = args[i + 1]
                    elif arg == "--cache-type-v" and i + 1 < len(args):
                        settings["kv_cache_quant_v"] = args[i + 1]
                        if settings.get("kv_cache_quant") == "Unknown":
                            settings["kv_cache_quant"] = args[i + 1]

                # Try preset parsing
                if preset:
                    for line in preset.split("\n"):
                        line = line.strip()
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip()
                            if k == "threads":
                                try:
                                    settings["threads"] = int(v)
                                except (ValueError, TypeError):
                                    settings["threads"] = None
                            elif k == "batch-size":
                                try:
                                    settings["batch_size"] = int(v)
                                except (ValueError, TypeError):
                                    settings["batch_size"] = None
                            elif k == "ubatch-size":
                                try:
                                    settings["ubatch_size"] = int(v)
                                except (ValueError, TypeError):
                                    settings["ubatch_size"] = None
                            elif k == "cache-type-k":
                                settings["kv_cache_quant_k"] = v
                                settings["kv_cache_quant"] = v
                            elif k == "cache-type-v":
                                settings["kv_cache_quant_v"] = v
                                if settings.get("kv_cache_quant") == "Unknown":
                                    settings["kv_cache_quant"] = v

                repo_or_id = model_info.get("id", "")
                repo_or_id_lower = repo_or_id.lower()
                for q_lower, q_orig in QUANT_TYPES:
                    if q_lower in repo_or_id_lower:
                        settings["base_quantization"] = q_orig
                        break

                if (
                    "mtp" in repo_or_id_lower
                    or any(
                        term in str(arg).lower()
                        for arg in args
                        for term in ["spec", "draft", "ngram", "lookup"]
                    )
                    or any(
                        term in preset.lower()
                        for term in ["spec", "draft", "ngram", "mtp"]
                    )
                ):
                    settings["speculative_draft_type"] = "ngram"
                else:
                    settings["speculative_draft_type"] = "None"
    except (requests.RequestException, ConnectionError, ValueError) as e:
        print(f"[*] Could not fetch model settings from endpoint: {e}")

    if advanced_benchmarks:
        profile_name = advanced_benchmarks.map_repo_to_preset_alias(target_model)
        presets_meta = advanced_benchmarks.get_preset_metadata(profile_name)
        for k, v in presets_meta.items():
            if k in ("threads", "batch_size", "ubatch_size"):
                continue
            settings[k] = v
        settings["profile_alias"] = profile_name

    return settings


def run_throughput(endpoint, model, api_key=None):
    print("\n=========================================================")
    print(" Running Throughput Benchmarks (benchmark.sh)")
    print("=========================================================")
    bench_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmark.sh"
    )

    # Ensure executable
    if os.path.exists(bench_script) and not os.access(bench_script, os.X_OK):
        os.chmod(bench_script, 0o755)

    cmd = [bench_script, model, endpoint]
    env = os.environ.copy()
    effective_key = (
        api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    )
    if effective_key:
        env["OPENAI_API_KEY"] = effective_key
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
    )
    print(result.stdout)
    if result.stderr:
        print("Error/Stderr:", result.stderr)

    if result.returncode != 0:
        print(f"[-] benchmark.sh failed with exit code {result.returncode}")
        return {"prefill_speed": 0.0, "decode_speed": 0.0, "ttft": 0.0}

    # Parse turns
    p_evals = []
    gens = []
    ttfts = []

    peval_re = re.compile(
        r"Prompt Eval \(p/s\)\s*:\s*([0-9.]+)\s*tokens/sec\s*\(TTFT:\s*([0-9.]+)s\)"
    )
    gen_re = re.compile(r"Generation\s*\(t/s\)\s*:\s*([0-9.]+)\s*tokens/sec")

    for line in result.stdout.split("\n"):
        peval_match = peval_re.search(line)
        if peval_match:
            p_evals.append(float(peval_match.group(1)))
            ttfts.append(float(peval_match.group(2)))
        gen_match = gen_re.search(line)
        if gen_match:
            gens.append(float(gen_match.group(1)))

    return {
        "prefill_speed": sum(p_evals) / len(p_evals) if p_evals else 0.0,
        "decode_speed": sum(gens) / len(gens) if gens else 0.0,
        "ttft": sum(ttfts) / len(ttfts) if ttfts else 0.0,
    }


def run_reasoning(endpoint, model, tokens, api_key=None, max_tokens=16384):
    print("\n=========================================================")
    print(" Running Reasoning Benchmarks (advanced_benchmarks.py)")
    print("=========================================================")

    results = {}

    if advanced_benchmarks:
        tokens = parse_context_tokens(tokens)
        call_kw = {}
        effective_key = (
            api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        )
        if effective_key:
            call_kw["api_key"] = effective_key
        print("[*] Executing Needle test...")
        try:
            res = advanced_benchmarks.run_needle_test(
                endpoint, model, tokens=tokens, max_tokens=max_tokens, **call_kw
            )
            results["needle"] = "Pass" if res and res.get("passed") else "Fail"
        except Exception as e:  # noqa: BLE001
            print(f"Needle failed: {e}")
            results["needle"] = "Fail"

        print("[*] Executing RULER test...")
        try:
            res = advanced_benchmarks.run_ruler_test(
                endpoint, model, tokens=tokens, max_tokens=max_tokens, **call_kw
            )
            results["ruler"] = "Pass" if res and res.get("passed") else "Fail"
        except Exception as e:  # noqa: BLE001
            print(f"RULER failed: {e}")
            results["ruler"] = "Fail"

        print("[*] Executing LongBench test...")
        try:
            res = advanced_benchmarks.run_longbench_test(
                endpoint, model, tokens=tokens, max_tokens=max_tokens, **call_kw
            )
            results["longbench"] = "Pass" if res and res.get("passed") else "Fail"
        except Exception as e:  # noqa: BLE001
            print(f"LongBench failed: {e}")
            results["longbench"] = "Fail"

        print("[*] Executing SWE-bench test...")
        try:
            res = advanced_benchmarks.run_swe_test(
                endpoint, model, max_tokens=max_tokens, **call_kw
            )
            results["swe_bench"] = "Pass" if res and res.get("passed") else "Fail"
        except Exception as e:  # noqa: BLE001
            print(f"SWE-bench failed: {e}")
            results["swe_bench"] = "Fail"
    else:
        print("[-] advanced_benchmarks module not found.")
        results = {
            "needle": "Fail",
            "ruler": "Fail",
            "longbench": "Fail",
            "swe_bench": "Fail",
        }

    return results


def run_agentic(endpoint, model, task_filter="all", api_key=None, max_tokens=16384):
    print("\n=========================================================")
    print(" Running Agentic Benchmarks (agentic_benchmarks.py)")
    print("=========================================================")

    if agentic_benchmarks:
        call_kw = {}
        effective_key = (
            api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        )
        if effective_key:
            call_kw["api_key"] = effective_key
        try:
            return agentic_benchmarks.run_agentic_suite(
                endpoint,
                model,
                task_filter=task_filter,
                max_tokens=max_tokens,
                **call_kw,
            )
        except Exception as e:
            print(f"Agentic suite failed: {e}")
            return {
                "suite": "standalone-agentic",
                "tasks_total": 0,
                "tasks_passed": 0,
                "average_turns": 0.0,
                "total_tool_calls": 0,
                "tasks": [],
                "token_breakdown": {
                    "prompt_tokens": 0,
                    "reasoning_tokens": 0,
                    "completion_tokens": 0,
                },
            }
    else:
        print("[-] agentic_benchmarks module not found.")
        return {
            "suite": "standalone-agentic",
            "tasks_total": 0,
            "tasks_passed": 0,
            "average_turns": 0.0,
            "total_tool_calls": 0,
            "tasks": [],
            "token_breakdown": {
                "prompt_tokens": 0,
                "reasoning_tokens": 0,
                "completion_tokens": 0,
            },
        }


def run_kld(model_path, corpus):
    print("\n=========================================================")
    print(" Running KLD Benchmarks (kld_benchmark.py)")
    print("=========================================================")

    cmd = [sys.executable, "kld_benchmark.py"]
    if model_path:
        cmd.extend(["--model", model_path])
    if corpus:
        cmd.extend(["--corpus", corpus])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    print(result.stdout)
    if result.stderr:
        print("Error/Stderr:", result.stderr)

    kld_results = {}
    lines = result.stdout.split("\n")
    for line in lines:
        if (
            "|" in line
            and "Perplexity" not in line
            and "KV Cache" not in line
            and "===" not in line
            and "---" not in line
        ):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 4:
                quant_name = parts[0].replace(" (Baseline)", "")
                try:
                    ppl = float(parts[1]) if parts[1] != "N/A" else None
                    kld = float(parts[2]) if parts[2] != "N/A" else None
                    same_top = (
                        float(parts[3].replace("%", "")) if parts[3] != "N/A" else None
                    )
                    kld_results[quant_name] = {
                        "perplexity": ppl,
                        "mean_kld": kld,
                        "same_top_match_percent": same_top,
                    }
                except ValueError:
                    pass
    return kld_results


def main():
    parser = argparse.ArgumentParser(description="Unified LLM Benchmarking Suite")
    parser.add_argument(
        "--mode",
        choices=["throughput", "reasoning", "agentic", "kld", "all"],
        default="all",
        help="Benchmark mode to run",
    )
    parser.add_argument(
        "--agentic-tasks",
        default="all",
        help="Task filter for agentic benchmark suite (default: 'all')",
    )
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8083", help="LLM server API endpoint"
    )
    parser.add_argument(
        "--model", default="Qwen3.6-27B", help="Model name / alias on the server"
    )
    parser.add_argument(
        "--tokens",
        type=parse_context_tokens,
        default=200000,
        help="Target context token length for reasoning benchmarks (or tier: '8k', '32k', '64k', '128k', '240k')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum completion tokens to generate",
    )
    parser.add_argument(
        "--gguf-path", help="Local path to the GGUF model file (for KLD benchmark)"
    )
    parser.add_argument(
        "--corpus",
        default="kld_corpus.txt",
        help="Path to text corpus for KLD perplexity calculation",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        help="API key for Bearer authentication",
    )

    args = parser.parse_args()

    timestamp = datetime.datetime.now().isoformat()

    throughput_metrics = {"prefill_speed": None, "decode_speed": None, "ttft": None}

    reasoning_accuracy = {
        "needle": "N/A",
        "ruler": "N/A",
        "longbench": "N/A",
        "swe_bench": "N/A",
    }

    agentic_metrics = None
    token_breakdown = None

    quantization_loss = {
        "perplexity": None,
        "mean_kld": None,
        "same_top_match_percent": None,
    }

    call_kw = {}
    if getattr(args, "api_key", None):
        call_kw["api_key"] = args.api_key

    # 1. Run throughput if mode is 'throughput' or 'all'
    if args.mode in ["throughput", "all"]:
        metrics = run_throughput(args.endpoint, args.model, **call_kw)
        throughput_metrics.update(metrics)

    # 2. Run reasoning if mode is 'reasoning' or 'all'
    if args.mode in ["reasoning", "all"]:
        accuracy = run_reasoning(
            args.endpoint,
            args.model,
            args.tokens,
            max_tokens=args.max_tokens,
            **call_kw,
        )
        reasoning_accuracy.update(accuracy)

    # 3. Run agentic if mode is 'agentic' or 'all'
    if args.mode in ["agentic", "all"]:
        agentic_results = run_agentic(
            args.endpoint,
            args.model,
            task_filter=args.agentic_tasks,
            max_tokens=args.max_tokens,
            **call_kw,
        )
        agentic_metrics = agentic_results
        token_breakdown = (
            agentic_results.get("token_breakdown") if agentic_results else None
        )

    # 4. Run KLD if mode is 'kld' or 'all'
    kld_results = None
    if args.mode in ["kld", "all"]:
        kld_results = run_kld(args.gguf_path, args.corpus)

    # 5. Extract model settings
    model_settings = get_model_settings(args.endpoint, args.model, **call_kw)

    # If KLD was run, choose the quantization loss values matching the model's KV Cache quant setting
    if kld_results:
        kv_quant = model_settings.get("kv_cache_quant", "q5_1")
        if not kv_quant or kv_quant == "Unknown":
            kv_quant = "q5_1"
        match_quant = kv_quant
        if match_quant not in kld_results:
            for k in QUANT_PRIORITIES:
                if k in kld_results:
                    match_quant = k
                    break
        if match_quant in kld_results:
            quantization_loss.update(kld_results[match_quant])
            if model_settings["kv_cache_quant"] == "Unknown":
                model_settings["kv_cache_quant"] = match_quant

    # Create final run object
    run_data = {
        "run_metadata": {
            "timestamp": timestamp,
            "target_endpoint": args.endpoint,
            "cli_arguments": redact_cli_args(sys.argv[1:]),
        },
        "model_settings": model_settings,
        "throughput_metrics": throughput_metrics,
        "reasoning_accuracy": reasoning_accuracy,
        "agentic_metrics": agentic_metrics,
        "token_breakdown": token_breakdown,
        "quantization_loss": quantization_loss,
    }

    # Write output to history/
    history_dir = (Path(__file__).parent / "history").resolve()
    history_dir.mkdir(parents=True, exist_ok=True)
    safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
    output_file = history_dir / f"run_{safe_timestamp}.json"
    tmp_file = output_file.with_suffix(".tmp")

    with open(tmp_file, "w") as f:
        json.dump(run_data, f, indent=4)
    os.replace(tmp_file, output_file)

    print(f"\n[+] Unified benchmark run completed and logged to {output_file}")


if __name__ == "__main__":
    main()
