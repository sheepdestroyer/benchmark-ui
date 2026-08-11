#!/usr/bin/env python3
import argparse
import time
import json
import random
import requests
import sys
import os
import subprocess
import shutil
import ast

# Module constants for quantization options and cache types
QUANTIZATION_OPTIONS = ("Q4_K_S", "Q4_K_M", "Q4_K_L", "Q4_K_XL", "Q5_K_S", "Q5_K_M", "Q8_0", "f16")
QUANTIZATIONS = tuple((q.lower(), q) for q in QUANTIZATION_OPTIONS)

CACHE_TYPE_CLI_ARGS = {"--cache-type-k", "--cache-type-v"}
CACHE_TYPE_KEYS = {"cache-type-k", "cache-type-v"}
SWE_BENCH_KEYS = {"swe-bench", "swe_bench"}
QUANTIZATION_OPTIONS = ("Q4_K_S", "Q4_K_M", "Q4_K_L", "Q4_K_XL", "Q5_K_S", "Q5_K_M", "Q8_0", "f16")
QUANTIZATION_OPTIONS_LOWER = tuple((q, q.lower()) for q in QUANTIZATION_OPTIONS)

def is_safe_code(code_str):
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    dangerous_names = {'os.system', 'shutil.rmtree', 'eval', 'exec', 'subprocess', 'socket'}
    dangerous_modules = {'subprocess', 'socket'}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split('.')[0]
                if mod in dangerous_modules or alias.name in dangerous_names:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or '').split('.')[0]
            if mod in dangerous_modules:
                return False, f"Forbidden import module: {node.module}"
            for alias in node.names:
                full_import = f"{node.module}.{alias.name}"
                if full_import in dangerous_names or alias.name in ('eval', 'exec', 'subprocess', 'socket'):
                    return False, f"Forbidden import: {full_import}"
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                full_attr = f"{node.value.id}.{node.attr}"
                if full_attr in dangerous_names:
                    return False, f"Forbidden attribute usage: {full_attr}"
        elif isinstance(node, ast.Name):
            if node.id in ('eval', 'exec', 'subprocess', 'socket'):
                return False, f"Forbidden identifier usage: {node.id}"

    return True, None

# ==============================================================================
# COMMON UTILITIES & DATA GENERATION
# ==============================================================================

def generate_filler_text(target_tokens=200000):
    distractors = [
        "The software architecture patterns dictate that services must be decoupled.",
        "Quantum computing relies on superposition and entanglement to perform computations.",
        "A database transaction must satisfy the ACID properties to ensure reliability.",
        "Deep learning models require optimization algorithms like Adam or SGD to converge.",
        "The history of web browsers is characterized by intense competition and standardization.",
        "Distributed systems face challenges like network partitions, latency, and consensus protocols.",
        "Compiler design involves lexical analysis, parsing, semantic analysis, and code generation.",
        "Operating systems manage system resources, hardware devices, and process scheduling.",
        "Garbage collection algorithms reclaim memory occupied by objects that are no longer in use.",
        "Regular expressions are powerful tools for pattern matching and text manipulation."
    ]
    
    target_chars = target_tokens * 4.5
    current_chars = 0
    paragraphs = []
    
    while current_chars < target_chars:
        paragraph_sentences = [random.choice(distractors) for _ in range(5)]
        paragraph = " ".join(paragraph_sentences)
        paragraphs.append(paragraph)
        current_chars += len(paragraph) + 1
        
    return paragraphs

def call_endpoint(endpoint, model, prompt, max_tokens=512):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    url = f"{endpoint}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    first_token_time = None
    response_text = ""
    reasoning_text = ""
    usage = None
    
    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_content = line_str[6:]
                    if data_content.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_content)
                        if chunk.get("choices"):
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            reasoning_content = delta.get("reasoning_content", "")
                            
                            if first_token_time is None and (content or reasoning_content):
                                first_token_time = time.time()
                            
                            if content:
                                response_text += content
                            if reasoning_content:
                                reasoning_text += reasoning_content
                            
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"Warning: Unexpected error parsing SSE chunk: {e}")
    except Exception as e:
        print(f"Request failed: {e}")
        return None
        
    end_time = time.time()
    if first_token_time is None:
        first_token_time = end_time
        
    ttft = first_token_time - start_time
    total_time = end_time - start_time
    decode_time = end_time - first_token_time
    
    prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
    completion_tokens = usage.get("completion_tokens", 0) if usage else 0
    
    prefill_speed = prompt_tokens / ttft if ttft > 0 else 0
    decode_speed = completion_tokens / decode_time if decode_time > 0 else 0
    
    return {
        "response": response_text,
        "reasoning": reasoning_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft": ttft,
        "decode_time": decode_time,
        "prefill_speed": prefill_speed,
        "decode_speed": decode_speed,
        "total_time": total_time
    }

# ==============================================================================
# BENCHMARK 1: NEEDLE IN A HAYSTACK
# ==============================================================================

def run_needle_test(endpoint, model, tokens=200000, depth=0.5):
    print(f"\n=== Running Needle in a Haystack Test (~{tokens} tokens) ===")
    
    paragraphs = generate_filler_text(tokens)
    needle = "The secret code word is: BANANA_SPLIT. Remember this code word, as it is required to pass the test."
    query = "What is the secret code word? Provide only the code word and nothing else."
    
    try:
        insert_index = int(len(paragraphs) * depth)
    except (ValueError, TypeError):
        insert_index = 0
    paragraphs.insert(insert_index, needle)
    
    full_context = "\n\n".join(paragraphs)
    prompt = f"Context:\n{full_context}\n\nQuestion: {query}"
    
    res = call_endpoint(endpoint, model, prompt, max_tokens=256)
    if not res:
        return None
        
    is_correct = bool(re.search(r"\bBANANA_SPLIT\b", res["response"])) or bool(re.search(r"\bBANANA_SPLIT\b", res["reasoning"]))
    
    print("\n---------------------------------------------------------")
    print(f"Needle Result      : {'PASSED' if is_correct else 'FAILED'}")
    if res["reasoning"]:
        print(f"Model Reasoning    : {res['reasoning'].strip()}")
    print(f"Model Answer       : {res['response'].strip()}")
    print(f"TTFT (Prefill Lat) : {res['ttft']:.2f}s (Speed: {res['prefill_speed']:.2f} t/s)")
    print(f"Decode Time        : {res['decode_time']:.2f}s (Speed: {res['decode_speed']:.2f} t/s)")
    print("---------------------------------------------------------")
    
    res["benchmark"] = "Needle"
    res["passed"] = is_correct
    return res

# ==============================================================================
# BENCHMARK 2: RULER (VARIABLE TRACKING CHAIN)
# ==============================================================================

def run_ruler_test(endpoint, model, tokens=200000):
    print(f"\n=== Running RULER Variable Tracking Test (~{tokens} tokens) ===")
    
    paragraphs = generate_filler_text(tokens)
    
    # We assign: var_a = 93 -> var_b = var_a -> var_c = var_b
    fact_1 = "The variable alpha is assigned the value 93."
    fact_2 = "The variable beta is assigned the value of variable alpha."
    fact_3 = "The variable gamma is assigned the value of variable beta."
    query = "What is the final value of variable gamma? Provide only the numerical value and nothing else."
    
    # Insert facts at 75%, 50%, and 25% depth
    p_len = len(paragraphs)
    try:
        idx75 = int(p_len * 0.75)
    except (ValueError, TypeError):
        idx75 = 0
    try:
        idx50 = int(p_len * 0.50)
    except (ValueError, TypeError):
        idx50 = 0
    try:
        idx25 = int(p_len * 0.25)
    except (ValueError, TypeError):
        idx25 = 0
    paragraphs.insert(idx75, fact_3)
    paragraphs.insert(idx50, fact_2)
    paragraphs.insert(idx25, fact_1)
    
    full_context = "\n\n".join(paragraphs)
    prompt = f"Context:\n{full_context}\n\nQuestion: {query}"
    
    res = call_endpoint(endpoint, model, prompt, max_tokens=256)
    if not res:
        return None
        
    is_correct = bool(re.search(r"\b93\b", res["response"])) or bool(re.search(r"\b93\b", res["reasoning"]))
    
    print("\n---------------------------------------------------------")
    print(f"RULER Result       : {'PASSED' if is_correct else 'FAILED'}")
    if res["reasoning"]:
        print(f"Model Reasoning    : {res['reasoning'].strip()}")
    print(f"Model Answer       : {res['response'].strip()}")
    print(f"TTFT (Prefill Lat) : {res['ttft']:.2f}s (Speed: {res['prefill_speed']:.2f} t/s)")
    print(f"Decode Time        : {res['decode_time']:.2f}s (Speed: {res['decode_speed']:.2f} t/s)")
    print("---------------------------------------------------------")
    
    res["benchmark"] = "RULER"
    res["passed"] = is_correct
    return res

# ==============================================================================
# BENCHMARK 3: LONGBENCH (DOCUMENT QA)
# ==============================================================================

def run_longbench_test(endpoint, model, tokens=200000):
    print(f"\n=== Running LongBench Document QA Test (~{tokens} tokens) ===")
    
    paragraphs = generate_filler_text(tokens)
    
    # Insert a target historical fact into the text
    fact = "In the year 1452, King Elidor signed the Treaty of Oakhaven, which ceded the northern hills to the dwarves."
    query = "In what year did King Elidor sign the Treaty of Oakhaven? Provide only the year and nothing else."
    
    # Insert at 50% depth
    try:
        lb_idx = int(len(paragraphs) * 0.5)
    except (ValueError, TypeError):
        lb_idx = 0
    paragraphs.insert(lb_idx, fact)
    
    full_context = "\n\n".join(paragraphs)
    prompt = f"Context:\n{full_context}\n\nQuestion: {query}"
    
    res = call_endpoint(endpoint, model, prompt, max_tokens=256)
    if not res:
        return None
        
    is_correct = bool(re.search(r"\b1452\b", res["response"])) or bool(re.search(r"\b1452\b", res["reasoning"]))
    
    print("\n---------------------------------------------------------")
    print(f"LongBench Result   : {'PASSED' if is_correct else 'FAILED'}")
    if res["reasoning"]:
        print(f"Model Reasoning    : {res['reasoning'].strip()}")
    print(f"Model Answer       : {res['response'].strip()}")
    print(f"TTFT (Prefill Lat) : {res['ttft']:.2f}s (Speed: {res['prefill_speed']:.2f} t/s)")
    print(f"Decode Time        : {res['decode_time']:.2f}s (Speed: {res['decode_speed']:.2f} t/s)")
    print("---------------------------------------------------------")
    
    res["benchmark"] = "LongBench"
    res["passed"] = is_correct
    return res

# ==============================================================================
# BENCHMARK 4: SWE-BENCH (TOY CODEBASE DEBUGGING)
# ==============================================================================

def run_swe_test(endpoint, model):
    print("\n=== Running SWE-bench Codebase Debugging Test ===")
    
    toy_repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_repo")
    code_path = os.path.join(toy_repo_dir, "calculator.py")
    test_path = os.path.join(toy_repo_dir, "test_calculator.py")
    
    if not os.path.exists(code_path) or not os.path.exists(test_path):
        print("Error: Toy repository files not found.")
        return None
        
    # Read files
    with open(code_path, "r", encoding="utf-8") as f:
        code_content = f.read()
    with open(test_path, "r", encoding="utf-8") as f:
        test_content = f.read()
        
    # Construct prompt
    prompt = f"""You are an automated software engineer. Fix the order-of-operations bug in the file calculator.py so that all tests pass.

Here is the code of calculator.py:
```python
{code_content}
```

Here is the test suite in test_calculator.py:
```python
{test_content}
```

Be extremely concise. Keep your internal thought trace minimal. Please output the COMPLETE corrected code of calculator.py inside a single python code block (wrapped in ```python ... ```). Do not output other text or conversational filler."""

    print("Sending codebase issue to LLM...")
    res = call_endpoint(endpoint, model, prompt, max_tokens=4096)
    if not res:
        return None
        
    # Parse code block from response
    new_code = ""
    raw_response = res["response"]
    if "```python" in raw_response:
        parts = raw_response.split("```python")
        if len(parts) > 1:
            new_code = parts[1].split("```")[0].strip()
            
    if not new_code and "```python" in res["reasoning"]:
        print("Parsing code block from reasoning trace fallback...")
        parts = res["reasoning"].split("```python")
        if len(parts) > 1:
            new_code = parts[1].split("```")[0].strip()
            
    if not new_code:
        print("Error: Could not parse python code block from response.")
        is_correct = False
    else:
        is_safe, reason = is_safe_code(new_code)
        if not is_safe:
            print(f"Error: Generated code failed security sandboxing check: {reason}")
            is_correct = False
        else:
            # Backup original file
            backup_path = code_path + ".bak"
            shutil.copy2(code_path, backup_path)
            
            try:
                # Write new code
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(new_code)
                    
                # Run unit tests
                test_run = subprocess.run(
                    [sys.executable, "-m", "unittest", "test_calculator.py"],
                    cwd=toy_repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                print(test_run.stdout)
                print(test_run.stderr)
                
                is_correct = (test_run.returncode == 0)
            except Exception as e:
                print(f"Failed to execute tests: {e}")
                is_correct = False
            finally:
                # Restore backup
                if 'backup_path' in locals() and os.path.exists(backup_path):
                    shutil.copy2(backup_path, code_path)
                    os.remove(backup_path)
            
    print("\n---------------------------------------------------------")
    print(f"SWE-bench Result   : {'PASSED' if is_correct else 'FAILED'}")
    if res["reasoning"]:
        print(f"Model Reasoning    : {res['reasoning'].strip()}")
    print(f"TTFT (Prefill Lat) : {res['ttft']:.2f}s (Speed: {res['prefill_speed']:.2f} t/s)")
    print(f"Decode Time        : {res['decode_time']:.2f}s (Speed: {res['decode_speed']:.2f} t/s)")
    print("---------------------------------------------------------")
    
    res["benchmark"] = "SWE-bench"
    res["passed"] = is_correct
    return res

# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def map_repo_to_preset_alias(repo_or_id):
    presets_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini"))
    if not os.path.exists(presets_file):
        return repo_or_id
        
    try:
        import configparser
        config = configparser.ConfigParser(strict=False)
        config.read(presets_file)
        
        for section in config.sections():
            if section.lower() == repo_or_id.lower():
                return section
                
        for section in config.sections():
            if section == "*":
                continue
            section_repo = config.get(section, "hf-repo", fallback="")
            section_alias = config.get(section, "alias", fallback="")
            
            if section_repo and section_repo.lower() == repo_or_id.lower():
                return section
                    
            if section_alias and section_alias.lower() == repo_or_id.lower():
                return section
    except Exception:
        pass
        
    return repo_or_id

def get_preset_metadata(profile_name):
    metadata = {
        "spec_type": "None",
        "spec_draft_type_k": "None",
        "spec_draft_type_v": "None",
        "flash_attn": "true",
        "parallel": "1",
        "n_gpu_layers": "99",
        "fit": "true"
    }
    
    presets_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini"))
    if os.path.exists(presets_file):
        try:
            import configparser
            config = configparser.ConfigParser(strict=False)
            config.read(presets_file)
            
            # Load globals if they exist
            if "*" in config.sections():
                for key in config["*"]:
                    clean_key = key.replace("-", "_")
                    metadata[clean_key] = config["*"][key]
                    
            # Load specific section
            if profile_name in config.sections():
                for key in config[profile_name]:
                    clean_key = key.replace("-", "_")
                    metadata[clean_key] = config[profile_name][key]
        except Exception:
            pass
            
    return metadata

def get_model_settings_from_endpoint(endpoint, target_model):
    settings = {
        "model_name": target_model,
        "base_quantization": "Unknown",
        "kv_cache_quant": "Unknown",
        "threads": None,
        "ubatch_size": None,
        "batch_size": None,
        "speculative_draft_type": "None"
    }
    
    # Try parsing base quantization from target_model name if it's there
    target_model_lower = target_model.lower()
    for q_lower, q in QUANTIZATIONS:
        if q_lower in target_model_lower:
            settings["base_quantization"] = q
            break
            
    try:
        url = f"{endpoint}/v1/models"
        with requests.get(url, timeout=5) as response:
            if response.status_code == 200:
                data = response.json()
                model_info = None
                for item in data.get("data", []):
                    if item.get("id") == target_model or target_model in item.get("id", ""):
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
                                settings["threads"] = int(args[i+1])
                            except (ValueError, TypeError):
                                settings["threads"] = None
                        elif arg == "--batch-size" and i + 1 < len(args):
                            try:
                                settings["batch_size"] = int(args[i+1])
                            except (ValueError, TypeError):
                                settings["batch_size"] = None
                        elif arg == "--ubatch-size" and i + 1 < len(args):
                            try:
                                settings["ubatch_size"] = int(args[i+1])
                            except (ValueError, TypeError):
                                settings["ubatch_size"] = None
                        elif arg in CACHE_TYPE_CLI_ARGS and i + 1 < len(args):
                            settings["kv_cache_quant"] = args[i+1]
                    
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
                                elif k in CACHE_TYPE_KEYS:
                                    settings["kv_cache_quant"] = v
                                    
                    repo_or_id = model_info.get("id", "")
                    repo_or_id_lower = repo_or_id.lower()
                    for q_lower, q in QUANTIZATIONS:
                    for q, q_lower in QUANTIZATION_OPTIONS_LOWER:
                        if q_lower in repo_or_id_lower:
                            settings["base_quantization"] = q
                            break
                    
                    if "mtp" in repo_or_id_lower or any("spec" in str(arg).lower() for arg in args):
                        settings["speculative_draft_type"] = "ngram"
                    else:
                        settings["speculative_draft_type"] = "None"
    except Exception as e:
        print(f"[*] Could not fetch model settings from endpoint: {e}")
        
    # Merge preset metadata fields
    profile_name = map_repo_to_preset_alias(target_model)
    presets_meta = get_preset_metadata(profile_name)
    for k, v in presets_meta.items():
        settings.setdefault(k, v)
    settings["profile_alias"] = profile_name
        
    return settings

def main():
    parser = argparse.ArgumentParser(description="Advanced Benchmarks Runner")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081", help="LLM server API endpoint")
    parser.add_argument("--model", default="Qwen3.6-27B", help="Model name / alias to target")
    parser.add_argument("--tokens", type=int, default=200000, help="Number of context tokens for synthetic benchmarks (Needle, RULER, LongBench)")
    parser.add_argument("--needle", action="store_true", help="Run Needle in a Haystack benchmark (Phase 1)")
    parser.add_argument("--ruler", action="store_true", help="Run RULER variable tracking benchmark (Phase 2)")
    parser.add_argument("--longbench", action="store_true", help="Run LongBench QA benchmark (Phase 3)")
    parser.add_argument("--swe", action="store_true", help="Run SWE-bench toy repository debugging benchmark (Phase 4)")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks sequentially")
    
    args = parser.parse_args()
    
    # If no flags are set, default to running all of them
    run_all = args.all
    if not (args.needle or args.ruler or args.longbench or args.swe or args.all):
        run_all = True
        
    results = []
    
    if args.needle or run_all:
        res = run_needle_test(args.endpoint, args.model, tokens=args.tokens)
        if res:
            results.append(res)
            
    if args.ruler or run_all:
        res = run_ruler_test(args.endpoint, args.model, tokens=args.tokens)
        if res:
            results.append(res)
            
    if args.longbench or run_all:
        res = run_longbench_test(args.endpoint, args.model, tokens=args.tokens)
        if res:
            results.append(res)
            
    if args.swe or run_all:
        res = run_swe_test(args.endpoint, args.model)
        if res:
            results.append(res)

    if results:
        print("\n=== Advanced Benchmark Suite Summary ===")
        print("--------------------------------------------------------------------------------")
        print(f"{'Benchmark':<12} | {'Status':<6} | {'Prompt tks':<10} | {'TTFT':<6} | {'Prefill t/s':<12} | {'Decode t/s':<10}")
        print("--------------------------------------------------------------------------------")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"{r['benchmark']:<12} | {status:<6} | {r['prompt_tokens']:<10} | {r['ttft']:>5.2f}s | {r['prefill_speed']:>10.2f}  | {r['decode_speed']:>8.2f}")
        print("--------------------------------------------------------------------------------")

        import datetime
        timestamp = datetime.datetime.now().isoformat()
        model_settings = get_model_settings_from_endpoint(args.endpoint, args.model)
        
        valid_prefill = [r["prefill_speed"] for r in results if r.get("prefill_speed", 0) > 0]
        valid_decode = [r["decode_speed"] for r in results if r.get("decode_speed", 0) > 0]
        valid_ttft = [r["ttft"] for r in results if r.get("ttft", 0) > 0]
        
        throughput_metrics = {
            "prefill_speed": sum(valid_prefill) / len(valid_prefill) if valid_prefill else 0.0,
            "decode_speed": sum(valid_decode) / len(valid_decode) if valid_decode else 0.0,
            "ttft": sum(valid_ttft) / len(valid_ttft) if valid_ttft else 0.0
        }
        
        reasoning_accuracy = {
            "needle": "N/A",
            "ruler": "N/A",
            "longbench": "N/A",
            "swe_bench": "N/A"
        }
        for r in results:
            bench_key = r["benchmark"].lower().replace("-", "_")
            if bench_key in SWE_BENCH_KEYS:
                bench_key = "swe_bench"
            if bench_key in reasoning_accuracy:
                reasoning_accuracy[bench_key] = "Pass" if r["passed"] else "Fail"
                
        run_data = {
            "run_metadata": {
                "timestamp": timestamp,
                "target_endpoint": args.endpoint,
                "cli_arguments": sys.argv[1:]
            },
            "model_settings": model_settings,
            "throughput_metrics": throughput_metrics,
            "reasoning_accuracy": reasoning_accuracy,
            "quantization_loss": {
                "perplexity": None,
                "mean_kld": None,
                "same_top_match_percent": None
            }
        }
        
        history_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
        os.makedirs(history_dir, exist_ok=True)
        safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
        output_file = os.path.join(history_dir, f"run_{safe_timestamp}.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=4)
        print(f"[+] Saved structured historical run to {output_file}")

if __name__ == "__main__":
    main()
