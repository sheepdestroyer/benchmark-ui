#!/usr/bin/env python3
import argparse
import ast
import configparser
import datetime
import functools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

import requests

from utils import CONTEXT_TIERS as CONTEXT_TIERS, parse_context_tokens, redact_cli_args

# Pre-built mapping of (lowercase_search_string, canonical_quantization)
QUANTIZATION_OPTIONS = (
    "Q4_K_S",
    "Q4_K_M",
    "Q4_K_L",
    "Q4_K_XL",
    "Q5_K_S",
    "Q5_K_M",
    "Q8_0",
    "f16",
)
QUANTIZATIONS = tuple((q.lower(), q) for q in QUANTIZATION_OPTIONS)
CACHE_TYPE_CLI_ARGS = {"--cache-type-k", "--cache-type-v"}
CACHE_TYPE_KEYS = {"cache-type-k", "cache-type-v"}
SWE_BENCH_KEYS = {"swe-bench", "swe_bench"}
QUANTIZATION_OPTIONS = (
    "Q4_K_S",
    "Q4_K_M",
    "Q4_K_L",
    "Q4_K_XL",
    "Q5_K_S",
    "Q5_K_M",
    "Q8_0",
    "f16",
)
QUANTIZATION_OPTIONS_LOWER = tuple((q, q.lower()) for q in QUANTIZATION_OPTIONS)


DANGEROUS_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "pty",
    "posix",
    "builtins",
    "_frozen_importlib",
    "importlib",
    "ctypes",
    "inspect",
    "pickle",
    "shelve",
    "multiprocessing",
    "threading",
    "signal",
    "asyncio",
    "pathlib",
    "io",
    "runpy",
    "operator",
    "urllib",
    "http",
    "webbrowser",
    "tempfile",
    "pdb",
    "code",
}

DANGEROUS_BUILTINS = {
    "eval",
    "exec",
    "open",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "dir",
    "exit",
    "quit",
    "help",
}

SAFE_DUNDER_IDENTIFIERS = {"__name__", "__doc__"}

SAFE_DUNDER_ATTRIBUTES = {
    "__init__",
    "__new__",
    "__enter__",
    "__exit__",
    "__len__",
    "__repr__",
    "__str__",
    "__bytes__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__iter__",
    "__next__",
    "__contains__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__and__",
    "__or__",
    "__xor__",
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rtruediv__",
    "__bool__",
    "__int__",
    "__float__",
    "__abs__",
    "__hash__",
    "__call__",
    "__reversed__",
    "__format__",
    "__doc__",
    "__name__",
}

DANGER_DUNDER_CONSTANTS = (
    "__builtins__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__import__",
)


def _get_attribute_full_path(node):
    parts = []
    curr = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    if isinstance(curr, ast.Name):
        parts.append(curr.id)
        return ".".join(reversed(parts)), curr.id
    return None, None


def is_safe_code(code_str):
    if not isinstance(code_str, str):
        return False, "Invalid code input: code must be a string"
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except (TypeError, ValueError) as e:
        return False, f"Invalid code input: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "__" in alias.name and alias.name != "__future__":
                    return False, f"Forbidden import: {alias.name}"
                if alias.asname and "__" in alias.asname:
                    return False, f"Forbidden alias: {alias.asname}"
                mod = alias.name.split(".")[0]
                if mod in DANGEROUS_MODULES or alias.name in DANGEROUS_MODULES:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if "__" in node.module and node.module != "__future__":
                    return False, f"Forbidden import module: {node.module}"
                mod = node.module.split(".")[0]
                if mod in DANGEROUS_MODULES or node.module in DANGEROUS_MODULES:
                    return False, f"Forbidden import module: {node.module}"
            for alias in node.names:
                if "__" in alias.name:
                    return False, f"Forbidden import: {alias.name}"
                if alias.asname and "__" in alias.asname:
                    return False, f"Forbidden alias: {alias.asname}"
                if alias.name in DANGEROUS_MODULES:
                    return False, f"Forbidden import: {alias.name}"
                if alias.name in DANGEROUS_BUILTINS:
                    return False, f"Forbidden import: {alias.name}"
                if node.module:
                    full_import = f"{node.module}.{alias.name}"
                    if full_import in DANGEROUS_MODULES:
                        return False, f"Forbidden import: {full_import}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                if node.attr not in SAFE_DUNDER_ATTRIBUTES:
                    return False, f"Forbidden dunder attribute: {node.attr}"
            full_path, root = _get_attribute_full_path(node)
            if root and root in DANGEROUS_MODULES:
                return False, f"Forbidden attribute usage: {full_path}"
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                if node.id not in SAFE_DUNDER_IDENTIFIERS:
                    return False, f"Forbidden dunder identifier: {node.id}"
            if node.id in DANGEROUS_BUILTINS:
                return False, f"Forbidden identifier usage: {node.id}"
            if node.id in DANGEROUS_MODULES:
                return False, f"Forbidden identifier usage: {node.id}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if (
                    node.func.id.startswith("__")
                    and node.func.id not in SAFE_DUNDER_IDENTIFIERS
                ):
                    return False, f"Forbidden dunder identifier: {node.func.id}"
                if node.func.id in DANGEROUS_BUILTINS:
                    return False, f"Forbidden identifier usage: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr.startswith("__") and node.func.attr.endswith("__"):
                    if node.func.attr not in SAFE_DUNDER_ATTRIBUTES:
                        return False, f"Forbidden dunder attribute: {node.func.attr}"
                if node.func.attr in DANGEROUS_BUILTINS:
                    full_path, root = _get_attribute_full_path(node.func)
                    if node.func.attr == "compile" and root == "re":
                        pass
                    else:
                        return False, f"Forbidden call: {node.func.attr}"
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                for danger in DANGER_DUNDER_CONSTANTS:
                    if danger in node.value:
                        return False, f"Forbidden dunder constant: {danger}"

    return True, None


# ==============================================================================
# COMMON UTILITIES & DATA GENERATION
# ==============================================================================


def generate_filler_text(target_tokens=200000):
    if not (
        isinstance(target_tokens, (int, float)) and not isinstance(target_tokens, bool)
    ):
        raise TypeError("target_tokens must be an integer or float")
    if target_tokens <= 0:
        return []

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
        "Regular expressions are powerful tools for pattern matching and text manipulation.",
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


def call_endpoint(endpoint, model, prompt, max_tokens=512, api_key=None, timeout=None):
    if timeout is None:
        prompt_len = (
            len(prompt) if hasattr(prompt, "__len__") else len(str(prompt or ""))
        )
        timeout = min(max(300, prompt_len // 600), 7200)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    url = f"{endpoint}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = (
        api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    )
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    start_time = time.time()
    first_token_time = None
    response_chunks = []
    reasoning_chunks = []
    usage = None

    try:
        with requests.post(
            url, json=payload, headers=headers, stream=True, timeout=timeout
        ) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                return None

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_content = line_str[6:]
                    if data_content.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_content)
                        if chunk.get("choices"):
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content")
                            reasoning_content = delta.get("reasoning_content")

                            if content is not None and not isinstance(content, str):
                                raise TypeError(
                                    f"delta.content must be str, got {type(content).__name__}"
                                )
                            if reasoning_content is not None and not isinstance(
                                reasoning_content, str
                            ):
                                raise TypeError(
                                    f"delta.reasoning_content must be str, got {type(reasoning_content).__name__}"
                                )

                            if first_token_time is None and (
                                content or reasoning_content
                            ):
                                first_token_time = time.time()

                            if content:
                                response_chunks.append(content)
                            if reasoning_content:
                                reasoning_chunks.append(reasoning_content)

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

    response_text = "".join(response_chunks)
    reasoning_text = "".join(reasoning_chunks)

    return {
        "response": response_text,
        "reasoning": reasoning_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft": ttft,
        "decode_time": decode_time,
        "prefill_speed": prefill_speed,
        "decode_speed": decode_speed,
        "total_time": total_time,
    }


# ==============================================================================
# BENCHMARK 1: NEEDLE IN A HAYSTACK
# ==============================================================================


def run_needle_test(
    endpoint, model, tokens=200000, depth=0.5, api_key=None, max_tokens=2048
):
    tokens = parse_context_tokens(tokens)
    effective_timeout = max(300, tokens // 150)
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

    res = call_endpoint(
        endpoint,
        model,
        prompt,
        max_tokens=max_tokens,
        api_key=api_key,
        timeout=effective_timeout,
    )
    if not res:
        return None

    is_correct = bool(re.search(r"\bBANANA_SPLIT\b", res["response"])) or bool(
        re.search(r"\bBANANA_SPLIT\b", res["reasoning"])
    )

    print("\n---------------------------------------------------------")
    print(f"Needle Result      : {'PASSED' if is_correct else 'FAILED'}")
    if res["reasoning"]:
        print(f"Model Reasoning    : {res['reasoning'].strip()}")
    print(f"Model Answer       : {res['response'].strip()}")
    print(
        f"TTFT (Prefill Lat) : {res['ttft']:.2f}s (Speed: {res['prefill_speed']:.2f} t/s)"
    )
    print(
        f"Decode Time        : {res['decode_time']:.2f}s (Speed: {res['decode_speed']:.2f} t/s)"
    )
    print("---------------------------------------------------------")

    res["benchmark"] = "Needle"
    res["passed"] = is_correct
    return res


# ==============================================================================
# BENCHMARK 2: RULER (VARIABLE TRACKING CHAIN)
# ==============================================================================


def run_ruler_test(endpoint, model, tokens=200000, api_key=None, max_tokens=2048):
    tokens = parse_context_tokens(tokens)
    effective_timeout = max(300, tokens // 150)
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

    res = call_endpoint(
        endpoint,
        model,
        prompt,
        max_tokens=max_tokens,
        api_key=api_key,
        timeout=effective_timeout,
    )
    if not res:
        return None

    is_correct = bool(re.search(r"\b93\b", res["response"])) or bool(
        re.search(r"\b93\b", res["reasoning"])
    )

    print("\n---------------------------------------------------------")
    print(f"RULER Result       : {'PASSED' if is_correct else 'FAILED'}")
    if res["reasoning"]:
        print(f"Model Reasoning    : {res['reasoning'].strip()}")
    print(f"Model Answer       : {res['response'].strip()}")
    print(
        f"TTFT (Prefill Lat) : {res['ttft']:.2f}s (Speed: {res['prefill_speed']:.2f} t/s)"
    )
    print(
        f"Decode Time        : {res['decode_time']:.2f}s (Speed: {res['decode_speed']:.2f} t/s)"
    )
    print("---------------------------------------------------------")

    res["benchmark"] = "RULER"
    res["passed"] = is_correct
    return res


# ==============================================================================
# BENCHMARK 3: LONGBENCH (DOCUMENT QA)
# ==============================================================================


def run_longbench_test(endpoint, model, tokens=200000, api_key=None, max_tokens=2048):
    tokens = parse_context_tokens(tokens)
    effective_timeout = max(300, tokens // 150)
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

    res = call_endpoint(
        endpoint,
        model,
        prompt,
        max_tokens=max_tokens,
        api_key=api_key,
        timeout=effective_timeout,
    )
    if not res:
        return None

    is_correct = bool(re.search(r"\b1452\b", res["response"])) or bool(
        re.search(r"\b1452\b", res["reasoning"])
    )

    print("\n---------------------------------------------------------")
    print(f"LongBench Result   : {'PASSED' if is_correct else 'FAILED'}")
    if res["reasoning"]:
        print(f"Model Reasoning    : {res['reasoning'].strip()}")
    print(f"Model Answer       : {res['response'].strip()}")
    print(
        f"TTFT (Prefill Lat) : {res['ttft']:.2f}s (Speed: {res['prefill_speed']:.2f} t/s)"
    )
    print(
        f"Decode Time        : {res['decode_time']:.2f}s (Speed: {res['decode_speed']:.2f} t/s)"
    )
    print("---------------------------------------------------------")

    res["benchmark"] = "LongBench"
    res["passed"] = is_correct
    return res


# ==============================================================================
# BENCHMARK 4: SWE-BENCH (TOY CODEBASE DEBUGGING)
# ==============================================================================


def _extract_code_block_from_response(raw_response, reasoning):
    """Extract a python code block from raw response or reasoning trace fallback."""
    new_code = ""
    raw_response = str(raw_response or "")
    reasoning = str(reasoning or "")
    if "```python" in raw_response:
        parts = raw_response.split("```python")
        if len(parts) > 1:
            new_code = parts[1].split("```")[0].strip()

    if not new_code and "```python" in reasoning:
        print("Parsing code block from reasoning trace fallback...")
        parts = reasoning.split("```python")
        if len(parts) > 1:
            new_code = parts[1].split("```")[0].strip()

    return new_code


def run_swe_test(endpoint, model, max_tokens=16384, api_key=None):
    print("\n=== Running SWE-bench Codebase Debugging Test ===")

    toy_repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_repo")
    code_path = os.path.join(toy_repo_dir, "calculator.py")
    test_path = os.path.join(toy_repo_dir, "test_calculator.py")
    buggy_fixture_path = os.path.join(toy_repo_dir, "fixtures", "buggy_calculator.py")

    if not os.path.exists(code_path) or not os.path.exists(test_path):
        print("Error: Toy repository files not found.")
        return None

    backup_path = code_path + ".bak"

    # Recover cleanly from a previously interrupted run if backup exists
    if os.path.exists(backup_path):
        try:
            os.replace(backup_path, code_path)
        except OSError:
            shutil.copy2(backup_path, code_path)
            os.remove(backup_path)

    has_backup = False

    try:
        if os.path.exists(buggy_fixture_path):
            shutil.copy2(code_path, backup_path)
            has_backup = True
            shutil.copy2(buggy_fixture_path, code_path)

        # Read files
        with open(code_path, "r", encoding="utf-8") as f:
            code_content = f.read()
        with open(test_path, "r", encoding="utf-8") as f:
            test_content = f.read()

        # Construct prompt
        prompt = f"""You are an automated software engineer. Fix the order-of-operations bug in the file calculator.py so that all tests pass. If all tests already pass or once fixed, output the complete python code block immediately without exhaustive verification.

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
        res = call_endpoint(
            endpoint, model, prompt, max_tokens=max_tokens, api_key=api_key, timeout=600
        )
        if not res:
            return None

        # Parse code block from response
        new_code = _extract_code_block_from_response(
            res.get("response", ""), res.get("reasoning", "")
        )

        if not new_code:
            print("Error: Could not parse python code block from response.")
            is_correct = False
        else:
            is_safe, reason = is_safe_code(new_code)
            if not is_safe:
                print(
                    f"Error: Generated code failed security sandboxing check: {reason}"
                )
                is_correct = False
            else:
                if not has_backup:
                    shutil.copy2(code_path, backup_path)
                    has_backup = True

                # Write new code
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(new_code)

                # Run unit tests
                try:
                    test_run = subprocess.run(
                        [sys.executable, "-m", "unittest", "test_calculator.py"],
                        cwd=toy_repo_dir,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )

                    print(test_run.stdout)
                    print(test_run.stderr)

                    is_correct = test_run.returncode == 0
                except Exception as e:
                    print(f"Failed to execute tests: {e}")
                    is_correct = False
    finally:
        # Restore backup
        if os.path.exists(backup_path):
            try:
                os.replace(backup_path, code_path)
            except OSError:
                shutil.copy2(backup_path, code_path)
                os.remove(backup_path)

    print("\n---------------------------------------------------------")
    print(f"SWE-bench Result   : {'PASSED' if is_correct else 'FAILED'}")
    if res.get("reasoning"):
        print(f"Model Reasoning    : {res['reasoning'].strip()}")
    print(
        f"TTFT (Prefill Lat) : {res.get('ttft', 0.0):.2f}s (Speed: {res.get('prefill_speed', 0.0):.2f} t/s)"
    )
    print(
        f"Decode Time        : {res.get('decode_time', 0.0):.2f}s (Speed: {res.get('decode_speed', 0.0):.2f} t/s)"
    )
    print("---------------------------------------------------------")

    res["benchmark"] = "SWE-bench"
    res["passed"] = is_correct
    return res


# ==============================================================================
# MAIN RUNNER
# ==============================================================================


def _normalize_repo_id(val):
    if not val or not isinstance(val, str):
        return ""
    val = val.strip()
    if val.lower().startswith("unsloth/"):
        return val[len("unsloth/") :].strip()
    return val


def _repo_id_matches(target, candidate):
    if (
        not target
        or not candidate
        or not isinstance(target, str)
        or not isinstance(candidate, str)
    ):
        return False
    target_clean = target.strip()
    candidate_clean = candidate.strip()
    if not target_clean or not candidate_clean:
        return False
    if target_clean.lower() == candidate_clean.lower():
        return True
    norm_target = _normalize_repo_id(target_clean).lower()
    norm_candidate = _normalize_repo_id(candidate_clean).lower()
    return bool(norm_target and norm_candidate and norm_target == norm_candidate)


def resolve_presets_path(presets_file=None):
    """Resolve presets file path with PRESETS_FILE env var and default fallback paths."""
    if presets_file is not None:
        return str(presets_file)
    env_file = os.environ.get("PRESETS_FILE")
    if env_file:
        return env_file
    primary = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini"
        )
    )
    if os.path.exists(primary):
        return primary
    fallback = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../llama.cpp/model_presets.ini")
    )
    if os.path.exists(fallback):
        return fallback
    return primary


@functools.lru_cache(maxsize=4)
def load_presets_config(presets_file=None):
    """Load and parse model_presets.ini with LRU caching."""
    target_file = resolve_presets_path(presets_file)
    if not os.path.exists(target_file):
        return None

    try:
        config = configparser.ConfigParser(strict=False)
        config.read(target_file, encoding="utf-8")
        return config
    except Exception:
        return None


_get_presets_config = load_presets_config


@functools.lru_cache(maxsize=128)
def map_repo_to_preset_alias(repo_or_id, presets_file=None):
    """Map model repository or ID string to preset alias defined in model_presets.ini."""
    if not repo_or_id or not isinstance(repo_or_id, str):
        return repo_or_id

    config = load_presets_config(presets_file)
    if config is None:
        return repo_or_id

    try:
        for section in config.sections():
            if _repo_id_matches(section, repo_or_id):
                return section

        for section in config.sections():
            if section == "*":
                continue
            section_repo = config.get(section, "hf-repo", fallback="")
            section_alias = config.get(section, "alias", fallback="")

            if section_repo and _repo_id_matches(section_repo, repo_or_id):
                return section

            if section_alias:
                alias_parts = [a.strip() for a in section_alias.split(",") if a.strip()]
                for a in alias_parts:
                    if _repo_id_matches(a, repo_or_id):
                        return section
                if _repo_id_matches(section_alias, repo_or_id):
                    return section
    except Exception:
        pass

    return repo_or_id


def get_preset_metadata(profile_name, presets_file=None):
    metadata = {
        "spec_type": "None",
        "spec_draft_type_k": "None",
        "spec_draft_type_v": "None",
        "flash_attn": "true",
        "parallel": "1",
        "n_gpu_layers": "99",
        "fit": "true",
    }

    config = load_presets_config(presets_file)
    if config:
        try:
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


def _parse_endpoint_model_args(args):
    """Parse runtime model CLI arguments for threads, batch sizes, and cache types."""
    parsed = {}
    if not args or not isinstance(args, (list, tuple)):
        return parsed

    for i, arg in enumerate(args):
        if not isinstance(arg, str):
            continue
        if arg == "--threads" and i + 1 < len(args):
            try:
                parsed["threads"] = int(args[i + 1])
            except (ValueError, TypeError):
                parsed["threads"] = None
        elif arg == "--batch-size" and i + 1 < len(args):
            try:
                parsed["batch_size"] = int(args[i + 1])
            except (ValueError, TypeError):
                parsed["batch_size"] = None
        elif arg == "--ubatch-size" and i + 1 < len(args):
            try:
                parsed["ubatch_size"] = int(args[i + 1])
            except (ValueError, TypeError):
                parsed["ubatch_size"] = None
        elif arg in CACHE_TYPE_CLI_ARGS and i + 1 < len(args):
            parsed["kv_cache_quant"] = args[i + 1]

    return parsed


def _parse_endpoint_preset_block(preset_str):
    """Parse preset block string for threads, batch sizes, and cache type keys."""
    parsed = {}
    if not preset_str or not isinstance(preset_str, str):
        return parsed

    for line in preset_str.split("\n"):
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k == "threads":
                try:
                    parsed["threads"] = int(v)
                except (ValueError, TypeError):
                    parsed["threads"] = None
            elif k == "batch-size":
                try:
                    parsed["batch_size"] = int(v)
                except (ValueError, TypeError):
                    parsed["batch_size"] = None
            elif k == "ubatch-size":
                try:
                    parsed["ubatch_size"] = int(v)
                except (ValueError, TypeError):
                    parsed["ubatch_size"] = None
            elif k in CACHE_TYPE_KEYS:
                parsed["kv_cache_quant"] = v

    return parsed


def get_model_settings_from_endpoint(endpoint, target_model, api_key=None):
    settings = {
        "model_name": target_model,
        "base_quantization": "Unknown",
        "kv_cache_quant": "Unknown",
        "threads": None,
        "ubatch_size": None,
        "batch_size": None,
        "speculative_draft_type": "None",
    }

    # Try parsing base quantization from target_model name if it's there
    target_model_lower = target_model.lower()
    for q_lower, q in QUANTIZATIONS:
        if q_lower in target_model_lower:
            settings["base_quantization"] = q
            break

    try:
        url = f"{endpoint}/v1/models"
        api_key = (
            api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        )
        headers = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        with requests.get(url, headers=headers, timeout=5) as response:
            if response.status_code == 200:
                data = response.json()
                model_info = None
                for item in data.get("data", []):
                    if item.get("id") == target_model or target_model in item.get(
                        "id", ""
                    ):
                        model_info = item
                        break

                if model_info:
                    status = model_info.get("status", {})
                    args = status.get("args", [])
                    preset = status.get("preset", "")

                    settings.update(_parse_endpoint_model_args(args))
                    settings.update(_parse_endpoint_preset_block(preset))

                    repo_or_id = model_info.get("id", "")
                    repo_or_id_lower = repo_or_id.lower()
                    for q_lower, q in QUANTIZATIONS:
                        if q_lower in repo_or_id_lower:
                            settings["base_quantization"] = q
                            break

                    if "mtp" in repo_or_id_lower or any(
                        "spec" in str(arg).lower() for arg in args
                    ):
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


def _print_benchmark_summary(results):
    """Print formatted summary table of benchmark results."""
    print("\n=== Advanced Benchmark Suite Summary ===")
    print(
        "--------------------------------------------------------------------------------"
    )
    print(
        f"{'Benchmark':<12} | {'Status':<6} | {'Prompt tks':<10} | {'TTFT':<6} | {'Prefill t/s':<12} | {'Decode t/s':<10}"
    )
    print(
        "--------------------------------------------------------------------------------"
    )
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        prompt_tokens = r.get("prompt_tokens", 0)
        ttft = r.get("ttft", 0.0)
        prefill_speed = r.get("prefill_speed", 0.0)
        decode_speed = r.get("decode_speed", 0.0)
        print(
            f"{r.get('benchmark', ''):<12} | {status:<6} | {prompt_tokens:<10} | {ttft:>5.2f}s | {prefill_speed:>10.2f}  | {decode_speed:>8.2f}"
        )
    print(
        "--------------------------------------------------------------------------------"
    )


def _save_run_data(
    results, endpoint, model, cli_arguments, output_path=None, api_key=None
):
    """Aggregate benchmark results and write structured historical run JSON."""
    timestamp = datetime.datetime.now().isoformat()
    model_settings = get_model_settings_from_endpoint(endpoint, model, api_key=api_key)

    valid_prefill = [
        r["prefill_speed"] for r in results if r.get("prefill_speed", 0) > 0
    ]
    valid_decode = [r["decode_speed"] for r in results if r.get("decode_speed", 0) > 0]
    valid_ttft = [r["ttft"] for r in results if r.get("ttft", 0) > 0]

    throughput_metrics = {
        "prefill_speed": sum(valid_prefill) / len(valid_prefill)
        if valid_prefill
        else 0.0,
        "decode_speed": sum(valid_decode) / len(valid_decode) if valid_decode else 0.0,
        "ttft": sum(valid_ttft) / len(valid_ttft) if valid_ttft else 0.0,
    }

    reasoning_accuracy = {
        "needle": "N/A",
        "ruler": "N/A",
        "longbench": "N/A",
        "swe_bench": "N/A",
    }
    for r in results:
        bench_key = r.get("benchmark", "").lower().replace("-", "_")
        if bench_key in SWE_BENCH_KEYS:
            bench_key = "swe_bench"
        if bench_key in reasoning_accuracy:
            reasoning_accuracy[bench_key] = "Pass" if r.get("passed") else "Fail"

    run_data = {
        "run_metadata": {
            "timestamp": timestamp,
            "target_endpoint": endpoint,
            "cli_arguments": redact_cli_args(cli_arguments),
        },
        "model_settings": model_settings,
        "throughput_metrics": throughput_metrics,
        "reasoning_accuracy": reasoning_accuracy,
        "quantization_loss": {
            "perplexity": None,
            "mean_kld": None,
            "same_top_match_percent": None,
        },
    }

    if output_path:
        output_file = output_path
        parent_dir = os.path.dirname(output_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    else:
        history_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "history"
        )
        os.makedirs(history_dir, exist_ok=True)
        safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
        output_file = os.path.join(history_dir, f"run_{safe_timestamp}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=4)

    print(f"[+] Saved structured historical run to {output_file}")
    return output_file


def main(args=None):
    parser = argparse.ArgumentParser(description="Advanced Benchmarks Runner")
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8083", help="LLM server API endpoint"
    )
    parser.add_argument(
        "--model", default="Qwen3.6-27B", help="Model name / alias to target"
    )
    parser.add_argument(
        "--tokens",
        type=parse_context_tokens,
        default=200000,
        help="Number of context tokens for synthetic benchmarks (Needle, RULER, LongBench) or tier ('8k', '32k', '64k', '128k', '240k')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum completion tokens to generate",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        help="API key for Bearer authentication",
    )
    parser.add_argument(
        "--needle",
        action="store_true",
        help="Run Needle in a Haystack benchmark (Phase 1)",
    )
    parser.add_argument(
        "--ruler",
        action="store_true",
        help="Run RULER variable tracking benchmark (Phase 2)",
    )
    parser.add_argument(
        "--longbench", action="store_true", help="Run LongBench QA benchmark (Phase 3)"
    )
    parser.add_argument(
        "--swe",
        action="store_true",
        help="Run SWE-bench toy repository debugging benchmark (Phase 4)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all benchmarks sequentially"
    )
    parser.add_argument(
        "--benchmark",
        choices=["needle", "ruler", "longbench", "swe", "all"],
        default=None,
        help="Benchmark to execute",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output file path for run JSON",
    )

    parsed_args = parser.parse_args(args)

    run_needle = parsed_args.needle or (parsed_args.benchmark == "needle")
    run_ruler = parsed_args.ruler or (parsed_args.benchmark == "ruler")
    run_longbench = parsed_args.longbench or (parsed_args.benchmark == "longbench")
    run_swe = parsed_args.swe or (parsed_args.benchmark == "swe")
    run_all = parsed_args.all or (parsed_args.benchmark == "all")

    # If no flags are set, default to running all of them
    if not (run_needle or run_ruler or run_longbench or run_swe or run_all):
        run_all = True

    results = []
    call_kwargs = {}
    if parsed_args.api_key:
        call_kwargs["api_key"] = parsed_args.api_key

    if run_needle or run_all:
        res = run_needle_test(
            parsed_args.endpoint,
            parsed_args.model,
            tokens=parsed_args.tokens,
            max_tokens=parsed_args.max_tokens,
            **call_kwargs,
        )
        if res:
            results.append(res)

    if run_ruler or run_all:
        res = run_ruler_test(
            parsed_args.endpoint,
            parsed_args.model,
            tokens=parsed_args.tokens,
            max_tokens=parsed_args.max_tokens,
            **call_kwargs,
        )
        if res:
            results.append(res)

    if run_longbench or run_all:
        res = run_longbench_test(
            parsed_args.endpoint,
            parsed_args.model,
            tokens=parsed_args.tokens,
            max_tokens=parsed_args.max_tokens,
            **call_kwargs,
        )
        if res:
            results.append(res)

    if run_swe or run_all:
        res = run_swe_test(
            parsed_args.endpoint,
            parsed_args.model,
            max_tokens=parsed_args.max_tokens,
            **call_kwargs,
        )
        if res:
            results.append(res)

    if results:
        _print_benchmark_summary(results)
        cli_args = sys.argv[1:] if args is None else list(args)
        _save_run_data(
            results,
            parsed_args.endpoint,
            parsed_args.model,
            cli_args,
            output_path=parsed_args.output,
            **call_kwargs,
        )

    return results


if __name__ == "__main__":
    main()
