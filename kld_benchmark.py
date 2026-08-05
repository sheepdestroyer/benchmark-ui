#!/usr/bin/env python3
import os
import math
import sys
import subprocess
import re
import argparse
import tempfile
import datetime
import json
from pathlib import Path

BENCH_DIR = Path(__file__).parent.resolve()

# Default corpus text for perplexity calculation (approx 800 tokens of technical prose)
DEFAULT_CORPUS = """
Artificial intelligence and machine learning have revolutionized the way we approach complex computational tasks. 
At the heart of modern large language models is the transformer architecture, which relies on the self-attention mechanism.
The self-attention mechanism computes a weighted representation of the input sequence, allowing the model to focus on different tokens at different positions.
However, as the context window of these models scales up to 128K, 256K, or even a million tokens, the memory footprint of the Key-Value (KV) cache becomes a severe bottleneck.
For a standard model, the KV cache grows linearly with both the sequence length and the batch size.
To mitigate this memory overhead, researchers and developers have proposed various quantization techniques for the KV cache.
By compressing the keys and values from 16-bit floating-point numbers (FP16 or BF16) to lower-precision formats like 8-bit (Q8_0) or 5-bit (Q5_1) integers, the memory footprint can be reduced by 2x to 3x.
However, down-quantization introduces quantization noise, which can degrade the model's perplexity and multi-step reasoning capabilities.
To quantify this information loss, we calculate the Kullback-Leibler (KL) divergence of the model's output probability distribution compared to a high-precision baseline.
A KL divergence of zero indicates that the output distributions are identical, while higher values indicate increasing quality degradation.
In this test, we evaluate different KV cache quantization types against a baseline reference to determine the sweet spot for long-context agentic reasoning.
"""

def compile_perplexity_binary(root_dir):
    binary_path = os.path.join(root_dir, "build/bin/llama-perplexity")
    if os.path.exists(binary_path):
        return binary_path

    print("[*] llama-perplexity binary not found. Compiling...")
    try:
        subprocess.run(
            ["cmake", "--build", "build", "--target", "llama-perplexity", "-j"],
            cwd=root_dir,
            check=True
        )
        if os.path.exists(binary_path):
            print("[+] Successfully compiled llama-perplexity.")
            return binary_path
    except Exception as e:
        print(f"[-] Compilation failed: {e}")
        sys.exit(1)
    
    print("[-] Binary missing: llama-perplexity binary not found after compilation.")
    sys.exit(1)

<<<<<<< HEAD

def sanitize_nan_inf(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_nan_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nan_inf(v) for v in obj]
    return obj

def run_perplexity(binary, model, corpus, cache_k, cache_v, base_kld=None, evaluate=False, threads=None):
    if threads is None:
        threads = os.cpu_count() or 4
    cmd = [
        binary,
        "-m", model,
        "-f", corpus,
        "--cache-type-k", cache_k,
        "--cache-type-v", cache_v,
        "--threads", str(threads),
        "--ctx-size", "128"
    ]
    
    if base_kld:
        cmd.extend(["--kl-divergence-base", base_kld])
        if evaluate:
            cmd.append("--kl-divergence")
            
    print(f"[*] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[-] llama-perplexity failed with exit code {result.returncode}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            return None, result.stderr
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        print("[-] Execution of llama-perplexity timed out after 600 seconds.")
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode('utf-8') if e.stdout else "")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode('utf-8') if e.stderr else "")
        return None, stderr

def parse_metrics(output):
    metrics = {
        "ppl": None,
        "kld": None,
        "same_top": None
    }
    
    float_pattern = r"[-+]?(?:[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|inf|nan)"
    
    # Parse final perplexity (PPL)
    ppl_match = re.search(r"(?:Final perplexity:\s*|Final estimate:\s*PPL\s*=\s*|Mean PPL\(Q\)\s*:\s*)(" + float_pattern + ")", output, re.IGNORECASE)
    if ppl_match:
        metrics["ppl"] = float(ppl_match.group(1))
        
    # Parse Mean KL divergence
    kld_match = re.search(r"Mean\s+KLD:\s*(" + float_pattern + ")", output, re.IGNORECASE)
    if kld_match:
        metrics["kld"] = float(kld_match.group(1))
        
    # Parse Same top token percentage
    top_match = re.search(r"Same top p:\s*(" + float_pattern + ")", output, re.IGNORECASE)
    if top_match:
        metrics["same_top"] = float(top_match.group(1))
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate KL-Divergence and Perplexity of KV Cache Quantizations")
    parser.add_argument("--model", type=str, help="Path to GGUF model file")
    parser.add_argument("--corpus", type=str, default="kld_corpus.txt", help="Path to evaluation text corpus")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="Number of threads for llama-perplexity")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama.cpp"))
    binary = compile_perplexity_binary(root_dir)
    
    # Resolve Model Path
    model_path = args.model
    if not model_path:
        # Auto-detect locally downloaded Qwen3.6-27B GGUF
        search_path = os.path.expanduser("~/.cache/huggingface/hub/")
        ggufs = []
        if os.path.exists(search_path):
            for root, _, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".gguf") and "27b" in file.lower() and "mtp" not in file.lower():
                        ggufs.append(os.path.join(root, file))
        if ggufs:
            model_path = ggufs[0]
            print(f"[*] Auto-detected model: {model_path}")
        else:
            print("[-] No local model file passed or detected. Please use --model <path>")
            sys.exit(1)

    # Setup Corpus
    corpus_file = args.corpus
    if not os.path.isabs(corpus_file):
        corpus_file = str(BENCH_DIR / corpus_file)

    if not os.path.exists(corpus_file):
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CORPUS.strip())
        print(f"[*] Created default corpus at {corpus_file}")

    tmp_kld_file = tempfile.NamedTemporaryFile(delete=False, suffix=".kld")
    baseline_kld_file = tmp_kld_file.name
    tmp_kld_file.close()

    try:
        # 1. Generate Baseline Logits (f16 / unquantized cache)
        print("\n[1/2] Generating baseline logits (f16 KV cache)...")
        stdout, stderr = run_perplexity(binary, model_path, corpus_file, "f16", "f16", base_kld=baseline_kld_file, threads=args.threads)
        if stdout is None:
            print("[-] Failed to generate baseline perplexity. Check logs:")
            print(stderr)
            sys.exit(1)
        baseline_metrics = parse_metrics(stdout + "\n" + stderr)
        
        if not baseline_metrics["ppl"]:
            print("[-] Failed to generate baseline perplexity. Check logs:")
            print(stderr)
            sys.exit(1)
            
        print(f"[+] Baseline PPL: {baseline_metrics['ppl']:.4f}")

        # 2. Evaluate Target Configurations
        targets = [
            ("q8_0", "q8_0"),
            ("q5_1", "q5_1"),
            ("q4_0", "q4_0")
        ]
        
        results = [
            {"name": "f16 (Baseline)", "ppl": baseline_metrics["ppl"], "kld": 0.0, "same_top": 100.0}
        ]
        
        print("\n[2/2] Evaluating quantized target caches...")
        for cache_k, cache_v in targets:
            name = f"{cache_k}"
            print(f"\n---> Evaluating {name}...")
            stdout, stderr = run_perplexity(
                binary, model_path, corpus_file, cache_k, cache_v, 
                base_kld=baseline_kld_file, evaluate=True, threads=args.threads
            )
            if stdout is None:
                m = {"ppl": None, "kld": None, "same_top": None}
            else:
                m = parse_metrics(stdout + "\n" + stderr)
            results.append({
                "name": name,
                "ppl": m["ppl"],
                "kld": m["kld"],
                "same_top": m["same_top"]
            })
    finally:
        # Cleanup temp baseline file
        if baseline_kld_file and os.path.exists(baseline_kld_file):
            os.remove(baseline_kld_file)

    # 3. Print Results Table
    print("\n" + "="*70)
    print(f"{'KV Cache Quant':<20} | {'Perplexity (PPL)':<18} | {'KL Divergence':<15} | {'Same Top %':<10}")
    print("="*70)
    for r in results:
        ppl_str = f"{r['ppl']:.4f}" if r['ppl'] is not None else "N/A"
        kld_str = f"{r['kld']:.6f}" if r['kld'] is not None else "N/A"
        top_str = f"{r['same_top']:.2f}%" if r['same_top'] is not None else "N/A"
        print(f"{r['name']:<20} | {ppl_str:<18} | {kld_str:<15} | {top_str:<10}")
    print("="*70)
    
    # Save output to markdown log
    results_md = BENCH_DIR / "kld_results.md"
    with open(results_md, "w", encoding="utf-8") as f:
        f.write("# KV Cache Quantization KLD Results\n\n")
        f.write(f"**Model**: `{os.path.basename(model_path)}`\n\n")
        f.write("| KV Cache Quant | Perplexity (PPL) | KL Divergence | Same Top % |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for r in results:
            ppl_str = f"{r['ppl']:.4f}" if r['ppl'] is not None else "N/A"
            kld_str = f"{r['kld']:.6f}" if r['kld'] is not None else "N/A"
            top_str = f"{r['same_top']:.2f}%" if r['same_top'] is not None else "N/A"
            f.write(f"| {r['name']} | {ppl_str} | {kld_str} | {top_str} |\n")
            
    print(f"[+] Saved markdown summary to {results_md}")

    # Save output to historical run registry JSON files
    history_dir = (BENCH_DIR / "history").resolve()
    history_dir.mkdir(parents=True, exist_ok=True)
    
    base_model_name = os.path.basename(model_path)
    base_quant = "Unknown"
    for q in ["Q4_K_S", "Q4_K_M", "Q4_K_L", "Q4_K_XL", "Q5_K_S", "Q5_K_M", "Q8_0", "f16"]:
        if q.lower() in base_model_name.lower():
            base_quant = q
            break
            
    for r in results:
        timestamp = datetime.datetime.now().isoformat()
        kv_quant = r["name"].replace(" (Baseline)", "")
        
        run_data = {
            "run_metadata": {
                "timestamp": timestamp,
                "target_endpoint": "local",
                "cli_arguments": sys.argv[1:]
            },
            "model_settings": {
                "model_name": base_model_name,
                "base_quantization": base_quant,
                "kv_cache_quant": kv_quant,
                "threads": args.threads,
                "ubatch_size": None,
                "batch_size": None,
                "speculative_draft_type": "None"
            },
            "throughput_metrics": {
                "prefill_speed": None,
                "decode_speed": None,
                "ttft": None
            },
            "reasoning_accuracy": {
                "needle": "N/A",
                "ruler": "N/A",
                "longbench": "N/A",
                "swe_bench": "N/A"
            },
            "quantization_loss": {
                "perplexity": r["ppl"],
                "mean_kld": r["kld"],
                "same_top_match_percent": r["same_top"]
            }
        }
        
        safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
<<<<<<< HEAD
        output_file = os.path.join(history_dir, f"run_{safe_timestamp}_{kv_quant}.json")
        run_data = sanitize_nan_inf(run_data)
=======
        output_file = history_dir / f"run_{safe_timestamp}_{kv_quant}.json"
>>>>>>> 8937156 (Fix infrastructure issues — TeeLogger, file paths, history management, and thread counts)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=4)
        print(f"[+] Saved structured KLD historical run to {output_file}")

if __name__ == "__main__":
    main()
