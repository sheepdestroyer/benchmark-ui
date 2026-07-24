# LLM Benchmarking & Optimization Registry Suite

A comprehensive benchmarking pipeline and Streamlit dashboard designed to evaluate, optimize, and visualize multi-GPU `llama.cpp` server Router deployments.

## Directory Structure

- `dashboard.py` - The state-of-the-art interactive Streamlit UI dashboard.
- `run_suite.py` - The unified run orchestrator for executing different benchmark modes.
- `advanced_benchmarks.py` - The long-context and agentic reasoning benchmarks pipeline.
- `kld_benchmark.py` - KV Cache quantization Kullback-Leibler (KL) Divergence and perplexity evaluation.
- `history/` - Registry directory storing historical run JSON logs.
- `benchmark.sh` - The original bash ORM throughput benchmark script.
- `requirements.txt` - Python package dependencies (Streamlit, Pandas, Plotly).
- `run_ui.sh` - Convenience script to launch the interactive dashboard.

## Installation

1. Ensure you have Python 3.8+ installed.
2. Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Launch the Dashboard

Use the convenience script to launch the Streamlit dashboard:

```bash
./run_ui.sh
```

Or run manually:

```bash
streamlit run dashboard.py
```

### 2. Execute the Unified Runner Script

To run all benchmarks (throughput, reasoning, and local KLD quantization analysis) under a single command:

```bash
python3 run_suite.py --mode all --endpoint http://127.0.0.1:8081 --model unsloth/Qwen3.6-27B-GGUF:Q4_K_S
```

#### Run Modes:
- `--mode throughput`: Runs only the ORM and Markdown throughput workloads via `benchmark.sh`.
- `--mode reasoning`: Runs the long-context synthetic and real-world reasoning tasks.
- `--mode kld`: Compiles the native `llama-perplexity` binary and runs KLD analysis locally.
- `--mode all`: Orchestrates all of the above sequentially.

#### CLI Arguments:
- `--endpoint`: Server endpoint URL (default: `http://127.0.0.1:8081`).
- `--model`: Model alias or name loaded on the endpoint.
- `--tokens`: Target token context length for reasoning tests (e.g. 5000 for validation, 200000 for full scaling).
- `--gguf-path`: Local GGUF file path (for KLD mode, auto-detects Hugging Face cache if blank).
- `--corpus`: Corpus prose file path for local perplexity calculation.

---

## Historical Run Registry Schema

All benchmark executions export structured JSON logs saved under `history/run_[timestamp].json`. The schema matches:

```json
{
    "run_metadata": {
        "timestamp": "2026-07-08T02:48:25.506401",
        "target_endpoint": "http://127.0.0.1:8081",
        "cli_arguments": ["--mode", "throughput", "--model", "Qwen3.6-27B"]
    },
    "model_settings": {
        "model_name": "Qwen3.6-27B",
        "base_quantization": "Q4_K_S",
        "kv_cache_quant": "q5_1",
        "threads": 16,
        "ubatch_size": 512,
        "batch_size": 2048,
        "speculative_draft_type": "ngram"
    },
    "throughput_metrics": {
        "prefill_speed": 1326.0,
        "decode_speed": 42.04,
        "ttft": 3.43
    },
    "reasoning_accuracy": {
        "needle": "Pass",
        "ruler": "Pass",
        "longbench": "Fail",
        "swe_bench": "Pass"
    },
    "quantization_loss": {
        "perplexity": 3.9682,
        "mean_kld": 0.00115,
        "same_top_match_percent": 98.1
    }
}
```

---

## Detailed Benchmark Components

### 1. Throughput Benchmarks (`benchmark.sh`)
Runs four sequential multi-turn prompts measuring:
1. **Turn 1 (Cold Start)**: Initial latency and prefill speed during cold ORM generation.
2. **Turn 2 (KV Cache Hit)**: Cached response speed for repetitive ORM queries.
3. **Turn 3 (JSON Tool Calls)**: Performance and formatting rates for structured outputs.
4. **Turn 4 (Markdown)**: Document throughput and decode speeds.

### 2. Advanced Reasoners (`advanced_benchmarks.py`)
- **Needle in a Haystack (Needle)**: Verifies factual retrieval from configurable depths inside a deep synthetic context window.
- **RULER**: Evaluates multi-hop variable tracking chains (e.g. `alpha` -> `beta` -> `gamma`) inside long context inputs.
- **LongBench**: Validates document QA capabilities using technical corpus materials.
- **SWE-bench (Toy)**: Dynamically tests code generation by tasking the model with fixing a math calculator library, saving the fix, and validating via local unit tests.

### 3. KV Cache Quantization Loss (`kld_benchmark.py`)
Wraps the native `llama-perplexity` executable to assess information loss of quantized key-value caches (`q8_0`, `q5_1`, `q4_0`) against an unquantized `f16` baseline. Tracks:
- **Perplexity (PPL)**: Shift in prediction confidence.
- **KL Divergence (KLD)**: Statistical divergence distance compared to high-precision reference.
- **Same Top Token %**: Token matching fidelity.

---

## Interactive Dashboard Features

The Streamlit UI (`dashboard.py`) enables:
- **Run Browser**: Filter, sort, and search historical registry records.
- **Multi-GPU / Optimization Plots**: Bar charts comparing Prefill vs. Decode speeds and accuracy scores.
- **Quantization Optimizer**: Scatter plot mapping KL Divergence vs. VRAM Savings (%) to identify the ideal sweet-spot for cache quantization.
- **Side-by-Side Comparator**: Directly compare two benchmark configurations head-to-head.
- **Background Runner Control**: Run tests on the fly and stream logs to the UI.