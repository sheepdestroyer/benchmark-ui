# Advanced Benchmarking Pipeline Roadmap

This document outlines the roadmap for implementing, executing, and visualizing benchmarks for long-context capabilities (`~200K` tokens), model perplexity/quantization loss, and standard ORM/throughput workloads.

## Roadmap & Status

| Phase | Benchmark | Description | Type | Status | File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **Needle in a Haystack** | Find a custom fact ("needle") placed at a specific depth inside a synthetic corpus ("haystack"). | Synthetic | **COMPLETED** | [advanced_benchmarks.py](file:///home/sheepdestroyer/LAB/IA/bench/advanced_benchmarks.py) |
| **Phase 2** | **RULER** | Run a representative multi-hop variable assignment tracking task over deep contexts. | Synthetic | **COMPLETED** | [advanced_benchmarks.py](file:///home/sheepdestroyer/LAB/IA/bench/advanced_benchmarks.py) |
| **Phase 3** | **LongBench** | Load a long historical narrative QA task to verify document reading comprehension. | Real-world | **COMPLETED** | [advanced_benchmarks.py](file:///home/sheepdestroyer/LAB/IA/bench/advanced_benchmarks.py) |
| **Phase 4** | **SWE-bench** | Local toy repository issue debugging and dynamic unit-test validation loop (`calculator.py`). | Real-world | **COMPLETED** | [advanced_benchmarks.py](file:///home/sheepdestroyer/LAB/IA/bench/advanced_benchmarks.py) |
| **Phase 4.5**| **KLD / Perplexity** | Measure information loss (KL Divergence, Perplexity, Same Top Token matching) of quantized KV caches. | Quant Loss | **COMPLETED** | [kld_benchmark.py](file:///home/sheepdestroyer/LAB/IA/bench/kld_benchmark.py) |
| **Phase 5** | **Unified Dashboard**| Regroup, unify running benchmarks, record historical runs, and plot comparative metrics. | WebUI / DB | **PLANNED** | *Pending* |

---

## Phase 5: Unified Benchmarking & Visualization Suite (Plan)

To centralize all benchmark results and provide a state-of-the-art WebUI, we will build the following system:

### 1. Unified Run Orchestration (`bench/run_suite.py`)
Create a single python entrypoint to execute all benchmarks:
*   Allows executing the original ORM throughput tests (`benchmark.sh`), the long-context reasoning tests (`advanced_benchmarks.py`), and the KLD perplexity tests (`kld_benchmark.py`).
*   Accepts customizable parameters (`--endpoint`, `--model`, `--tokens`, `--kv-quant`, etc.).

### 2. Historical Run Database & Storage (`bench/history/`)
Establish a persistent storage system for benchmark runs:
*   **Storage Format**: Structured JSON records saved in `bench/history/run_[timestamp].json`.
*   **Schema**:
    *   `run_metadata`: Timestamp, target endpoint, CLI arguments.
    *   `model_settings`: Model name, base quantization, KV cache quant (`q5_1`/`q8_0`/`f16`), threads, ubatch/batch sizes, speculative draft type.
    *   `throughput_metrics`: Prefill speed (t/s), Decode speed (t/s), TTFT (s).
    *   `reasoning_accuracy`: Needle (Pass/Fail), RULER (Pass/Fail), LongBench (Pass/Fail), SWE-bench (Pass/Fail).
    *   `quantization_loss`: Perplexity (PPL), Mean KLD, Same Top % matching.

### 3. Configurable WebUI Dashboard (`bench/dashboard.py`)
Build a high-performance Streamlit WebUI to view and filter historical runs:
*   **Run History Browser**: A table summarizing all past runs with sorting.
*   **Advanced Filters Sidebar**: Filter runs by model name, endpoint, KV cache quant, context length, etc.
*   **Comparative Plotting**:
    *   *Throughput vs. Context Length*: Plot prefill/decode speeds as context grows.
    *   *Accuracy Comparison*: Bar charts comparing reasoning benchmark accuracy across configurations.
    *   *Quantization Trade-offs*: Plot KLD/Perplexity vs. VRAM savings (e.g. `f16` vs. `q8_0` vs. `q5_1`).
*   **Side-by-Side Model Comparison**: Compare two specific model runs side-by-side.

---

## Next Steps
1. Create a `bench/history/` directory to store historical run JSONs.
2. Update the existing benchmarking tools to automatically export their metrics to JSON under `bench/history/`.
3. Create the unified run orchestrator (`bench/run_suite.py`).
4. Implement the interactive dashboard (`bench/dashboard.py`).
