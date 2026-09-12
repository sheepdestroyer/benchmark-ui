# LLM Benchmarking & Optimization Registry Suite

A comprehensive benchmarking pipeline and Streamlit dashboard designed to evaluate, optimize, and visualize multi-GPU `llama.cpp` server Router deployments.

## Directory Structure

- `dashboard.py` - Main interactive Streamlit UI dashboard for browsing historical run data, multi-GPU evaluation, and quantization optimization.
- `benchmark_ui.py` - Simple runner Streamlit UI for direct benchmark execution and real-time streaming output.
- `utils.py` - Centralized security, URL/SSRF validation, and input sanitization helpers.
- `run_suite.py` - Unified run orchestrator for executing throughput, reasoning, and KLD benchmark modes.
- `run_matrix.py` - Automated benchmark matrix generator sweeping across models, threads, and quantization settings.
- `populate_history.py` - Utility script to populate the history registry with synthetic benchmark run logs.
- `advanced_benchmarks.py` - Long-context (Needle, RULER, LongBench) and agentic reasoning (SWE-bench) benchmark pipeline.
- `kld_benchmark.py` - KV cache quantization Kullback-Leibler (KL) Divergence and perplexity evaluation.
- `deploy/` - Production deployment specifications, including systemd Quadlet container definitions (`benchmark-ui.container`).
- `history/` - Registry directory storing historical run JSON logs.
- `benchmark.sh` - Bash throughput benchmark script evaluating cold start, KV cache hit, tool calls, and document decode speed.
- `run-tb-pi.sh` - Execution script for Terminal-Bench 2.0 with the pi agent and local llama.cpp server.
- `run_ui.sh` - Convenience launcher script for the Streamlit dashboard.
- `requirements.txt` - Python package dependencies (Streamlit, Pandas, Plotly, Requests).
- `requirements-dev.txt` - Development and test suite dependencies (Pytest, Pytest-Cov, Pytest-Mock, Pytest-Asyncio, Pytest-Xdist).

## Installation

1. Ensure you have Python 3.10+ installed.
2. Install the required Python dependencies:

```bash
pip install -r requirements.txt
# For development and testing tooling:
pip install -r requirements-dev.txt
```

## Quick Start

### 1. Launch the Dashboards

Use the convenience script to launch the main Streamlit dashboard:

```bash
./run_ui.sh
```

Or run either UI dashboard manually:

- **Main Interactive Dashboard (`dashboard.py`)**:
  ```bash
  streamlit run dashboard.py
  ```

- **Simple Runner UI (`benchmark_ui.py`)**:
  ```bash
  streamlit run benchmark_ui.py
  ```

### 2. Execute the Unified Runner Script

To run all benchmarks (throughput, reasoning, and local KLD quantization analysis) under a single command:

```bash
python3 run_suite.py --mode all --endpoint http://127.0.0.1:8083 --model Qwen3.6-35B-A3B
```

#### Run Modes:
- `--mode throughput`: Runs only the ORM and Markdown throughput workloads via `benchmark.sh`.
- `--mode reasoning`: Runs the long-context synthetic and real-world reasoning tasks.
- `--mode kld`: Compiles the native `llama-perplexity` binary and runs KLD analysis locally.
- `--mode all`: Orchestrates all of the above sequentially.

#### CLI Arguments:
- `--endpoint`: Server endpoint URL (default: `http://127.0.0.1:8083`).
- `--model`: Model alias or name loaded on the endpoint.
- `--tokens`: Target token context length for reasoning tests (e.g. 5000 for validation, 200000 for full scaling).
- `--gguf-path`: Local GGUF file path (for KLD mode, auto-detects Hugging Face cache if blank).
- `--corpus`: Corpus prose file path for local perplexity calculation.

---

## Automated Testing & CI/CD
 
The repository maintains an automated unit test suite (242 tests) covering input validation, UI logic, benchmark execution, and data formatting.
 
Install development dependencies and run the test suite using `pytest`:
 
```bash
pip install -r requirements-dev.txt
pytest
```

Continuous integration and dependency maintenance are automated via GitHub Actions:
- **CI Pipeline (`.github/workflows/ci.yml`)**: Automatically runs `pytest` across Python 3.14 with dependency caching on pushes and pull requests to `master`.
- **Dependabot Automation (`.github/workflows/dependabot.yml`)**: Automated review, auto-approval, and auto-merge for non-breaking (minor/patch) dependency updates.

The test suite covers:
- `test_utils.py` (100% coverage): SSRF security boundaries, URL scheme/port checks, multi-IP resolution filtering, and model/corpus parameter validation.
- `test_dashboard.py` (93% coverage): Validator paths (`validate_gguf_path` with `.gguf` extension and model cache restriction), token bounds (`validate_new_tokens` within `[1, 262144]`), `@st.cache_data(ttl=60)` run loading, number formatting (`fmt_num`), INI config caching, and DataFrame reasoning metric extraction.
- `test_advanced_benchmarks.py` (99% coverage): AST sandbox security validation (`is_safe_code` blocking `__import__`, `globals()`, dangerous modules/builtins, and reflection), LRU-cached `model_presets.ini` parsing, synthetic filler text generation, and context sizing.
- `test_benchmark_ui.py` (99% coverage): Full UI workflow testing, `list_models` endpoint interactions, `run_benchmark_stream` real-time stdout streaming, dynamic chmod fallback, subprocess failure handling, and session cleanup.
- `test_kld_benchmark.py` (99% coverage): Quantization loss calculations, perplexity metric parsing, and native `llama-perplexity` compilation lifecycle (`compile_perplexity_binary`).
- `test_run_matrix.py` (100% coverage): Automated sweep matrix generation, TeeLogger I/O buffering, service restart error handling, and LRU-cached preset sections.
- `test_run_suite.py` (99% coverage): Multi-mode benchmark orchestration, endpoint model setting querying, and atomic run record writing.
- `test_populate_history.py` (98% coverage): Synthetic history dataset generation and schema validation.

---

## Historical Run Registry Schema

All benchmark executions export structured JSON logs saved under `history/run_[timestamp].json`. The schema matches:

```json
{
    "run_metadata": {
        "timestamp": "2026-07-08T02:48:25.506401",
        "target_endpoint": "http://127.0.0.1:8083",
        "cli_arguments": ["--mode", "throughput", "--model", "Qwen3.6-35B-A3B"]
    },
    "model_settings": {
        "model_name": "Qwen3.6-35B-A3B",
        "base_quantization": "Q4_K_S",
        "kv_cache_quant": "q5_1",
        "threads": 16,
        "ubatch_size": 512,
        "batch_size": 2048,
        "speculative_draft_type": "ngram",
        "flash_attn": "true",
        "n_gpu_layers": 99,
        "tensor_split": "28,14",
        "cache_type_k": "q5_1",
        "cache_type_v": "q5_1",
        "alias": "Qwen3.6-35B-A3B",
        "hf_repo": "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S",
        "chat_template_kwargs": "{\"preserve_thinking\": true}"
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

The project includes two UI dashboard components:

1. **Main Interactive Dashboard (`dashboard.py`)**:
   - **Run Browser**: Filter, sort, and search historical registry records.
   - **Multi-GPU / Optimization Plots**: Bar charts comparing Prefill vs. Decode speeds and accuracy scores.
   - **Quantization Optimizer**: Scatter plot mapping KL Divergence vs. VRAM Savings (%) to identify the ideal sweet-spot for cache quantization.
   - **Side-by-Side Comparator**: Directly compare two benchmark configurations head-to-head.
   - **Background Runner Control**: Run tests on the fly and stream logs to the UI.

2. **Simple Runner UI (`benchmark_ui.py`)**:
   - **Direct Benchmark Trigger**: Lightweight UI interface to select model targets and launch `benchmark.sh` benchmarks directly.
   - **Real-Time Streaming**: Real-time console log streaming and output capture.

---

## Testing

The test suite provides comprehensive unit test coverage across all modules (170+ test cases):

```bash
# Run all tests
pytest

# Run tests with coverage breakdown
pytest --cov=dashboard --cov=utils --cov=run_matrix --cov=run_suite --cov=populate_history --cov-report=term-missing
```

| Test Suite | Target Module | Scope |
| :--- | :--- | :--- |
| `test_dashboard.py` | `dashboard.py` | Caching, validators, presets config, formatters, metrics extraction |
| `test_utils.py` | `utils.py` | SSRF prevention, IP validation, model name parsing |
| `test_run_matrix.py` | `run_matrix.py` | `TeeLogger`, service restart, run matching, error logging |
| `test_run_suite.py` | `run_suite.py` | Model settings retrieval, CLI fallback, atomic file persistence |
| `test_populate_history.py` | `populate_history.py` | Synthetic run record generation, schema consistency, purge options |
| `test_advanced_benchmarks.py`| `advanced_benchmarks.py` | Needle, RULER, filler text generation |
| `test_kld_benchmark.py` | `kld_benchmark.py` | Quantization loss, perplexity calculation |
| `test_benchmark_ui.py` | `benchmark_ui.py` | UI runner rendering and launch commands |
| `test_deploy.py` | `deploy/benchmark-ui.container` | Quadlet unit validation, persistent volume mount, loopback binding |

---

## Production Deployment & Persistence

The Benchmark UI is deployed in production as a rootless Podman Quadlet container managed by `systemd`.

### Persistence Architecture

Benchmark run outputs are written to `/app/history` inside the container (`dashboard.py`, `run_suite.py`, etc.). To prevent benchmark run history from being clobbered across container image updates (`AutoUpdate=registry`) or Quadlet service restarts, the container definition mounts a dedicated host directory with SELinux private relabeling (`:Z`):

```text
Host Path:      /mnt/DATA/boy/prod/benchmark-ui/data/history
Container Path: /app/history
SELinux Mode:   :Z
```

### Installation

To install or update the Quadlet container definition:

1. Ensure the host persistence directory exists (or seed it from existing runs):
   ```bash
   mkdir -p /mnt/DATA/boy/prod/benchmark-ui/data/history
   # Optional: seed existing tracked history files
   cp -n history/run_*.json /mnt/DATA/boy/prod/benchmark-ui/data/history/
   ```

2. Copy the vendored Quadlet unit into the user's systemd Quadlet directory:
   ```bash
   cp deploy/benchmark-ui.container ~/.config/containers/systemd/
   ```

3. Reload systemd and restart the service:
   ```bash
   systemctl --user daemon-reload
   systemctl --user restart benchmark-ui
   ```

### Backup Recommendations

Because benchmark runs are persisted to `/mnt/DATA/boy/prod/benchmark-ui/data/history`, backups can be taken directly from the host filesystem without stopping the container:

```bash
# Create a timestamped archive of the benchmark run history
tar -czvf benchmark-history-$(date +%Y%m%d_%H%M%S).tar.gz -C /mnt/DATA/boy/prod/benchmark-ui/data history
```

Periodic backup routines (such as BorgBackup, Restic, or daily rsync jobs) should include `/mnt/DATA/boy/prod/benchmark-ui/data/history` to ensure historical evaluation benchmarks remain preserved across host migrations and disaster recovery.

