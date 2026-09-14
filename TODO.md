# Advanced Benchmarking Pipeline Roadmap

This document outlines the roadmap for implementing, executing, and visualizing benchmarks for long-context capabilities (`~200K` tokens), model perplexity/quantization loss, and standard ORM/throughput workloads.

## Roadmap & Status

| Phase | Benchmark | Description | Type | Status | File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **Needle in a Haystack** | Find a custom fact ("needle") placed at a specific depth inside a synthetic corpus ("haystack"). | Synthetic | **COMPLETED** | [advanced_benchmarks.py](advanced_benchmarks.py) |
| **Phase 2** | **RULER** | Run a representative multi-hop variable assignment tracking task over deep contexts. | Synthetic | **COMPLETED** | [advanced_benchmarks.py](advanced_benchmarks.py) |
| **Phase 3** | **LongBench** | Load a long historical narrative QA task to verify document reading comprehension. | Real-world | **COMPLETED** | [advanced_benchmarks.py](advanced_benchmarks.py) |
| **Phase 4** | **SWE-bench** | Local toy repository issue debugging and dynamic unit-test validation loop (`calculator.py`). | Real-world | **COMPLETED** | [advanced_benchmarks.py](advanced_benchmarks.py) |
| **Phase 4.5**| **KLD / Perplexity** | Measure information loss (KL Divergence, Perplexity, Same Top Token matching) of quantized KV caches. | Quant Loss | **COMPLETED** | [kld_benchmark.py](kld_benchmark.py) |
| **Phase 5** | **Unified Dashboard**| Regroup, unify running benchmarks, record historical runs, and plot comparative metrics. | WebUI / DB | **COMPLETED** | [dashboard.py](dashboard.py) |

---

## Phase 5: Unified Benchmarking & Visualization Suite (Plan)

To centralize all benchmark results and provide a state-of-the-art WebUI, we will build the following system:

### 1. Unified Run Orchestration (`run_suite.py`)
Create a single python entrypoint to execute all benchmarks:
*   Allows executing the original ORM throughput tests (`benchmark.sh`), the long-context reasoning tests (`advanced_benchmarks.py`), and the KLD perplexity tests (`kld_benchmark.py`).
*   Accepts customizable parameters (`--endpoint`, `--model`, `--tokens`, `--kv-quant`, etc.).

### 2. Historical Run Database & Storage (`history/`)
Establish a persistent storage system for benchmark runs:
*   **Storage Format**: Structured JSON records saved in `history/run_[timestamp].json`.
*   **Schema**:
    *   `run_metadata`: Timestamp, target endpoint, CLI arguments.
    *   `model_settings`: Model name, base quantization, KV cache quant (`q5_1`/`q8_0`/`f16`), threads, ubatch/batch sizes, speculative draft type.
    *   `throughput_metrics`: Prefill speed (t/s), Decode speed (t/s), TTFT (s).
    *   `reasoning_accuracy`: Needle (Pass/Fail), RULER (Pass/Fail), LongBench (Pass/Fail), SWE-bench (Pass/Fail).
    *   `quantization_loss`: Perplexity (PPL), Mean KLD, Same Top % matching.

### 3. Configurable WebUI Dashboard (`dashboard.py`)
Build a high-performance Streamlit WebUI to view and filter historical runs:
*   **Run History Browser**: A table summarizing all past runs with sorting.
*   **Advanced Filters Sidebar**: Filter runs by model name, endpoint, KV cache quant, context length, etc.
*   **Comparative Plotting**:
    *   *Throughput vs. Context Length*: Plot prefill/decode speeds as context grows.
    *   *Accuracy Comparison*: Bar charts comparing reasoning benchmark accuracy across configurations.
    *   *Quantization Trade-offs*: Plot KLD/Perplexity vs. VRAM savings (e.g. `f16` vs. `q8_0` vs. `q5_1`).
*   **Side-by-Side Model Comparison**: Compare two specific model runs side-by-side.

---

## Phase 6: Hardening, Test Coverage & Performance (Completed 2026-09-10)

- [x] **SSRF & Input Security**: Centralized validation in `utils.py` and pruned duplicate `url_validation.py` (PR #71).
- [x] **INI Configuration Caching**: Cached `model_presets.ini` parsing with `@lru_cache` in `dashboard.py` (PR #74).
- [x] **DataFrame Optimization**: Switched from `.iterrows()` to `.itertuples()` for chart data preparation (PR #73).
- [x] **Metric Formatting Hardening**: Protected `fmt_num` against array ambiguity and `pd.NA` errors (PR #76).
- [x] **Unit Testing Consolidation**: Added unit tests for `fmt_num`, `generate_filler_text` (PR #75), reasoning metrics, and `utils.py`.
- [x] **Clean Import Hierarchy**: Eliminated unused `re`, `urllib.parse`, and redundant Plotly imports (PR #84).

---

## Phase 7: Test Coverage Completion & Cache Optimization (Completed 2026-09-11)

- [x] **Run Matrix Test Suite**: 100% test coverage for `run_matrix.py` (PR #100 / Issue #94).
- [x] **Run Suite Test Suite**: 99% test coverage for `run_suite.py` (PR #101 / Issue #95).
- [x] **Populate History Test Suite**: 98% test coverage for `populate_history.py` (PR #97 / Issue #96).
- [x] **Dashboard Path & Corpus Validators**: Comprehensive tests for `validate_gguf_path` and `validate_corpus_name` with symlink and whitespace hardening (PR #98 / Issue #91).
- [x] **SSRF & Exception Paths Coverage**: 100% test coverage on `utils.py` with multi-IP resolution tests (PR #99 / Issue #92).
- [x] **Dashboard Run Loading Cache**: `@st.cache_data(ttl=60)` optimization for `load_runs` with fallback support (PR #104 / Issue #93).

---

## Phase 8: AST Sandbox Hardening, Presets Caching & UI Coverage (Completed 2026-09-11)

- [x] **AST Sandbox Security**: Comprehensive AST code execution sandboxing against `__import__`, `globals()`, unblocked modules, and reflection in `advanced_benchmarks.py` (PR #120 / Issue #109).
- [x] **Benchmark UI Test Suite**: 99% test coverage on `benchmark_ui.py` with full mocking of `list_models` and `run_benchmark_stream` (PR #123 / Issue #110).
- [x] **Model Presets LRU Caching**: Added `@functools.lru_cache` for INI config parsing and alias resolution in `advanced_benchmarks.py` and `run_matrix.py` (PR #117 / Issue #111).
- [x] **Input Bounds & GGUF Path Validation**: Added `new_tokens` bounds `[1, 262144]` and restricted `validate_gguf_path` to `.gguf` extension and model cache subdirectories in `dashboard.py` (PR #115 / Issue #112).
- [x] **Perplexity Binary Build Testing**: 100% test coverage for `compile_perplexity_binary` in `test_kld_benchmark.py` (PR #116 / Issue #113).

---

## Phase 9: CI Pipeline & Dependabot Automation (Completed 2026-09-12)

- [x] **Automated GitHub Actions CI**: Added `.github/workflows/ci.yml` running `pytest` with coverage on master and pull requests (Issue #124).
- [x] **Dependabot Automation Workflow**: Added `.github/workflows/dependabot.yml` for automated PR review, auto-approval, and auto-merge of minor and patch updates (Issue #124).
- [x] **Dependabot Configuration Alignment**: Updated `.github/dependabot.yml` to monitor `Containerfile` (`docker` ecosystem), added labels, commit prefixes, and grouping for dev dependencies (Issue #124).
- [x] **Dev Requirements**: Added `requirements-dev.txt` for development and test runner dependencies (Issue #124).

## Phase 10: Performance Optimization, Presets Normalization & Modular Test Suites (Completed 2026-09-14)

- [x] **Dashboard DataFrame Grouping Optimization**: Extracted `build_throughput_figure` with upfront Context Length sorting and `groupby(["Model", "KV Quant"])`, achieving ~4.2x faster chart plotting and eliminating redundant hover template/zip allocations (PR #133 / Issue #127).
- [x] **Model Presets Prefix Normalization & Path Fallbacks**: Added `_normalize_repo_id` and `resolve_presets_path` across `advanced_benchmarks.py`, `dashboard.py`, and `run_matrix.py`, supporting symmetrical unsloth/ matching and fallback detection of `model_presets.ini` (PR #134 / Issue #128).
- [x] **SSE Chunk Streaming Optimization & Test Suite**: Replaced string concatenation with list chunk accumulation in `call_endpoint`, added type checks, and added full unit test suite `TestCallEndpoint` (PR #135 / Issue #129).
- [x] **Modular Endpoint Settings Parser & Test Suite**: Extracted `_parse_endpoint_model_args` and `_parse_endpoint_preset_block` from `get_model_settings_from_endpoint` with comprehensive unit tests (PR #136 / Issue #130).
- [x] **Defensive Validation & Queue Reader Extraction**: Enforced numeric type checks on `generate_filler_text` and extracted `enqueue_output` to module scope in `dashboard.py` with 100% test coverage (PR #137 / Issue #131).
- [x] **Modular Run File Ingestion & Main Runner Tests**: Extracted `_parse_run_file` in `dashboard.py`, extracted runner helpers in `advanced_benchmarks.py`, added `--benchmark` selector and `--output` flags, and added comprehensive unit tests for `run_swe_test` and `main()` (PR #138 / Issue #132).

---

## Next Steps / Future Work
1. **Dynamic Live Telemetry**: Integrate GPU VRAM and temperature metrics directly into the Streamlit UI via NVML.
2. **Automated Export**: Add CSV/Excel export buttons for filtered historical benchmark sets.

