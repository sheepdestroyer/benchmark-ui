# Changelog

All notable changes to the LLM Benchmarking and Server Router project are documented below.

## [v0.1.2] - 2026-09-11 - Test Suites, Caching & Validator Hardening

### Added
*   **Unit Tests for `run_matrix.py` (`test_run_matrix.py`)**: 100% coverage suite testing `TeeLogger` I/O buffering, non-zero exit handling in `restart_router_service`, run identifier matching, corrupt run log parsing, and ISO timestamp error logging (PR #100 / Issue #94).
*   **Unit Tests for `run_suite.py` (`test_run_suite.py`)**: 99% coverage suite testing model settings endpoint querying, connection timeout/error fallback handling, malformed argument parsing, quantization priority resolution, and atomic history writing (PR #101 / Issue #95).
*   **Unit Tests for `populate_history.py` (`test_populate_history.py`)**: 98% coverage suite validating synthetic history file generation, schema integrity, and `--force` cleanup safety in isolated temporary directories (PR #97 / Issue #96).
*   **Unit Tests for Validators (`test_dashboard.py`)**: Added `TestDashboardValidators` validating `validate_gguf_path` existence/file checks, symlink resolution, directory traversal protection against allowed parent bounds, and `validate_corpus_name` basename sanitization (PR #98 / Issue #91).
*   **Expanded Security & Exception Testing (`test_utils.py`)**: Achieved 100% test coverage on `utils.py`, testing empty/whitespace URLs, unsupported schemes, port boundary validations, multi-IP DNS resolution blocking for SSRF, and private IPv6/IPv4 address ranges (PR #99 / Issue #92).

### Optimized
*   **Dashboard Run History Caching**: Added `@st.cache_data(ttl=60)` with fallback support to `load_runs` in `dashboard.py`, significantly improving responsiveness during UI reruns and filtering while automatically discovering new runs within 60 seconds (PR #104 / Issue #93).

### Fixed & Hardened
*   **Corpus Name Whitespace Sanitization**: Hardened `validate_corpus_name` in `dashboard.py` to reject whitespace-only strings and strip leading/trailing whitespace from sanitized basenames, preventing downstream `FileNotFoundError` (PR #98).

## [v0.1.1] - 2026-09-10 - Quality Hardening, Test Consolidation & Performance

### Added
*   **Unit Tests for `fmt_num`**: Added comprehensive test cases in `test_dashboard.py` covering formatting of valid floats, integers, negative values, string fallbacks, NaN/Inf, pandas `pd.NA` identity, booleans, and sequences (PR #76 / Issue #81).
*   **Unit Tests for `generate_filler_text`**: Added test suite in `test_advanced_benchmarks.py` validating word count, token estimation, non-emptiness, and repetition behavior (PR #75 / Issue #82).
*   **Unit Tests for `extract_reasoning_acc_data`**: Added edge-case test suite in `test_dashboard.py` asserting graceful degradation on missing required columns, missing test suites, empty DataFrames, and duplicate column names.
*   **Comprehensive `test_utils.py` Suite**: Expanded test coverage for `utils.py` to 97%, validating SSRF protections, loopback allowance, IPv6/IPv4 private subnet filtering, model name validation, corpus name validation, and GGUF path validation.

### Optimized
*   **INI Configuration Parsing Caching**: Implemented `@functools.lru_cache(maxsize=1)` for `_get_presets_config` in `dashboard.py` to cache parsed `model_presets.ini` configurations, eliminating redundant disk I/O across `map_repo_to_preset_alias` and `get_preset_metadata` (PR #74 / Issue #80).
*   **DataFrame Row Iteration**: Replaced high-overhead `.iterrows()` with `.itertuples(index=False, name=None)` in `dashboard.py` for chart data extraction, significantly improving chart rendering performance for large historical benchmark sets (PR #73 / Issue #78).

### Fixed & Hardened
*   **`fmt_num` Robustness**: Hardened `fmt_num` against array ambiguity exceptions (`ValueError: The truth value of an array with more than one element is ambiguous`), `pd.NA` equality evaluation errors, and handled NaN, Inf, and boolean values cleanly.
*   **Reasoning Benchmark Extraction Hardening**: Refactored chart data extraction into `extract_reasoning_acc_data(df)` with scalar loc resolution, preventing `KeyError` or duplicate-column boolean mask crashes.
*   **Test Isolation & Background Process Prevention**: Configured `StreamlitMock.button` in `test_dashboard.py` to return `False`, preventing test module imports from accidentally spawning background `run_suite.py` benchmark processes.

### Removed
*   **Dead Code Elimination**: Deleted redundant `url_validation.py` duplicate file in favor of centralized `utils.py` (PR #71 / Issue #79).
*   **Unused Imports**: Cleaned up unused top-level `urllib.parse` and `re` imports, and redundant inner `plotly.graph_objects` import in `dashboard.py` (PR #84 / Issue #83).

### Dependencies
*   **Requests Upgrade**: Bumped `requests` dependency to `>=2.34.2` (PR #68).
*   **Streamlit Upgrade**: Bumped `streamlit` dependency to `>=1.63.0` (PR #77).

## [2026-07-08] - Session Summary

### Added
*   **KLD Benchmarking Utility**: Created [kld_benchmark.py](kld_benchmark.py) to calculate and compare Kullback-Leibler (KL) Divergence, Perplexity (PPL), and top-token match rates of quantized KV caches (`q8_0`, `q5_1`, `q4_0`) against an `f16` baseline.
*   **Test Runner Script**: Created [run-server-tests.sh](llama.cpp/run-server-tests.sh) to automate running server unit tests on a non-conflicting port (`58080`).
*   **Unit Tests**: Added `test_router_global_preset_inheritance` in [test_router.py](llama.cpp/tools/server/tests/unit/test_router.py) to verify wildcard parameter inheritance and overrides.

### Optimized
*   **Asymmetrical Dual-GPU Presets**: Balanced weights and primary calculations across the RTX 3090 (24GB) and RTX 3080 (20GB). Swapped presets under `[*]` in [model_presets.ini](llama.cpp/profiles/model_presets.ini) to align with CUDA's native performance-based device ordering:
    *   `main-gpu = 0` (RTX 3090 handles primary logit/prefill calculations).
    *   `tensor-split = 28,14` (Loads 28 parts of layers on the 3090 and 14 parts on the 3080).
*   **Build Automation**: Updated [build_llama.sh](build_llama.sh) to compile the native `llama-perplexity` binary automatically alongside the server.

### Fixed
*   **Production-Grade Systemd Service**: Rewrote [llama-router.service](llama-router.service) to resolve port collision races and unclean shutdowns:
    *   Removed hacky `ExecStop=/bin/pkill llama-server`.
    *   Enabled `KillMode=control-group` to let systemd manage the process tree naturally.
    *   Added `TimeoutStopSec=10` and `RestartSec=2` to ensure fast, reliable service restarts.
*   **GPU Environment Reordering**: Cleaned up the `CUDA_VISIBLE_DEVICES` hacks from launch scripts to avoid conflict with CUDA's default device sorting.
