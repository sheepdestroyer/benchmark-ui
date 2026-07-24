# Changelog

All notable changes to the LLM Benchmarking and Server Router project in this session are documented below.

## [2026-07-08] - Session Summary

### Added
*   **KLD Benchmarking Utility**: Created [kld_benchmark.py](file:///home/sheepdestroyer/LAB/IA/bench/kld_benchmark.py) to calculate and compare Kullback-Leibler (KL) Divergence, Perplexity (PPL), and top-token match rates of quantized KV caches (`q8_0`, `q5_1`, `q4_0`) against an `f16` baseline.
*   **Test Runner Script**: Created [run-server-tests.sh](file:///home/sheepdestroyer/LAB/IA/llama.cpp/run-server-tests.sh) to automate running server unit tests on a non-conflicting port (`58080`).
*   **Unit Tests**: Added `test_router_global_preset_inheritance` in [test_router.py](file:///home/sheepdestroyer/LAB/IA/llama.cpp/tools/server/tests/unit/test_router.py) to verify wildcard parameter inheritance and overrides.

### Optimized
*   **Asymmetrical Dual-GPU Presets**: Balanced weights and primary calculations across the RTX 3090 (24GB) and RTX 3080 (20GB). Swapped presets under `[*]` in [model_presets.ini](file:///home/sheepdestroyer/LAB/IA/llama.cpp/profiles/model_presets.ini) to align with CUDA's native performance-based device ordering:
    *   `main-gpu = 0` (RTX 3090 handles primary logit/prefill calculations).
    *   `tensor-split = 28,14` (Loads 28 parts of layers on the 3090 and 14 parts on the 3080).
*   **Build Automation**: Updated [build_llama.sh](file:///home/sheepdestroyer/LAB/IA/build_llama.sh) to compile the native `llama-perplexity` binary automatically alongside the server.

### Fixed
*   **Production-Grade Systemd Service**: Rewrote [llama-router.service](file:///home/sheepdestroyer/.config/systemd/user/llama-router.service) to resolve port collision races and unclean shutdowns:
    *   Removed hacky `ExecStop=/bin/pkill llama-server`.
    *   Enabled `KillMode=control-group` to let systemd manage the process tree naturally.
    *   Added `TimeoutStopSec=10` and `RestartSec=2` to ensure fast, reliable service restarts.
*   **GPU Environment Reordering**: Cleaned up the `CUDA_VISIBLE_DEVICES` hacks from launch scripts to avoid conflict with CUDA's default device sorting.
