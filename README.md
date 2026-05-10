# LLM Benchmark UI

A lightweight Streamlit interface for testing and benchmarking LLM endpoints using the existing `benchmark.sh` script.

## Files in this directory

- `benchmark_ui.py` - Main Streamlit application
- `requirements.txt` - Python dependencies (Streamlit)
- `benchmark.sh` - The original benchmark script (symlinked or copied from parent directory)
- `run_ui.sh` - Convenience script to launch the UI
- `README.md` - This file

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Using the convenience script

```bash
./run_ui.sh
```

### Method 2: Manual launch

```bash
streamlit run benchmark_ui.py
```

### Method 3: Direct execution (if you prefer not to use Streamlit)

You can still use the original benchmark script directly:

```bash
../benchmark.sh "ModelName" "http://127.0.0.1:8081"
```

## Features

- **Model Listing**: Fetch and display available models from your LLM endpoint
- **Benchmark Execution**: Run the full benchmark suite with visual feedback
- **Results Display**: Clean, formatted output of benchmark metrics
- **Result Download**: Save benchmark results as text files
- **Configuration Sidebar**: Easy model and endpoint configuration

## Benchmark Details

The benchmark script performs four sequential tests:

1. **Turn 1**: Cold start -> Python ORMs (measures initial latency)
2. **Turn 2**: KV Cache Hit -> Repetitive Python ORMs (measures cached performance)
3. **Turn 3**: KV Cache Hit -> Repetitive JSON Tool Calls (measures structured output)
4. **Turn 4**: KV Cache Hit -> Repetitive Markdown (measures generation throughput)

For each test, it reports:
- Prompt and completion tokens
- Prompt evaluation speed (tokens/sec) and Time To First Token (TTFT)
- Generation speed (tokens/sec) and decode time

Finally, it attempts to pull metrics from the Prometheus endpoint (`/metrics?model=${MODEL}`) if available.

## Requirements

- Python 3.7+
- Streamlit (see requirements.txt)
- The original `benchmark.sh` script (must be in the same directory or parent directory)
- `jq` and `bc` command-line utilities (used by the benchmark script)
- `curl` for making HTTP requests

## Troubleshooting

If you encounter issues:

1. **Script not found**: Ensure `benchmark.sh` is present in the same directory as `benchmark_ui.py` or in the parent `LAB/IA/` directory.
2. **Permission denied**: Make sure `benchmark.sh` is executable (`chmod +x benchmark.sh`)
3. **Missing dependencies**: Install `jq` and `bc` via your package manager (e.g., `sudo apt install jq bc`)
4. **Endpoint unreachable**: Verify your LLM service is running and accessible at the provided URL
5. **Streamlit port conflicts**: The UI runs on port 8501 by default; change with `--server.port` if needed

## Customization

You can modify `benchmark_ui.py` to:
- Change the default model or endpoint
- Add additional input fields (e.g., API keys)
- Modify the layout or styling
- Add result parsing or visualization

## Notes

- The UI uses the existing benchmark script without modification, ensuring consistency with command-line usage.
- All benchmark output is captured and displayed verbatim for transparency.
- For long-running benchmarks, the UI shows a spinner and timeout protection (5 minutes).

---

Built for the LAB/IA/bench project.