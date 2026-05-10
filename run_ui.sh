#!/usr/bin/env bash
# Convenience script to launch the Streamlit UI for LLM benchmarking

# Change to the directory where this script is located
cd "$(dirname "$0")" || exit 1

# Check if streamlit is available, if not install it
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing from requirements.txt..."
    pip install -r requirements.txt
fi

# Launch the Streamlit app
echo "Launching LLM Benchmark UI..."
streamlit run benchmark_ui.py "$@"