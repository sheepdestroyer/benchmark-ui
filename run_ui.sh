#!/usr/bin/env bash
set -euo pipefail
# Convenience script to launch the Streamlit UI for LLM benchmarking

# Change to the directory where this script is located
cd "$(dirname "$0")" || exit 1

# Check if running in a virtual environment, activate if available
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    elif [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi
fi

# Check if streamlit is available, if not install it
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing from requirements.txt..."
    if [[ -f requirements.txt ]]; then
        pip install -r requirements.txt
    else
        echo "Error: requirements.txt not found." >&2
        exit 1
    fi
fi

# Launch the Streamlit app
echo "Launching LLM Benchmark UI..."
streamlit run dashboard.py "$@"