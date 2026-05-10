#!/usr/bin/env python3
"""
Streamlit UI for LLM Benchmarking
Interfaces with the existing benchmark.sh script to test and benchmark LLM endpoints.
With realtime streaming output.
"""

import streamlit as st
import subprocess
import os
import time
import threading
import queue
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="LLM Benchmark UI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BENCH_SCRIPT = Path(__file__).parent / "benchmark.sh"
WORK_DIR = Path(__file__).parent

def run_benchmark_stream(model, endpoint):
    """Run the benchmark script and yield lines in realtime."""
    if not BENCH_SCRIPT.exists():
        yield f"Error: Benchmark script not found at {BENCH_SCRIPT}"
        return
    
    if not os.access(BENCH_SCRIPT, os.X_OK):
        try:
            os.chmod(BENCH_SCRIPT, 0o755)
        except Exception as e:
            yield f"Error: Cannot make script executable: {e}"
            return
    
    cmd = [str(BENCH_SCRIPT), model, endpoint]
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORK_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in iter(proc.stdout.readline, ''):
        yield line
    
    proc.wait()
    if proc.returncode != 0:
        yield f"\n[EXIT CODE: {proc.returncode}]"

def list_models(endpoint):
    """List available models at the endpoint."""
    if not BENCH_SCRIPT.exists():
        return f"Error: Benchmark script not found at {BENCH_SCRIPT}"
    
    cmd = [str(BENCH_SCRIPT), "--list", endpoint]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORK_DIR),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n\nSTDERR:\n" + result.stderr
            
        if result.returncode != 0:
            output = f"Error (exit code {result.returncode}):\n{output}"
            
        return output
    except subprocess.TimeoutExpired:
        return "Error: List models timed out"
    except Exception as e:
        return f"Error listing models: {e}"

def main():
    """Main Streamlit application."""
    st.title("🚀 LLM Benchmark UI")
    st.markdown("Interface for testing and benchmarking LLM endpoints with realtime streaming output.")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        
        # Model input
        model = st.text_input(
            "Model Name",
            value="Qwen3.6-27B",
            help="Name of the model to benchmark (as recognized by the endpoint)"
        )
        
        # Endpoint input
        endpoint = st.text_input(
            "Endpoint URL",
            value="http://127.0.0.1:8081",
            help="Base URL of the LLM endpoint (e.g., http://127.0.0.1:8081)"
        )
        
        st.divider()
        
        # Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 List Models", use_container_width=True):
                st.session_state.list_models = True
                st.session_state.run_benchmark = False
        
        with col2:
            if st.button("▶️ Run Benchmark", use_container_width=True, type="primary"):
                st.session_state.run_benchmark = True
                st.session_state.list_models = False
                st.session_state.benchmark_lines = []
                st.session_state.benchmark_running = True
                st.session_state.benchmark_start = time.time()
    
    # Main area for output
    if st.session_state.get('list_models', False):
        with st.spinner("Fetching model list..."):
            output = list_models(endpoint)
        
        st.subheader("Available Models")
        st.code(output, language="bash")
    
    elif st.session_state.get('run_benchmark', False):
        st.subheader(f"📊 Benchmark Results for `{model}`")
        st.caption(f"Endpoint: {endpoint}")
        
        # Streaming output area
        output_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Accumulate all lines
        all_lines = []
        elapsed = 0
        
        for line in run_benchmark_stream(model, endpoint):
            all_lines.append(line)
            elapsed = time.time() - st.session_state.benchmark_start
            
            # Show live output with elapsed time
            full_output = "".join(all_lines)
            status_placeholder.caption(f"⏱️ Running... {elapsed:.1f}s elapsed")
            output_placeholder.code(full_output, language="bash")
        
        # Final display
        elapsed = time.time() - st.session_state.benchmark_start
        status_placeholder.caption(f"✅ Completed in {elapsed:.2f}s")
        
        full_output = "".join(all_lines)
        output_placeholder.code(full_output, language="bash")
        
        # Download button
        st.download_button(
            label="💾 Download Results",
            data=full_output,
            file_name=f"benchmark_{model.replace('.', '_')}_{int(time.time())}.txt",
            mime="text/plain"
        )
    
    else:
        # Default view - show instructions
        st.info("👈 Configure your model and endpoint in the sidebar, then choose an action.")
        
        st.markdown("""
        ### How to Use
        
        1. **Set Model Name**: Enter the model identifier as recognized by your LLM endpoint
        2. **Set Endpoint URL**: Enter the base URL of your LLM service (default: http://127.0.0.1:8081)
        3. **Choose Action**:
           - **List Models**: Fetch and display available models from the endpoint
           - **Run Benchmark**: Execute the full benchmark suite with live streaming output
        
        ### About the Benchmark
        
        The benchmark script performs four tests:
        - **Turn 1**: Cold start -> Python ORMs (measures initial latency)
        - **Turn 2**: KV Cache Hit -> Repetitive Python ORMs (measures cached performance)
        - **Turn 3**: KV Cache Hit -> Repetitive JSON Tool Calls (measures structured output)
        - **Turn 4**: KV Cache Hit -> Repetitive Markdown (measures generation throughput)
        
        For each test, it reports:
        - Prompt and completion tokens
        - Prompt evaluation speed (tokens/sec) and Time To First Token (TTFT)
        - Generation speed (tokens/sec) and decode time
        
        Finally, it pulls metrics from the Prometheus endpoint if available.
        """)
        
        # Show current directory info
        with st.expander("🔧 Technical Details"):
            st.text(f"Working directory: {WORK_DIR}")
            st.text(f"Benchmark script: {BENCH_SCRIPT}")
            st.text(f"Script exists: {BENCH_SCRIPT.exists()}")
            if BENCH_SCRIPT.exists():
                st.text(f"Script executable: {os.access(BENCH_SCRIPT, os.X_OK)}")

if __name__ == "__main__":
    # Initialize session state
    if 'list_models' not in st.session_state:
        st.session_state.list_models = False
    if 'run_benchmark' not in st.session_state:
        st.session_state.run_benchmark = False
    
    main()
