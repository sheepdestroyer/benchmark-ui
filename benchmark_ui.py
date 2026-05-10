#!/usr/bin/env python3
"""
Streamlit UI for LLM Benchmarking
Interfaces with the existing benchmark.sh script to test and benchmark LLM endpoints.
"""

import streamlit as st
import subprocess
import os
import time
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

def run_benchmark(model, endpoint):
    """Run the benchmark script and return output."""
    if not BENCH_SCRIPT.exists():
        return f"Error: Benchmark script not found at {BENCH_SCRIPT}"
    
    # Make sure script is executable
    if not os.access(BENCH_SCRIPT, os.X_OK):
        try:
            os.chmod(BENCH_SCRIPT, 0o755)
        except Exception as e:
            return f"Error: Cannot make script executable: {e}"
    
    cmd = [str(BENCH_SCRIPT), model, endpoint]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORK_DIR),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n\nSTDERR:\n" + result.stderr
            
        if result.returncode != 0:
            output = f"Error (exit code {result.returncode}):\n{output}"
            
        return output
    except subprocess.TimeoutExpired:
        return "Error: Benchmark timed out after 5 minutes"
    except Exception as e:
        return f"Error running benchmark: {e}"

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
    st.markdown("Interface for testing and benchmarking LLM endpoints using the existing benchmark script.")
    
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
    
    # Main area for output
    if st.session_state.get('list_models', False):
        with st.spinner("Fetching model list..."):
            output = list_models(endpoint)
        
        st.subheader("Available Models")
        st.code(output, language="bash")
        
        # Provide a way to copy the output
        if st.button("📋 Copy Output"):
            st.write("Output copied to clipboard! (Note: actual clipboard functionality requires streamlit-clipy)")
    
    elif st.session_state.get('run_benchmark', False):
        with st.spinner("Running benchmark... This may take several minutes."):
            start_time = time.time()
            output = run_benchmark(model, endpoint)
            elapsed_time = time.time() - start_time
        
        st.subheader(f"Benchmark Results for `{model}`")
        st.caption(f"Endpoint: {endpoint} | Elapsed time: {elapsed_time:.2f}s")
        
        # Display output in a nice container
        with st.container(border=True):
            st.code(output, language="bash")
        
        # Provide download option
        st.download_button(
            label="💾 Download Results",
            data=output,
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
           - **Run Benchmark**: Execute the full benchmark suite for the specified model
        
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