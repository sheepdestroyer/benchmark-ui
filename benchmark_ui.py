#!/usr/bin/env python3
"""
Streamlit UI for LLM Benchmarking
Interfaces with the existing benchmark.sh script to test and benchmark LLM endpoints.
With realtime streaming output.
"""

import streamlit as st
import uuid
import subprocess
import os
import time
import re
import pandas as pd
import plotly.express as px
from pathlib import Path
import urllib.parse
import ipaddress

UI_THROTTLE_INTERVAL = 0.1  # Seconds between UI updates during streaming

def validate_endpoint_url(url_str):
    if not url_str:
        raise ValueError("Endpoint URL cannot be empty.")
    parsed = urllib.parse.urlparse(url_str)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: missing hostname.")
    
    hostname_lower = hostname.lower()
    if hostname_lower == "localhost" or hostname == "127.0.0.1":
        return url_str
    
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or not ip.is_global:
            raise ValueError(f"Forbidden IP address range: {hostname}")
    except ValueError as e:
        if "Forbidden IP address range" in str(e):
            raise
        pass
    return url_str

def validate_model_name(model_name):
    if not model_name or not re.match(r'^[a-zA-Z0-9._:/-]+$', model_name):
        raise ValueError(f"Invalid model name '{model_name}'. Must match ^[a-zA-Z0-9._:/-]+$")
    return model_name

# Page configuration
st.set_page_config(
    page_title="LLM Benchmark UI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state defaults at top-level
if 'list_models' not in st.session_state:
    st.session_state.list_models = False
if 'run_benchmark' not in st.session_state:
    st.session_state.run_benchmark = False
if 'benchmark_history' not in st.session_state:
    st.session_state.benchmark_history = []
if 'benchmark_running' not in st.session_state:
    st.session_state.benchmark_running = False
if 'current_proc' not in st.session_state:
    st.session_state.current_proc = None
if 'benchmark_start' not in st.session_state:
    st.session_state.benchmark_start = time.time()

# Paths
BENCH_SCRIPT = Path(__file__).parent / "benchmark.sh"
WORK_DIR = Path(__file__).parent

def cleanup_current_proc():
    proc = st.session_state.get('current_proc')
    if proc is not None:
        try:
            if proc.stdout and not proc.stdout.closed:
                proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)
        st.session_state.current_proc = None

METRIC_PATTERNS = {
    "Prompt Tokens": (re.compile(r"Prompt Tokens\s+:\s+(\d+)"), int),
    "Completion Tokens": (re.compile(r"Completion Tokens\s+:\s+(\d+)"), int),
    "Prompt Eval (p/s)": (re.compile(r"Prompt Eval \(p/s\)\s+:\s+([\d.]+)"), float),
    "TTFT (s)": (re.compile(r"TTFT:\s+([\d.]+)"), float),
    "Generation (t/s)": (re.compile(r"Generation\s+\(t/s\)\s+:\s+([\d.]+)"), float),
    "Decode Time (s)": (re.compile(r"Decode:\s+([\d.]+)"), float),
}

def parse_benchmark_output(output_text):
    """Parse the benchmark output to extract metrics for each turn."""
    turns = []
    # Split by turn indicator
    turn_blocks = re.split(r'---> Running (Turn \d+ \([^)]+\))...', output_text)

    # turn_blocks[0] is everything before first turn
    # subsequent elements are (turn_name, turn_content, turn_name, turn_content, ...)
    for i in range(1, len(turn_blocks), 2):
        name = turn_blocks[i]
        content = turn_blocks[i+1] if i+1 < len(turn_blocks) else ""

        metrics = {
            "Turn": name,
            "Prompt Tokens": 0,
            "Completion Tokens": 0,
            "Prompt Eval (p/s)": 0.0,
            "TTFT (s)": 0.0,
            "Generation (t/s)": 0.0,
            "Decode Time (s)": 0.0
        }

        # Extract metrics using regex
        for metric_name, (pattern, cast_type) in METRIC_PATTERNS.items():
            match = pattern.search(content)
            if match:
                metrics[metric_name] = cast_type(match.group(1))

        turns.append(metrics)

    all_defaults = True
    if turns:
        for t in turns:
            if any(t[k] != 0 for k in ["Prompt Tokens", "Completion Tokens", "Prompt Eval (p/s)", "TTFT (s)", "Generation (t/s)", "Decode Time (s)"]):
                all_defaults = False
                break

    if all_defaults:
        st.warning("Warning: Benchmark output parsing produced only default metric values (0.0). Check execution logs for errors.")

    return turns

def run_benchmark_stream(model, endpoint):
    """Run the benchmark script and yield lines in realtime."""
    try:
        endpoint = validate_endpoint_url(endpoint)
        model = validate_model_name(model)
    except ValueError as e:
        yield f"Error: Invalid input: {e}"
        return

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
    st.session_state.current_proc = proc
    
    try:
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                yield line
        proc.wait()
        if proc.returncode != 0:
            yield f"\n[EXIT CODE: {proc.returncode}]"
    finally:
        if proc.stdout and not proc.stdout.closed:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)
        st.session_state.current_proc = None

def list_models(endpoint):
    """List available models at the endpoint."""
    try:
        endpoint = validate_endpoint_url(endpoint)
    except ValueError as e:
        return f"Error: Invalid endpoint URL: {e}"

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
    cleanup_current_proc()
    st.title("🚀 LLM Benchmark UI")
    st.markdown("Interface for testing and benchmarking LLM endpoints with realtime streaming output.")

    # Tabs for different views
    tab_run, tab_viz = st.tabs(["▶️ Run Benchmark", "📊 Visualizations"])
    
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
                cleanup_current_proc()
                st.session_state.list_models = True
                st.session_state.run_benchmark = False
        
        with col2:
            if st.button("▶️ Run Benchmark", use_container_width=True, type="primary"):
                cleanup_current_proc()
                st.session_state.run_benchmark = True
                st.session_state.list_models = False
                st.session_state.benchmark_lines = []
                st.session_state.benchmark_running = True
                st.session_state.benchmark_start = time.time()
    
    # Main area for output
    with tab_run:
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

            # Check if it was just started or already completed
            if st.session_state.get('benchmark_running', False):
                last_update_time = 0.0
                for line in run_benchmark_stream(model, endpoint):
                    all_lines.append(line)
                    current_time = time.monotonic()

                    if current_time - last_update_time >= UI_THROTTLE_INTERVAL:
                        elapsed = time.time() - st.session_state.get('benchmark_start', time.time())
                        # Show live output with elapsed time
                        full_output = "".join(all_lines)
                        status_placeholder.caption(f"⏱️ Running... {elapsed:.1f}s elapsed")
                        output_placeholder.code(full_output, language="bash")
                        last_update_time = current_time

                # Final display
                elapsed = time.time() - st.session_state.get('benchmark_start', time.time())
                status_placeholder.caption(f"✅ Completed in {elapsed:.2f}s")

                full_output = "".join(all_lines)
                output_placeholder.code(full_output, language="bash")

                # Store in history
                run_data = {
                    "id": f"{int(time.time())}_{uuid.uuid4().hex[:6]}",
                    "model": model,
                    "endpoint": endpoint,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": f"{elapsed:.2f}s",
                    "raw_output": full_output,
                    "turns": parse_benchmark_output(full_output)
                }
                st.session_state.benchmark_history.append(run_data)
                if len(st.session_state.benchmark_history) > 50:
                    st.session_state.benchmark_history = st.session_state.benchmark_history[-50:]
                st.session_state.benchmark_running = False
                st.session_state.last_benchmark_output = full_output
                st.success(f"Benchmark results saved to history! (Total runs: {len(st.session_state.benchmark_history)})")

            # If not running but run_benchmark is true, it means it just finished
            elif 'last_benchmark_output' in st.session_state:
                output_placeholder.code(st.session_state.last_benchmark_output, language="bash")
                status_placeholder.caption("✅ Completed")

            # Download button
            if 'last_benchmark_output' in st.session_state:
                st.download_button(
                    label="💾 Download Results",
                    data=st.session_state.last_benchmark_output,
                    file_name=f"benchmark_{re.sub(r'[^a-zA-Z0-9_-]', '_', model)}_{int(time.time())}.txt",
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

    with tab_viz:
        st.subheader("Performance Visualizations")
        if not st.session_state.benchmark_history:
            st.info("No benchmark history yet. Run a benchmark first to see visualizations.")
        else:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.benchmark_history = []
                    st.rerun()

            with col1:
                # Selection of runs
                options = [f"#{idx + 1}: {run['timestamp']} - {run['model']}" for idx, run in enumerate(st.session_state.benchmark_history)]
                selected_options = st.multiselect("Select Benchmark Runs to Compare", options, default=[options[-1]] if options else [])

            selected_runs = [run for idx, run in enumerate(st.session_state.benchmark_history)
                             if f"#{idx + 1}: {run['timestamp']} - {run['model']}" in selected_options]

            if not selected_runs:
                st.warning("Please select at least one benchmark run.")
            else:
                # Prepare data for plotting
                all_turns_data = []
                for run in selected_runs:
                    for turn in run['turns']:
                        turn_copy = turn.copy()
                        turn_copy['Model'] = run['model']
                        turn_copy['Run ID'] = f"{run['model']} ({run['timestamp']})"
                        all_turns_data.append(turn_copy)

                df = pd.DataFrame(all_turns_data)

                # Visualizations
                viz_col1, viz_col2 = st.columns([1, 3])

                with viz_col1:
                    st.write("### Settings")
                    chart_type = st.radio("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])

                    metrics_to_plot = ["Generation (t/s)", "Prompt Eval (p/s)", "TTFT (s)", "Decode Time (s)", "Prompt Tokens", "Completion Tokens"]
                    selected_metric = st.selectbox("Metric", metrics_to_plot)

                with viz_col2:
                    if df.empty or 'Turn' not in df.columns:
                        st.warning("No valid turn data available for visualization.")
                    elif chart_type == "Bar Chart":
                        fig = px.bar(df, x="Turn", y=selected_metric, color="Run ID", barmode="group",
                                     title=f"{selected_metric} Comparison")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Line Chart":
                        fig = px.line(df, x="Turn", y=selected_metric, color="Run ID", markers=True,
                                      title=f"{selected_metric} Trend across Turns")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Pie Chart":
                        if len(selected_runs) > 1:
                            st.info("Pie chart shows token distribution for the first selected run.")

                        run = selected_runs[0]
                        pie_data = []
                        for turn in run["turns"]:
                            pie_data.append({"Category": f"{turn['Turn']} (Prompt)", "Tokens": turn["Prompt Tokens"]})
                            pie_data.append({"Category": f"{turn['Turn']} (Completion)", "Tokens": turn["Completion Tokens"]})

                        pie_df = pd.DataFrame(pie_data)
                        if pie_df.empty or pie_df["Tokens"].sum() == 0:
                            st.warning("No token data available for Pie Chart.")
                        else:
                            fig = px.pie(pie_df, values="Tokens", names="Category",
                                         title=f"Token Distribution across Turns: {run['model']} ({run['timestamp']})")
                            st.plotly_chart(fig, use_container_width=True)

                # Show raw data
                with st.expander("View Comparative Data Table"):
                    st.dataframe(df)

if __name__ == "__main__":
    main()
