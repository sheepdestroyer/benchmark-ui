#!/usr/bin/env python3
import urllib.parse
import re
import streamlit as st
import functools
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import time
import requests
import queue
import threading
from pathlib import Path
from utils import validate_endpoint_url, validate_model_name

BASE_QUANT_TYPES = ("Q4_K_XL", "Q6_K_XL", "Q4_K_S", "Q8_0", "Q5_1", "Q4_0", "F16", "Q5_K_M")
BASE_QUANT_ALIASES = tuple((q.lower(), q) for q in BASE_QUANT_TYPES)



def validate_gguf_path(gguf_path_str):
    if not gguf_path_str:
        return gguf_path_str
    resolved = Path(gguf_path_str).resolve()
    if not resolved.exists():
        raise ValueError(f"GGUF file path does not exist: {gguf_path_str}")
    if not resolved.is_file():
        raise ValueError(f"GGUF path is not a file: {gguf_path_str}")
    
    allowed_parents = [
        Path.cwd().resolve(),
        Path(__file__).parent.resolve(),
        Path.home().resolve()
    ]
    is_allowed = False
    for parent in allowed_parents:
        try:
            resolved.relative_to(parent)
            is_allowed = True
            break
        except ValueError:
            continue
    if not is_allowed:
        raise ValueError(f"GGUF path escapes allowed parent directories: {gguf_path_str}")
    return str(resolved)

def validate_corpus_name(corpus_str):
    if not corpus_str:
        raise ValueError("Corpus name cannot be empty.")
    safe_name = os.path.basename(corpus_str)
    if not safe_name or safe_name in ('.', '..'):
        raise ValueError(f"Invalid corpus name: {corpus_str}")
    return safe_name


# Page config
st.set_page_config(
    page_title="LLM Benchmarking Registry & Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

TEST_SUITES = ("Needle", "RULER", "LongBench", "SWE-bench")
PASS_FAIL_STATUSES = frozenset({"Pass", "Fail"})

# VRAM savings helper
VRAM_SAVINGS = {
    "f16": 0.0, "F16": 0.0,
    "q8_0": 50.0, "Q8_0": 50.0,
    "q5_1": 68.0, "Q5_1": 68.0,
    "q4_0": 75.0, "Q4_0": 75.0,
    "q4_k_m": 75.0, "Q4_K_M": 75.0,
    "q5_k_m": 68.0, "Q5_K_M": 68.0,
    "q4_k_s": 75.0, "Q4_K_S": 75.0,
    "q5_k_s": 68.0, "Q5_K_S": 68.0,
    "q8_k_m": 50.0, "Q8_K_M": 50.0,
    "Unknown": 0.0, "UNKNOWN": 0.0
}

def fmt_num(val, fmt="{:.2f}"):
    if val is None or pd.isna(val) or val == "N/A":
        return "N/A"
    try:
        return fmt.format(float(val))
    except (ValueError, TypeError):
        return str(val)

# Inject premium CSS
st.markdown("""
    <style>
        .main {
            background-color: #0f111a;
            color: #e2e8f0;
        }
        .stMetric {
            background-color: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #38bdf8 !important;
        }
        .header-gradient {
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        .card {
            background-color: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)



@functools.lru_cache(maxsize=1)
def _get_presets_config():
    import os
    presets_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini"))
    if not os.path.exists(presets_file):
        return None
        
    try:
        import configparser
        config = configparser.ConfigParser(strict=False)
        config.read(presets_file)
        return config
    except Exception:
        return None

def map_repo_to_preset_alias(repo_or_id):
    config = _get_presets_config()
    if not config:
        return repo_or_id

    try:
        
        # Exact section check first
        for section in config.sections():
            if section.lower() == repo_or_id.lower():
                return section
                
        # Fallback substring checks
        for section in config.sections():
            if section == "*":
                continue
            section_repo = config.get(section, "hf-repo", fallback="")
            section_alias = config.get(section, "alias", fallback="")
            
            if section_repo and section_repo.lower() in repo_or_id.lower():
                if "mtp" in repo_or_id.lower() and "spec" in section.lower():
                    return section
                if "mtp" not in repo_or_id.lower() and "spec" not in section.lower():
                    return section
                    
            if section_alias and section_alias.lower() in repo_or_id.lower():
                return section
    except (configparser.Error, OSError):
        pass
        
    # Fallback overrides
    if "qwen3.6-27b-gguf:q4_k_s" in repo_or_id.lower():
        return "Qwen3.6-27B"
    elif "qwen3.6-27b-mtp-gguf:q4_k_s" in repo_or_id.lower():
        return "Qwen3.6-27B-spec3"
    elif "qwen3.6-35b-a3b-gguf:q4_k_s" in repo_or_id.lower():
        return "Qwen3.6-35B-A3B"
    elif "gemma-4" in repo_or_id.lower():
        return "gemma4-26a4b-routing"
        
    return repo_or_id

def get_preset_metadata(profile_name):
    metadata = {
        "spec_type": "None",
        "spec_draft_type_k": "None",
        "spec_draft_type_v": "None",
        "flash_attn": "true",
        "parallel": "1",
        "n_gpu_layers": "99",
        "fit": "true"
    }
    
    config = _get_presets_config()
    if config:
        try:
            
            # Load globals if they exist
            if "*" in config.sections():
                for key in config["*"]:
                    clean_key = key.replace("-", "_")
                    metadata[clean_key] = config["*"][key]
                    
            # Load specific section
            if profile_name in config.sections():
                for key in config[profile_name]:
                    clean_key = key.replace("-", "_")
                    metadata[clean_key] = config[profile_name][key]
        except (configparser.Error, OSError):
            pass
            
    return metadata


# Load all runs
def load_runs():
    runs = []
    for filepath in HISTORY_DIR.glob("run_*.json"):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                
            metadata = data.get("run_metadata", {})
            settings = data.get("model_settings", {})
            throughput = data.get("throughput_metrics", {})
            accuracy = data.get("reasoning_accuracy", {})
            loss = data.get("quantization_loss", {})
            
            # Parse context length
            args = metadata.get("cli_arguments") or []
            ctx_len = 200000
            for i, arg in enumerate(args):
                if arg == "--tokens" and i + 1 < len(args):
                    try:
                        ctx_len = int(args[i+1])
                    except ValueError:
                        pass
            
            # Prefer explicit profile_alias from log first, fallback to model_name resolution
            profile_name = settings.get("profile_alias")
            if not profile_name or "/" in str(profile_name) or str(profile_name).endswith(".gguf"):
                profile_name = map_repo_to_preset_alias(settings.get("model_name", "Unknown"))
                
            # If profile_name is still a path or has a raw gguf filename, clean it
            if "/" in str(profile_name) or str(profile_name).endswith(".gguf"):
                name = str(profile_name).split("/")[-1]
                if name.endswith(".gguf"):
                    name = name[:-5]
                for q in ["-Q4_K_S", "-UD-Q4_K_XL", "-UD-Q4_K_S", "-Q6_K_XL", "-Q8_0", "-F16"]:
                    name = name.replace(q, "")
                if name.endswith("-UD"):
                    name = name[:-3]
                profile_name = name
                
            presets_meta = get_preset_metadata(profile_name)
            
            # Resolve Base Quant format
            base_quant = settings.get("base_quantization", "Unknown")
            if not base_quant or base_quant == "Unknown":
                model_name = settings.get("model_name", "")
                if ":" in str(model_name):
                    base_quant = str(model_name).split(":")[-1]
                else:
                    model_name_lower = str(model_name).lower()
                    for q_lower, q in BASE_QUANT_ALIASES:
                        if q_lower in model_name_lower:
                            base_quant = q
                            break
                if not base_quant or base_quant == "Unknown":
                    if "spec4" in str(profile_name).lower():
                        base_quant = "Q6_K_XL"
                    else:
                        base_quant = "Q4_K_S"
            
            spec_type = presets_meta.get("spec_type")
            if not spec_type or spec_type == "None":
                spec_type = settings.get("spec_type")
            if not spec_type or spec_type == "None":
                spec_type = settings.get("speculative_draft_type", "None")
                
            spec_k = presets_meta.get("spec_draft_type_k")
            if not spec_k or spec_k == "None":
                spec_k = settings.get("spec_draft_type_k", "None")
                
            spec_v = presets_meta.get("spec_draft_type_v")
            if not spec_v or spec_v == "None":
                spec_v = settings.get("spec_draft_type_v", "None")
            flash_attn = settings.get("flash_attn", presets_meta.get("flash_attn", "true"))
            parallel = settings.get("parallel", presets_meta.get("parallel", "1"))
            fit = settings.get("fit", presets_meta.get("fit", "true"))
            
            runs.append({
                "Filename": filepath.name,
                "Timestamp": metadata.get("timestamp", "Unknown"),
                "Endpoint": metadata.get("target_endpoint", "Unknown"),
                "Model": profile_name,
                "Base Quant": base_quant,
                "KV Quant": settings.get("kv_cache_quant", "Unknown"),
                "Threads": settings.get("threads"),
                "Ubatch Size": settings.get("ubatch_size"),
                "Batch Size": settings.get("batch_size"),
                "Speculative": spec_type,
                
                # Preset metadata fields
                "Spec Type": spec_type,
                "Spec Draft Type K": spec_k,
                "Spec Draft Type V": spec_v,
                "Flash Attn": flash_attn,
                "Parallel": parallel,
                "Fit": fit,
                
                "Prefill (t/s)": throughput.get("prefill_speed"),
                "Decode (t/s)": throughput.get("decode_speed"),
                "TTFT (s)": throughput.get("ttft"),
                "Needle": accuracy.get("needle", "N/A"),
                "RULER": accuracy.get("ruler", "N/A"),
                "LongBench": accuracy.get("longbench", "N/A"),
                "SWE-bench": accuracy.get("swe_bench", "N/A"),
                "PPL": loss.get("perplexity"),
                "KLD": loss.get("mean_kld"),
                "Same Top %": loss.get("same_top_match_percent"),
                "Context Length": ctx_len,
            })
        except Exception as e:
            st.error(f"Error loading {filepath.name}: {e}")
            
    df = pd.DataFrame(runs)
    if not df.empty:
        df["KLD"] = pd.to_numeric(df["KLD"], errors="coerce")
        df["PPL"] = pd.to_numeric(df["PPL"], errors="coerce")
        df = df.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
    return df

df = load_runs()

# Sidebar filter implementation
with st.sidebar:
    st.markdown("## 📊")
    st.markdown("### <span class='header-gradient'>Dashboard Filters</span>", unsafe_allow_html=True)
    
    if not df.empty:
        # Endpoint filter
        endpoints = sorted(list(df["Endpoint"].dropna().unique()))
        selected_endpoints = st.multiselect("Endpoints", endpoints, default=endpoints)
        
        # Model filter
        models = sorted(list(df["Model"].dropna().unique()))
        selected_models = st.multiselect("Models", models, default=models)
        
        # KV cache quant filter
        quants = sorted(list(df["KV Quant"].dropna().unique()))
        selected_quants = st.multiselect("KV Cache Quants", quants, default=quants)
        
        # Spec Type filter
        spec_types = sorted(list(df["Spec Type"].dropna().unique()))
        selected_spec_types = st.multiselect("Spec Type", spec_types, default=spec_types)
        
        # Spec Draft Type K filter
        spec_drafts = sorted(list(df["Spec Draft Type K"].dropna().unique()))
        selected_spec_drafts = st.multiselect("Spec Draft Type K", spec_drafts, default=spec_drafts)
        
        # Context length filter
        ctx_lens = sorted([int(x) for x in df["Context Length"].dropna().unique()])
        if ctx_lens:
            selected_ctx = st.multiselect("Context Lengths", ctx_lens, default=ctx_lens)
        else:
            selected_ctx = []
            
        # Filter dataframe
        filtered_df = df[
            df["Endpoint"].isin(selected_endpoints) &
            df["Model"].isin(selected_models) &
            df["KV Quant"].isin(selected_quants) &
            df["Spec Type"].isin(selected_spec_types) &
            df["Spec Draft Type K"].isin(selected_spec_drafts)
        ]
        if selected_ctx:
            filtered_df = filtered_df[filtered_df["Context Length"].isin(selected_ctx)]
    else:
        st.warning("No runs found in the registry database.")
        filtered_df = df

st.markdown("# 🚀 <span class='header-gradient'>llama.cpp Benchmark Registry & Optimizer</span>", unsafe_allow_html=True)
st.markdown("Phase 5 visualizer dashboard for multi-GPU performance, quantization loss trade-offs, and reasoning capabilities.")

# KPI metrics
if not filtered_df.empty:
    is_kld_file = filtered_df["Filename"].str.contains(r"_(?:f16|q8_0|q5_1|q4_0)\.json$")
    unified_kpi_df = filtered_df[~is_kld_file]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Registry Runs", len(unified_kpi_df))
    with col2:
        best_prefill = filtered_df["Prefill (t/s)"].dropna().max()
        st.metric("Max Prefill Speed", f"{fmt_num(best_prefill, '{:.1f}')} t/s" if fmt_num(best_prefill, '{:.1f}') != "N/A" else "N/A")
    with col3:
        best_decode = filtered_df["Decode (t/s)"].dropna().max()
        st.metric("Max Decode Speed", f"{fmt_num(best_decode, '{:.1f}')} t/s" if fmt_num(best_decode, '{:.1f}') != "N/A" else "N/A")
    with col4:
        kld_numeric = pd.to_numeric(filtered_df["KLD"], errors='coerce')
        min_kld = kld_numeric[kld_numeric > 0].min()
        st.metric("Best non-zero KLD", fmt_num(min_kld, "{:.6f}") if fmt_num(min_kld, "{:.6f}") != "N/A" else "0.000000")

# Tabs
tab_history, tab_plots, tab_compare, tab_run = st.tabs([
    "🗂️ Run History Browser", 
    "📈 Comparative Plots", 
    "⚖️ Side-by-Side Model Comparison",
    "⚙️ Run New Benchmark"
])

with tab_history:
    st.subheader("Historical Benchmark Runs")
    if not filtered_df.empty:
        # Filter out the raw KLD individual files from history tables (keep only unified complete runs)
        is_kld_file = filtered_df["Filename"].str.contains(r"_(?:f16|q8_0|q5_1|q4_0)\.json$")
        unified_df = filtered_df[~is_kld_file]
        
        # Grouped Summary table
        st.markdown("### 📊 Profile & Quantization Summary (Grouped Averages)")
        summary_cols = ["Model", "Base Quant", "KV Quant", "Spec Type", "Spec Draft Type K", "Context Length", "Prefill (t/s)", "Decode (t/s)", "TTFT (s)", "PPL", "KLD"]
        grouped_df = unified_df[summary_cols].groupby(["Model", "Base Quant", "KV Quant", "Spec Type", "Spec Draft Type K", "Context Length"]).mean().reset_index()
        
        # Calculate reasoning pass rates for the group
        acc_cols = ["Model", "Base Quant", "KV Quant", "Spec Type", "Spec Draft Type K", "Context Length", "Needle", "RULER", "LongBench", "SWE-bench"]
        acc_group = unified_df[acc_cols].copy()
        for col in ["Needle", "RULER", "LongBench", "SWE-bench"]:
            acc_group[col] = acc_group[col].map({"Pass": 1.0, "Fail": 0.0})
        acc_grouped = acc_group.groupby(["Model", "Base Quant", "KV Quant", "Spec Type", "Spec Draft Type K", "Context Length"]).mean().reset_index()
        
        merged_grouped = pd.merge(grouped_df, acc_grouped, on=["Model", "Base Quant", "KV Quant", "Spec Type", "Spec Draft Type K", "Context Length"])
        rename_dict = {
            "Prefill (t/s)": "Avg Prefill (t/s)",
            "Decode (t/s)": "Avg Decode (t/s)",
            "TTFT (s)": "Avg TTFT (s)",
            "PPL": "Avg PPL",
            "KLD": "Avg KLD",
            "Needle": "Needle Pass Rate",
            "RULER": "RULER Pass Rate",
            "LongBench": "LongBench Pass Rate",
            "SWE-bench": "SWE-bench Pass Rate"
        }
        merged_grouped = merged_grouped.rename(columns=rename_dict)
        
        st.dataframe(
            merged_grouped.style.format({
                "Avg Prefill (t/s)": "{:.2f}",
                "Avg Decode (t/s)": "{:.2f}",
                "Avg TTFT (s)": "{:.3f}",
                "Avg PPL": "{:.4f}",
                "Avg KLD": "{:.6f}",
                "Needle Pass Rate": "{:.0%}",
                "RULER Pass Rate": "{:.0%}",
                "LongBench Pass Rate": "{:.0%}",
                "SWE-bench Pass Rate": "{:.0%}"
            }),
        )
        
        st.markdown("### 🗂️ Detailed Flat Logs")
        
        # Display clean browser dataframe
        display_cols = [
            "Timestamp", "Model", "Base Quant", "KV Quant", "Prefill (t/s)", "Decode (t/s)", 
            "TTFT (s)", "Needle", "RULER", "LongBench", "SWE-bench", "PPL", "KLD"
        ]
        st.dataframe(
            unified_df[display_cols].style.format({
                "Prefill (t/s)": "{:.2f}",
                "Decode (t/s)": "{:.2f}",
                "TTFT (s)": "{:.3f}",
                "PPL": "{:.4f}",
                "KLD": "{:.6f}"
            }),
        )
    else:
        st.info("No runs match the filter criteria.")

with tab_plots:
    st.subheader("Performance & Quantization Trade-off Analysis")
    if not filtered_df.empty:
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("#### Throughput (t/s) vs. KV Cache Quantization")
            # Filter rows with throughput values
            tp_df = filtered_df.dropna(subset=["Prefill (t/s)", "Decode (t/s)"])
            if not tp_df.empty:
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go
                
                # Create subplot figure with secondary y-axis
                fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                
                model_colors = {
                    "Qwen3.6-27B": "#3b82f6",          # Blue
                    "Qwen3.6-27B-spec3": "#10b981",    # Green
                    "Qwen3.6-27B-spec4": "#8b5cf6",    # Purple
                    "Qwen3.6-35B-A3B-spec": "#f97316", # Orange
                    "Qwen3.6-35B-A3B": "#ef4444"       # Red
                }
                
                # Sort unique models so they appear consistently
                unique_models = sorted(list(tp_df["Model"].unique()))
                
                for model in unique_models:
                    m_df = tp_df[tp_df["Model"] == model]
                    color = model_colors.get(model, "#94a3b8")
                    
                    # Custom hover template
                    hover_template_pp = (
                        "<b>%{customdata[0]}</b> (PP)<br>"
                        "Context: %{x} tokens<br>"
                        "PP (Prefill): %{y:.2f} t/s<br>"
                        "KV Cache: %{customdata[1]}<br>"
                        "<extra></extra>"
                    )
                    hover_template_tg = (
                        "<b>%{customdata[0]}</b> (TG)<br>"
                        "Context: %{x} tokens<br>"
                        "TG (Decode): %{y:.2f} t/s<br>"
                        "KV Cache: %{customdata[1]}<br>"
                        "<extra></extra>"
                    )
                    
                    unique_quants = sorted(list(m_df["KV Quant"].dropna().unique()))
                    for idx, quant in enumerate(unique_quants):
                        q_df = m_df[m_df["KV Quant"] == quant].sort_values(by="Context Length")
                        if q_df.empty:
                            continue
                            
                        customdata = list(zip(q_df["Model"], q_df["KV Quant"]))
                        show_legend = (idx == 0) # Show in legend only once per model
                        
                        # Select dash style based on quant format
                        dash_style = "solid" if quant == "f16" else ("dash" if quant == "q8_0" else ("dot" if quant == "q5_1" else "dashdot"))
                        
                        # Add PP (Prefill) trace - Left Y-axis (secondary_y=False)
                        fig1.add_trace(
                            go.Scatter(
                                x=q_df["Context Length"],
                                y=q_df["Prefill (t/s)"],
                                mode="lines+markers",
                                name=f"{model} (PP)",
                                legendgroup=f"{model}_PP",
                                showlegend=show_legend,
                                marker=dict(
                                    symbol="square",
                                    size=10,
                                    color=color,
                                    opacity=0.8,
                                    line=dict(width=1, color="#1e293b")
                                ),
                                line=dict(
                                    color=color,
                                    width=1.5,
                                    dash=dash_style
                                ),
                                customdata=customdata,
                                hovertemplate=hover_template_pp
                            ),
                            secondary_y=False
                        )
                        
                        # Add TG (Decode) trace - Right Y-axis (secondary_y=True)
                        fig1.add_trace(
                            go.Scatter(
                                x=q_df["Context Length"],
                                y=q_df["Decode (t/s)"],
                                mode="lines+markers",
                                name=f"{model} (TG)",
                                legendgroup=f"{model}_TG",
                                showlegend=show_legend,
                                marker=dict(
                                    symbol="circle",
                                    size=10,
                                    color=color,
                                    opacity=0.8,
                                    line=dict(width=1, color="#1e293b")
                                ),
                                line=dict(
                                    color=color,
                                    width=1.5,
                                    dash=dash_style
                                ),
                                customdata=customdata,
                                hovertemplate=hover_template_tg
                            ),
                            secondary_y=True
                        )
                
                # Update layout, axes titles, log scale, and dark template
                fig1.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    title="Read/Write (PP vs TG) Generation Speeds across Cache Formats",
                    xaxis_title="Context Length (tokens, log scale)",
                    xaxis_type="log"
                )
                
                # Left Y-axis (PP)
                fig1.update_yaxes(title_text="Prompt Processing (PP) Speed (tokens/sec)", secondary_y=False)
                # Right Y-axis (TG)
                fig1.update_yaxes(title_text="Token Generation (TG) Speed (tokens/sec)", secondary_y=True)
                
                st.plotly_chart(fig1)
            else:
                st.info("No throughput metrics available for plots.")
                
        with col_plot2:
            st.markdown("#### Reasoning Benchmarks Pass Rates")
            # Map Pass/Fail/NA to numeric values for bar charting
            acc_data = []
            for _, row in filtered_df.iterrows():
                for test in TEST_SUITES:
                    val = row[test]
                    if val in PASS_FAIL_STATUSES:
                        acc_data.append({
                            "Model_Quant": f"{row['Model']} ({row['KV Quant']})",
                            "Test Suite": test,
                            "Score": 1.0 if val == "Pass" else 0.0
                        })
            if acc_data:
                acc_df = pd.DataFrame(acc_data)
                # Plot summary pass rates
                fig2 = px.bar(
                    acc_df.groupby(["Model_Quant", "Test Suite"]).mean().reset_index(),
                    x="Test Suite",
                    y="Score",
                    color="Model_Quant",
                    barmode="group",
                    title="Needle, RULER, LongBench, SWE-bench Scores",
                    labels={"Score": "Pass Rate (0 or 1)"}
                )
                fig2.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2)
            else:
                st.info("No reasoning accuracy data available for plots.")
                
        st.divider()
        
        col_plot3, col_plot4 = st.columns(2)
        with col_plot3:
            st.markdown("#### Quantization Loss: KL Divergence vs. VRAM Savings")
            # Filter rows with KLD values
            loss_df = filtered_df.copy()
            loss_df["KLD"] = pd.to_numeric(loss_df["KLD"], errors='coerce')
            loss_df["PPL"] = pd.to_numeric(loss_df["PPL"], errors='coerce')
            loss_df = loss_df[(loss_df["KLD"] > 0) & (loss_df["PPL"].notna()) & (loss_df["PPL"] > 0)].copy()
            if not loss_df.empty:
                # Add VRAM saving percentage
                loss_df["VRAM Savings (%)"] = loss_df["KV Quant"].map(VRAM_SAVINGS).fillna(0.0)
                fig3 = px.scatter(
                    loss_df,
                    x="VRAM Savings (%)",
                    y="KLD",
                    color="Model",
                    size="PPL",
                    text="KV Quant",
                    title="KL Divergence Distance vs. Estimated VRAM Cache Compression",
                    labels={"KLD": "Kullback-Leibler Divergence (Lower is better)"}
                )
                fig3.update_traces(textposition='top center')
                fig3.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig3)
            else:
                st.info("No KL Divergence data available for plots.")
                
        with col_plot4:
            st.markdown("#### Perplexity (PPL) vs. KV Cache Quantization")
            ppl_df = filtered_df.dropna(subset=["PPL"])
            if not ppl_df.empty:
                fig4 = px.line(
                    ppl_df.sort_values(by="PPL", ascending=False),
                    x="KV Quant",
                    y="PPL",
                    color="Model",
                    markers=True,
                    title="Perplexity Shift (Lower is better)"
                )
                fig4.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig4)
            else:
                st.info("No Perplexity metrics available for plots.")
    else:
        st.info("No runs logged to plot.")

with tab_compare:
    st.subheader("Side-by-Side Model Comparison")
    if len(filtered_df) >= 2:
        # Group runs by Model and KV Cache Quant to get valid options
        unique_combinations = filtered_df.groupby(["Model", "KV Quant"]).size().reset_index()[["Model", "KV Quant"]]
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Configuration A")
            models_a = sorted(list(unique_combinations["Model"].unique())) if not unique_combinations.empty else []
            selected_model_a = st.selectbox("Select Profile A", models_a, index=0 if models_a else None, key="model_a") if models_a else None
            
            quants_a = sorted(list(unique_combinations[unique_combinations["Model"] == selected_model_a]["KV Quant"].unique())) if selected_model_a else []
            selected_quant_a = st.selectbox("Select Quant A", quants_a, index=0 if quants_a else None, key="quant_a") if quants_a else None
            
            if selected_model_a and selected_quant_a:
                runA_candidates = filtered_df[(filtered_df["Model"] == selected_model_a) & (filtered_df["KV Quant"] == selected_quant_a)]
                runA = runA_candidates.iloc[0] if not runA_candidates.empty else None
            else:
                runA = None
            
        with col_c2:
            st.markdown("#### Configuration B")
            models_b = sorted(list(unique_combinations["Model"].unique())) if not unique_combinations.empty else []
            selected_model_b = st.selectbox("Select Profile B", models_b, index=0 if models_b else None, key="model_b") if models_b else None
            
            quants_b = sorted(list(unique_combinations[unique_combinations["Model"] == selected_model_b]["KV Quant"].unique())) if selected_model_b else []
            selected_quant_b = st.selectbox("Select Quant B", quants_b, index=0 if quants_b else None, key="quant_b") if quants_b else None
            
            if selected_model_b and selected_quant_b:
                runB_candidates = filtered_df[(filtered_df["Model"] == selected_model_b) & (filtered_df["KV Quant"] == selected_quant_b)]
                runB = runB_candidates.iloc[0] if not runB_candidates.empty else None
            else:
                runB = None
            
        if runA is not None and runB is not None:
            st.markdown("### Comparison Table")
            
            # Build comparison details
            compare_rows = [
                ("Model Name", str(runA["Model"]), str(runB["Model"])),
                ("Endpoint", str(runA["Endpoint"]), str(runB["Endpoint"])),
                ("Base Quant", str(runA["Base Quant"]), str(runB["Base Quant"])),
                ("KV Cache Quant", str(runA["KV Quant"]), str(runB["KV Quant"])),
                ("Context Length (tks)", str(runA["Context Length"]), str(runB["Context Length"])),
                ("Threads", str(runA["Threads"]) if pd.notna(runA["Threads"]) else "N/A", str(runB["Threads"]) if pd.notna(runB["Threads"]) else "N/A"),
                ("Prefill Speed (t/s)", fmt_num(runA['Prefill (t/s)'], "{:.2f}"), fmt_num(runB['Prefill (t/s)'], "{:.2f}")),
                ("Decode Speed (t/s)", fmt_num(runA['Decode (t/s)'], "{:.2f}"), fmt_num(runB['Decode (t/s)'], "{:.2f}")),
                ("TTFT (s)", fmt_num(runA['TTFT (s)'], "{:.3f}"), fmt_num(runB['TTFT (s)'], "{:.3f}")),
                ("Needle Retrieval", str(runA["Needle"]), str(runB["Needle"])),
                ("RULER Var Tracking", str(runA["RULER"]), str(runB["RULER"])),
                ("LongBench Document QA", str(runA["LongBench"]), str(runB["LongBench"])),
                ("SWE-bench Toy Debugging", str(runA["SWE-bench"]), str(runB["SWE-bench"])),
                ("Perplexity (PPL)", fmt_num(runA['PPL'], "{:.4f}"), fmt_num(runB['PPL'], "{:.4f}")),
                ("KL Divergence (KLD)", fmt_num(runA['KLD'], "{:.6f}"), fmt_num(runB['KLD'], "{:.6f}")),
                ("Same Top Token %", f"{fmt_num(runA['Same Top %'], '{:.2f}')}%" if fmt_num(runA['Same Top %'], '{:.2f}') != "N/A" else "N/A", f"{fmt_num(runB['Same Top %'], '{:.2f}')}%" if fmt_num(runB['Same Top %'], '{:.2f}') != "N/A" else "N/A")
            ]
            
            comp_df = pd.DataFrame(compare_rows, columns=["Metric", "Configuration A", "Configuration B"])
            st.table(comp_df)
        else:
            st.info("No matching runs found for comparisons.")
    else:
        st.info("Need at least 2 historical runs in the database to perform side-by-side comparison.")

with tab_run:
    st.subheader("Execute New Benchmarks")
    st.markdown("Select your configuration and execute the unified runner script (`run_suite.py`) in the background.")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        new_mode = st.selectbox("Benchmark Mode", ["all", "throughput", "reasoning", "kld"])
        new_endpoint = st.text_input("Server Endpoint URL", value="http://127.0.0.1:8081")
        
        # Load available models from the endpoint dynamically
        available_models = [
            "unsloth/Qwen3.6-27B-GGUF:Q4_K_S",
            "unsloth/Qwen3.6-27B-GGUF:Q4_K_XL",
            "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S",
            "unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_S",
            "unsloth/gemma-4-E2B-it-GGUF:Q4_K_XL"
        ]
        try:
            validate_endpoint_url(new_endpoint)
            url = f"{new_endpoint}/v1/models"
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    models = [item.get("id") for item in data.get("data", [])]
                    if models:
                        available_models = sorted(models)
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            pass
            
        new_model = st.selectbox("Model ID / Endpoint Alias", available_models)
    with col_r2:
        new_tokens = st.number_input("Context Length Tokens (for reasoning)", value=5000, step=1000)
        new_gguf = st.text_input("Local GGUF Path (for KLD mode, auto-detects if blank)", value="")
        new_corpus = st.text_input("Corpus Text File (for KLD mode)", value="kld_corpus.txt")
        
    if st.button("▶️ Launch Benchmark Process", type="primary", width="stretch"):
        try:
            valid_endpoint = validate_endpoint_url(new_endpoint)
            valid_model = validate_model_name(new_model)
            valid_corpus = validate_corpus_name(new_corpus)
            valid_gguf = validate_gguf_path(new_gguf) if new_gguf else ""
        except ValueError as e:
            st.error(f"Input validation error: {e}")
            st.stop()

        st.session_state.bench_running = True
        st.session_state.bench_output = []
        
        # Build arguments list
        args = [
            "python3", "run_suite.py",
            "--mode", new_mode,
            "--endpoint", valid_endpoint,
            "--model", valid_model,
            "--tokens", str(int(new_tokens)),
            "--corpus", valid_corpus
        ]
        if valid_gguf:
            args.extend(["--gguf-path", valid_gguf])
            
        st.info(f"Running command: {' '.join(args)}")
        
        # Execute with real-time feedback
        log_placeholder = st.empty()
        
        proc = subprocess.Popen(
            args,
            cwd=str(Path(__file__).parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        line_queue = queue.Queue()

        def enqueue_output(out, q):
            try:
                for line in iter(out.readline, ''):
                    q.put(line)
            finally:
                if out and not out.closed:
                    try:
                        out.close()
                    except Exception:
                        pass

        reader_thread = threading.Thread(target=enqueue_output, args=(proc.stdout, line_queue), daemon=True)
        reader_thread.start()

        timeout_sec = 3600
        start_time = time.time()

        try:
            while True:
                got_lines = False
                while True:
                    try:
                        line = line_queue.get_nowait()
                        output_lines.append(line)
                        got_lines = True
                    except queue.Empty:
                        break

                if got_lines:
                    log_placeholder.code("".join(output_lines[-40:]), language="bash")

                if proc.poll() is not None:
                    break

                if time.time() - start_time > timeout_sec:
                    st.error("Benchmark process timed out.")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1)
                    break

                time.sleep(0.1)

            while True:
                try:
                    line = line_queue.get_nowait()
                    output_lines.append(line)
                except queue.Empty:
                    break
            if output_lines:
                log_placeholder.code("".join(output_lines[-40:]), language="bash")

            proc.wait(timeout=5)
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
            reader_thread.join(timeout=1)
        
        if proc.returncode == 0:
            st.success("Benchmark completed successfully! Refreshing historical registry runs...")
            # Re-read historical runs
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Benchmark process failed with exit code: {proc.returncode}")
