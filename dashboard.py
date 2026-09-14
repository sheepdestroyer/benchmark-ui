#!/usr/bin/env python3
import configparser
import functools
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from utils import (
    CONTEXT_TIERS as CONTEXT_TIERS,
    redact_cli_args,
    validate_endpoint_url,
    validate_model_name,
)

BASE_QUANT_TYPES = (
    "Q4_K_XL",
    "Q6_K_XL",
    "Q4_K_S",
    "Q8_0",
    "Q5_1",
    "Q4_0",
    "F16",
    "Q5_K_M",
)
BASE_QUANT_ALIASES = tuple((q.lower(), q) for q in BASE_QUANT_TYPES)
REQUIRED_THROUGHPUT_COLS = (
    "Model",
    "KV Quant",
    "Context Length",
    "Prefill (t/s)",
    "Decode (t/s)",
)
REQUIRED_CONTEXT_SCALING_COLS = (
    "Context Length",
    "Prefill (t/s)",
    "Decode (t/s)",
    "TTFT (s)",
)
REQUIRED_REASONING_RATIO_COLS = (
    "Reasoning Tokens",
    "Completion Tokens",
)


def validate_gguf_path(gguf_path_str):
    if not gguf_path_str:
        return gguf_path_str
    resolved = Path(gguf_path_str).resolve()
    if not resolved.exists():
        raise ValueError(f"GGUF file path does not exist: {gguf_path_str}")
    if not resolved.is_file():
        raise ValueError(f"GGUF path is not a file: {gguf_path_str}")
    if resolved.suffix.lower() != ".gguf":
        raise ValueError(f"GGUF file must have a .gguf extension: {gguf_path_str}")

    allowed_parents = [
        Path.cwd().resolve(),
        Path(__file__).parent.resolve(),
        Path.home().resolve() / ".cache",
        (Path.home().resolve() / ".cache").resolve(),
        Path.home().resolve() / "models",
        (Path.home().resolve() / "models").resolve(),
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
        raise ValueError(
            f"GGUF path escapes allowed parent directories: {gguf_path_str}"
        )
    return str(resolved)


def validate_corpus_name(corpus_str):
    if not corpus_str or not str(corpus_str).strip():
        raise ValueError("Corpus name cannot be empty.")
    safe_name = os.path.basename(str(corpus_str).strip()).strip()
    if not safe_name or safe_name in (".", ".."):
        raise ValueError(f"Invalid corpus name: {corpus_str}")
    return safe_name


def validate_new_tokens(new_tokens):
    if new_tokens is None or isinstance(new_tokens, bool):
        raise ValueError("Context length tokens must be between 1 and 262144.")
    try:
        if isinstance(new_tokens, float) and not new_tokens.is_integer():
            raise ValueError("Context length tokens must be between 1 and 262144.")
        tokens_int = int(new_tokens)
    except (TypeError, ValueError):
        raise ValueError("Context length tokens must be between 1 and 262144.")

    if not (1 <= tokens_int <= 262144):
        raise ValueError("Context length tokens must be between 1 and 262144.")
    return tokens_int


validate_tokens = validate_new_tokens

CONTEXT_TIER_OPTIONS = [
    "8k (Smoke)",
    "32k (Standard Agentic)",
    "64k (Large Agentic)",
    "128k (Deep Window)",
    "240k (Full Window)",
    "Custom",
]

TIER_TO_TOKENS = {
    "8k (Smoke)": 8192,
    "32k (Standard Agentic)": 32768,
    "64k (Large Agentic)": 65536,
    "128k (Deep Window)": 131072,
    "240k (Full Window)": 240000,
}


def validate_max_tokens(max_tokens):
    if max_tokens is None or isinstance(max_tokens, bool):
        raise ValueError("Max output tokens must be between 256 and 32768.")
    try:
        if isinstance(max_tokens, float) and not max_tokens.is_integer():
            raise ValueError("Max output tokens must be between 256 and 32768.")
        tokens_int = int(max_tokens)
    except (TypeError, ValueError):
        raise ValueError("Max output tokens must be between 256 and 32768.")

    if not (256 <= tokens_int <= 32768):
        raise ValueError("Max output tokens must be between 256 and 32768.")
    return tokens_int


def build_runner_cmd(
    mode,
    endpoint,
    model,
    tokens,
    corpus,
    gguf_path=None,
    api_key=None,
    max_tokens=16384,
    agentic_tasks="all",
):
    cmd = [
        sys.executable,
        "run_suite.py",
        "--mode",
        str(mode),
        "--endpoint",
        str(endpoint),
        "--model",
        str(model),
        "--tokens",
        str(tokens),
        "--corpus",
        str(corpus),
        "--max-tokens",
        str(max_tokens),
    ]
    if mode in ["agentic", "all"]:
        cmd.extend(["--agentic-tasks", str(agentic_tasks)])
    if gguf_path:
        cmd.extend(["--gguf-path", str(gguf_path)])
    if api_key:
        cmd.extend(["--api-key", str(api_key)])
    return cmd


# Page config
st.set_page_config(
    page_title="LLM Benchmarking Registry & Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
ENDPOINTS_FILE = Path(os.environ.get("ENDPOINTS_FILE", HISTORY_DIR / "endpoints.json"))

DEFAULT_ENDPOINTS = [
    {
        "name": "Local Llama Router",
        "url": "http://127.0.0.1:8083",
        "api_key": "",
        "is_default": True,
    },
    {
        "name": "Production LLM-Routing",
        "url": "https://llm-routing.vendeuvre.lan",
        "api_key": "",
        "is_default": False,
    },
]


def get_endpoints_file(file_path: Path | str | None = None) -> Path:
    if file_path is not None:
        return Path(file_path)
    return Path(os.environ.get("ENDPOINTS_FILE", ENDPOINTS_FILE))


def save_endpoints(endpoints: list[dict], file_path: Path | str | None = None) -> None:
    if not isinstance(endpoints, list):
        raise ValueError("Endpoints must be a list of dicts.")

    sanitized = []
    for item in endpoints:
        if not isinstance(item, dict):
            raise ValueError("Each endpoint record must be a dict.")
        name = item.get("name")
        url = item.get("url")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Endpoint record must contain a non-empty string 'name'.")
        if not url or not isinstance(url, str) or not url.strip():
            raise ValueError("Endpoint record must contain a non-empty string 'url'.")
        api_key = item.get("api_key", "")
        if not isinstance(api_key, str):
            api_key = ""
        is_default = bool(item.get("is_default", False))
        sanitized.append(
            {
                "name": name.strip(),
                "url": url.strip().rstrip("/"),
                "api_key": api_key.strip(),
                "is_default": is_default,
            }
        )

    target_file = get_endpoints_file(file_path)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = target_file.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=4)
    os.chmod(tmp_file, 0o600)
    os.replace(tmp_file, target_file)


def load_endpoints(file_path: Path | str | None = None) -> list[dict]:
    target_file = get_endpoints_file(file_path)
    if not target_file.exists():
        save_endpoints(DEFAULT_ENDPOINTS, target_file)
        return [dict(e) for e in DEFAULT_ENDPOINTS]

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                save_endpoints(DEFAULT_ENDPOINTS, target_file)
                return [dict(e) for e in DEFAULT_ENDPOINTS]
            data = json.loads(content)
    except Exception:
        save_endpoints(DEFAULT_ENDPOINTS, target_file)
        return [dict(e) for e in DEFAULT_ENDPOINTS]

    if not isinstance(data, list):
        save_endpoints(DEFAULT_ENDPOINTS, target_file)
        return [dict(e) for e in DEFAULT_ENDPOINTS]

    validated = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        url = item.get("url")
        if not name or not isinstance(name, str) or not name.strip():
            continue
        if not url or not isinstance(url, str) or not url.strip():
            continue
        api_key = item.get("api_key", "")
        if not isinstance(api_key, str):
            api_key = ""
        is_default = bool(item.get("is_default", False))
        validated.append(
            {
                "name": name.strip(),
                "url": url.strip().rstrip("/"),
                "api_key": api_key.strip(),
                "is_default": is_default,
            }
        )

    if not validated:
        save_endpoints(DEFAULT_ENDPOINTS, target_file)
        return [dict(e) for e in DEFAULT_ENDPOINTS]

    return validated


def add_endpoint(
    name: str,
    url: str,
    api_key: str = "",
    is_default: bool = False,
    file_path: Path | str | None = None,
) -> list[dict]:
    if not name or not str(name).strip():
        raise ValueError("Endpoint name cannot be empty.")
    cleaned_name = str(name).strip()
    valid_url = validate_endpoint_url(str(url).strip(), allow_private=True).rstrip("/")
    endpoints = load_endpoints(file_path)

    if any(e["name"].lower() == cleaned_name.lower() for e in endpoints):
        raise ValueError(f"Endpoint with name '{cleaned_name}' already exists.")

    if is_default:
        for ep in endpoints:
            ep["is_default"] = False

    endpoints.append(
        {
            "name": cleaned_name,
            "url": valid_url,
            "api_key": str(api_key).strip() if api_key else "",
            "is_default": bool(is_default),
        }
    )
    save_endpoints(endpoints, file_path)
    return endpoints


def update_endpoint(
    original_name: str,
    name: str,
    url: str,
    api_key: str = "",
    is_default: bool = False,
    file_path: Path | str | None = None,
) -> list[dict]:
    if not name or not str(name).strip():
        raise ValueError("Endpoint name cannot be empty.")
    cleaned_name = str(name).strip()
    valid_url = validate_endpoint_url(str(url).strip(), allow_private=True).rstrip("/")
    endpoints = load_endpoints(file_path)

    target_idx = None
    for idx, ep in enumerate(endpoints):
        if ep["name"] == original_name:
            target_idx = idx
            break

    if target_idx is None:
        raise ValueError(f"Endpoint '{original_name}' not found.")

    if cleaned_name.lower() != original_name.lower():
        if any(e["name"].lower() == cleaned_name.lower() for e in endpoints):
            raise ValueError(f"Endpoint with name '{cleaned_name}' already exists.")

    if is_default:
        for ep in endpoints:
            ep["is_default"] = False

    endpoints[target_idx] = {
        "name": cleaned_name,
        "url": valid_url,
        "api_key": str(api_key).strip() if api_key else "",
        "is_default": bool(is_default),
    }
    save_endpoints(endpoints, file_path)
    return endpoints


def delete_endpoint(
    name: str,
    file_path: Path | str | None = None,
) -> list[dict]:
    endpoints = load_endpoints(file_path)
    if len(endpoints) <= 1:
        raise ValueError("Cannot delete the only configured endpoint.")

    target_idx = None
    for idx, ep in enumerate(endpoints):
        if ep["name"] == name:
            target_idx = idx
            break

    if target_idx is None:
        raise ValueError(f"Endpoint '{name}' not found.")

    removed = endpoints.pop(target_idx)
    if removed.get("is_default") and not any(e.get("is_default") for e in endpoints):
        endpoints[0]["is_default"] = True

    save_endpoints(endpoints, file_path)
    return endpoints


TEST_SUITES = ("Needle", "RULER", "LongBench", "SWE-bench")
PASS_FAIL_STATUSES = frozenset({"Pass", "Fail"})

# VRAM savings helper
VRAM_SAVINGS = {
    "f16": 0.0,
    "F16": 0.0,
    "q8_0": 50.0,
    "Q8_0": 50.0,
    "q5_1": 68.0,
    "Q5_1": 68.0,
    "q4_0": 75.0,
    "Q4_0": 75.0,
    "q4_k_m": 75.0,
    "Q4_K_M": 75.0,
    "q5_k_m": 68.0,
    "Q5_K_M": 68.0,
    "q4_k_s": 75.0,
    "Q4_K_S": 75.0,
    "q5_k_s": 68.0,
    "Q5_K_S": 68.0,
    "q8_k_m": 50.0,
    "Q8_K_M": 50.0,
    "Unknown": 0.0,
    "UNKNOWN": 0.0,
}


def fmt_num(val, fmt="{:.2f}"):
    if val is None or val is pd.NA:
        return "N/A"
    if isinstance(val, bool):
        return str(val)
    try:
        if val == "N/A":
            return "N/A"
    except (TypeError, ValueError):
        pass
    try:
        is_na = pd.isna(val)
        if isinstance(is_na, bool) and is_na:
            return "N/A"
        if hasattr(is_na, "item") and not hasattr(is_na, "__len__") and bool(is_na):
            return "N/A"
    except (ValueError, TypeError, Exception):
        pass
    try:
        fval = float(val)
        if math.isnan(fval):
            return "N/A"
        if math.isinf(fval):
            return "Inf" if fval > 0 else "-Inf"
        return fmt.format(fval)
    except (ValueError, TypeError):
        return str(val)


def extract_reasoning_acc_data(df):
    """Extract reasoning benchmark pass/fail scores for bar charting."""
    if df is None or getattr(df, "empty", True):
        return []
    if "Model" not in df.columns or "KV Quant" not in df.columns:
        return []

    def _scalar_loc(loc):
        if isinstance(loc, int):
            return loc
        if hasattr(loc, "__iter__"):
            for idx, val in enumerate(loc):
                if val:
                    return idx
        return int(loc)

    try:
        model_idx = _scalar_loc(df.columns.get_loc("Model"))
        kv_idx = _scalar_loc(df.columns.get_loc("KV Quant"))
    except (KeyError, TypeError, ValueError):
        return []

    test_indices = {}
    for test in TEST_SUITES:
        if test in df.columns:
            try:
                test_indices[test] = _scalar_loc(df.columns.get_loc(test))
            except (KeyError, TypeError, ValueError):
                continue

    if not test_indices:
        return []

    acc_data = []
    for row in df.itertuples(index=False, name=None):
        for test, t_idx in test_indices.items():
            val = row[t_idx]
            if val in PASS_FAIL_STATUSES:
                acc_data.append(
                    {
                        "Model_Quant": f"{row[model_idx]} ({row[kv_idx]})",
                        "Test Suite": test,
                        "Score": 1.0 if val == "Pass" else 0.0,
                    }
                )
    return acc_data


# Inject premium CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


def _normalize_repo_id(val):
    if not val or not isinstance(val, str):
        return ""
    val = val.strip()
    if val.lower().startswith("unsloth/"):
        return val[len("unsloth/") :].strip()
    return val


def _repo_id_matches(target, candidate):
    if (
        not target
        or not candidate
        or not isinstance(target, str)
        or not isinstance(candidate, str)
    ):
        return False
    target_clean = target.strip()
    candidate_clean = candidate.strip()
    if not target_clean or not candidate_clean:
        return False
    if target_clean.lower() == candidate_clean.lower():
        return True
    norm_target = _normalize_repo_id(target_clean).lower()
    norm_candidate = _normalize_repo_id(candidate_clean).lower()
    return bool(norm_target and norm_candidate and norm_target == norm_candidate)


def resolve_presets_path(presets_file=None):
    """Resolve presets file path with PRESETS_FILE env var and default fallback paths."""
    if presets_file is not None:
        return str(presets_file)
    env_file = os.environ.get("PRESETS_FILE")
    if env_file:
        return env_file
    primary = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini"
        )
    )
    if os.path.exists(primary):
        return primary
    fallback = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../llama.cpp/model_presets.ini")
    )
    if os.path.exists(fallback):
        return fallback
    return primary


@functools.lru_cache(maxsize=4)
def _get_presets_config(presets_file=None):
    presets_path = resolve_presets_path(presets_file)
    if not os.path.exists(presets_path):
        return None

    try:
        config = configparser.ConfigParser(strict=False)
        config.read(presets_path, encoding="utf-8")
        return config
    except Exception:
        return None


def map_repo_to_preset_alias(repo_or_id, presets_file=None):
    if not repo_or_id or not isinstance(repo_or_id, str):
        return repo_or_id

    config = _get_presets_config(presets_file)
    if config:
        try:
            # Exact or normalized section check first
            for section in config.sections():
                if _repo_id_matches(section, repo_or_id):
                    return section

            # Exact or normalized hf-repo and alias check
            for section in config.sections():
                if section == "*":
                    continue
                section_repo = config.get(section, "hf-repo", fallback="")
                section_alias = config.get(section, "alias", fallback="")

                if section_repo and _repo_id_matches(section_repo, repo_or_id):
                    return section

                if section_alias:
                    alias_parts = [
                        a.strip() for a in section_alias.split(",") if a.strip()
                    ]
                    for a in alias_parts:
                        if _repo_id_matches(a, repo_or_id):
                            return section
                    if _repo_id_matches(section_alias, repo_or_id):
                        return section

            # Fallback substring checks
            for section in config.sections():
                if section == "*":
                    continue
                section_repo = config.get(section, "hf-repo", fallback="")
                section_alias = config.get(section, "alias", fallback="")

                if section_repo:
                    sec_repo_norm = _normalize_repo_id(section_repo).lower()
                    query_norm = _normalize_repo_id(repo_or_id).lower()
                    if (section_repo.lower() in repo_or_id.lower()) or (
                        sec_repo_norm and sec_repo_norm in query_norm
                    ):
                        if "mtp" in repo_or_id.lower() and "spec" in section.lower():
                            return section
                        if (
                            "mtp" not in repo_or_id.lower()
                            and "spec" not in section.lower()
                        ):
                            return section

                if section_alias:
                    sec_alias_norm = _normalize_repo_id(section_alias).lower()
                    query_norm = _normalize_repo_id(repo_or_id).lower()
                    if (section_alias.lower() in repo_or_id.lower()) or (
                        sec_alias_norm and sec_alias_norm in query_norm
                    ):
                        return section
        except (configparser.Error, OSError):
            pass

    # Fallback overrides
    lower_id = repo_or_id.lower()
    if "qwen3.6-27b-gguf:q4_k_s" in lower_id:
        return "Qwen3.6-27B"
    elif "qwen3.6-27b-mtp-gguf:q4_k_s" in lower_id:
        return "Qwen3.6-27B-spec3"
    elif "qwen3.6-35b-a3b-gguf:q4_k_s" in lower_id:
        return "Qwen3.6-35B-A3B"
    elif "gemma-4" in lower_id:
        return "gemma4-26a4b-routing"

    return repo_or_id


def get_preset_metadata(profile_name, presets_file=None):
    metadata = {
        "spec_type": "None",
        "spec_draft_type_k": "None",
        "spec_draft_type_v": "None",
        "flash_attn": "true",
        "parallel": "1",
        "n_gpu_layers": "99",
        "fit": "true",
    }

    config = _get_presets_config(presets_file)
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


def _is_mock(obj):
    return (
        hasattr(obj, "_mock_return_value")
        or hasattr(obj, "_mock_self")
        or getattr(type(obj), "__name__", "")
        in ("MagicMock", "Mock", "NonCallableMagicMock", "AsyncMock")
    )


def _fallback_cache_data(*dargs, **dkwargs):
    ttl = dkwargs.get("ttl")
    if dargs and isinstance(dargs[0], (int, float)):
        ttl = dargs[0]

    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                val, expiry = cache[key]
                if expiry is None or now < expiry:
                    return val
            result = func(*args, **kwargs)
            cache[key] = (result, now + ttl if ttl else None)
            return result

        def clear():
            cache.clear()

        wrapper.clear = clear
        return wrapper

    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return decorator(dargs[0])
    return decorator


if not hasattr(st, "cache_data") or _is_mock(getattr(st, "cache_data", None)):
    try:
        st.cache_data = _fallback_cache_data
    except Exception:
        pass


def _parse_run_file(filepath):
    """Read a single run_*.json file, parse metrics and metadata, returning a run dictionary or None."""
    filepath = Path(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            st.error(f"Error loading {filepath.name}: root object must be a dictionary")
            return None

        metadata = data.get("run_metadata") or {}
        settings = data.get("model_settings") or {}
        throughput = data.get("throughput_metrics") or {}
        accuracy = data.get("reasoning_accuracy") or {}
        loss = data.get("quantization_loss") or {}

        # Parse context length
        args = metadata.get("cli_arguments") or []
        ctx_len = 200000
        for i, arg in enumerate(args):
            if arg == "--tokens" and i + 1 < len(args):
                try:
                    ctx_len = int(args[i + 1])
                except ValueError:
                    pass

        # Prefer explicit profile_alias from log first, fallback to model_name resolution
        profile_name = settings.get("profile_alias")
        if not profile_name:
            profile_name = map_repo_to_preset_alias(
                settings.get("model_name", "Unknown")
            )
        elif "/" in str(profile_name) or str(profile_name).endswith(".gguf"):
            mapped = map_repo_to_preset_alias(settings.get("model_name", ""))
            if mapped and mapped != "Unknown":
                profile_name = mapped

        # If profile_name is still a path or has a raw gguf filename, clean it
        if "/" in str(profile_name) or str(profile_name).endswith(".gguf"):
            name = str(profile_name).split("/")[-1]
            name = name.removesuffix(".gguf")
            for q in [
                "-Q4_K_S",
                "-UD-Q4_K_XL",
                "-UD-Q4_K_S",
                "-Q6_K_XL",
                "-Q8_0",
                "-F16",
            ]:
                name = name.replace(q, "")
            name = name.removesuffix("-UD")
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

        # Safely extract agentic metrics and token breakdown
        agentic = data.get("agentic_metrics") or {}
        token_breakdown = data.get("token_breakdown") or {}
        tasks_total = agentic.get("tasks_total")
        tasks_passed = agentic.get("tasks_passed")
        pass_rate = None
        if tasks_total and tasks_total > 0 and tasks_passed is not None:
            pass_rate = round((tasks_passed / tasks_total) * 100, 1)

        return {
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
            "Agentic Suite": agentic.get("suite") or "N/A",
            "Agentic Total": tasks_total,
            "Agentic Passed": tasks_passed,
            "Agentic Pass Rate": pass_rate,
            "Agentic Turns": agentic.get("average_turns"),
            "Agentic Tool Calls": agentic.get("total_tool_calls"),
            "Prompt Tokens": token_breakdown.get("prompt_tokens"),
            "Reasoning Tokens": token_breakdown.get("reasoning_tokens"),
            "Completion Tokens": token_breakdown.get("completion_tokens"),
            "PPL": loss.get("perplexity"),
            "KLD": loss.get("mean_kld"),
            "Same Top %": loss.get("same_top_match_percent"),
            "Context Length": ctx_len,
        }
    except Exception as e:
        st.error(f"Error loading {filepath.name}: {e}")
        return None


# Load all runs
@st.cache_data(ttl=60)
def load_runs():
    runs = []
    for filepath in HISTORY_DIR.glob("run_*.json"):
        record = _parse_run_file(filepath)
        if record is not None:
            runs.append(record)

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
    st.markdown(
        "### <span class='header-gradient'>Dashboard Filters</span>",
        unsafe_allow_html=True,
    )

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
        selected_spec_types = st.multiselect(
            "Spec Type", spec_types, default=spec_types
        )

        # Spec Draft Type K filter
        spec_drafts = sorted(list(df["Spec Draft Type K"].dropna().unique()))
        selected_spec_drafts = st.multiselect(
            "Spec Draft Type K", spec_drafts, default=spec_drafts
        )

        # Context length filter
        ctx_lens = sorted([int(x) for x in df["Context Length"].dropna().unique()])
        if ctx_lens:
            selected_ctx = st.multiselect("Context Lengths", ctx_lens, default=ctx_lens)
        else:
            selected_ctx = []

        # Filter dataframe
        filtered_df = df[
            df["Endpoint"].isin(selected_endpoints)
            & df["Model"].isin(selected_models)
            & df["KV Quant"].isin(selected_quants)
            & df["Spec Type"].isin(selected_spec_types)
            & df["Spec Draft Type K"].isin(selected_spec_drafts)
        ]
        if selected_ctx:
            filtered_df = filtered_df[filtered_df["Context Length"].isin(selected_ctx)]
    else:
        st.warning("No runs found in the registry database.")
        filtered_df = df

st.markdown(
    "# 🚀 <span class='header-gradient'>llama.cpp Benchmark Registry & Optimizer</span>",
    unsafe_allow_html=True,
)
st.markdown(
    "Phase 5 visualizer dashboard for multi-GPU performance, quantization loss trade-offs, and reasoning capabilities."
)

# KPI metrics
if not filtered_df.empty:
    is_kld_file = filtered_df["Filename"].str.contains(
        r"_(?:f16|q8_0|q5_1|q4_0)\.json$"
    )
    unified_kpi_df = filtered_df[~is_kld_file]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Registry Runs", len(unified_kpi_df))
    with col2:
        best_prefill = filtered_df["Prefill (t/s)"].dropna().max()
        st.metric(
            "Max Prefill Speed",
            f"{fmt_num(best_prefill, '{:.1f}')} t/s"
            if fmt_num(best_prefill, "{:.1f}") != "N/A"
            else "N/A",
        )
    with col3:
        best_decode = filtered_df["Decode (t/s)"].dropna().max()
        st.metric(
            "Max Decode Speed",
            f"{fmt_num(best_decode, '{:.1f}')} t/s"
            if fmt_num(best_decode, "{:.1f}") != "N/A"
            else "N/A",
        )
    with col4:
        kld_numeric = pd.to_numeric(filtered_df["KLD"], errors="coerce")
        min_kld = kld_numeric[kld_numeric > 0].min()
        st.metric(
            "Best non-zero KLD",
            fmt_num(min_kld, "{:.6f}")
            if fmt_num(min_kld, "{:.6f}") != "N/A"
            else "0.000000",
        )

# Tabs
tab_history, tab_plots, tab_compare, tab_run = st.tabs(
    [
        "🗂️ Run History Browser",
        "📈 Comparative Plots",
        "⚖️ Side-by-Side Model Comparison",
        "⚙️ Run New Benchmark",
    ]
)

with tab_history:
    st.subheader("Historical Benchmark Runs")
    if not filtered_df.empty:
        # Filter out the raw KLD individual files from history tables (keep only unified complete runs)
        is_kld_file = filtered_df["Filename"].str.contains(
            r"_(?:f16|q8_0|q5_1|q4_0)\.json$"
        )
        unified_df = filtered_df[~is_kld_file]

        # Check and display Agentic Benchmark Summary metrics if present
        has_agentic = (
            "Agentic Total" in unified_df.columns
            and unified_df["Agentic Total"].notna().any()
        )
        if has_agentic:
            st.markdown("### 🤖 Agentic Benchmark Summary")
            agentic_df = unified_df[unified_df["Agentic Total"].notna()]
            total_tasks = int(agentic_df["Agentic Total"].sum())
            total_passed = int(agentic_df["Agentic Passed"].fillna(0).sum())
            overall_pass_rate = (
                round((total_passed / total_tasks) * 100, 1) if total_tasks > 0 else 0.0
            )
            avg_turns = (
                round(agentic_df["Agentic Turns"].dropna().mean(), 1)
                if not agentic_df["Agentic Turns"].dropna().empty
                else 0.0
            )
            total_tools = (
                int(agentic_df["Agentic Tool Calls"].dropna().sum())
                if not agentic_df["Agentic Tool Calls"].dropna().empty
                else 0
            )

            col_ag1, col_ag2, col_ag3, col_ag4 = st.columns(4)
            with col_ag1:
                st.metric("Agentic Tasks Run", total_tasks)
            with col_ag2:
                st.metric("Pass Rate %", f"{overall_pass_rate}%")
            with col_ag3:
                st.metric("Average Turns", avg_turns)
            with col_ag4:
                st.metric("Tool Calls", total_tools)

        # Grouped Summary table
        st.markdown("### 📊 Profile & Quantization Summary (Grouped Averages)")
        summary_cols = [
            "Model",
            "Base Quant",
            "KV Quant",
            "Spec Type",
            "Spec Draft Type K",
            "Context Length",
            "Prefill (t/s)",
            "Decode (t/s)",
            "TTFT (s)",
            "PPL",
            "KLD",
        ]
        if (
            "Agentic Pass Rate" in unified_df.columns
            and unified_df["Agentic Pass Rate"].notna().any()
        ):
            summary_cols.extend(
                ["Agentic Pass Rate", "Agentic Turns", "Agentic Tool Calls"]
            )
        grouped_df = (
            unified_df[summary_cols]
            .groupby(
                [
                    "Model",
                    "Base Quant",
                    "KV Quant",
                    "Spec Type",
                    "Spec Draft Type K",
                    "Context Length",
                ]
            )
            .mean()
            .reset_index()
        )

        # Calculate reasoning pass rates for the group
        acc_cols = [
            "Model",
            "Base Quant",
            "KV Quant",
            "Spec Type",
            "Spec Draft Type K",
            "Context Length",
            "Needle",
            "RULER",
            "LongBench",
            "SWE-bench",
        ]
        acc_group = unified_df[acc_cols].copy()
        for col in ["Needle", "RULER", "LongBench", "SWE-bench"]:
            acc_group[col] = acc_group[col].map({"Pass": 1.0, "Fail": 0.0})
        acc_grouped = (
            acc_group.groupby(
                [
                    "Model",
                    "Base Quant",
                    "KV Quant",
                    "Spec Type",
                    "Spec Draft Type K",
                    "Context Length",
                ]
            )
            .mean()
            .reset_index()
        )

        merged_grouped = pd.merge(
            grouped_df,
            acc_grouped,
            on=[
                "Model",
                "Base Quant",
                "KV Quant",
                "Spec Type",
                "Spec Draft Type K",
                "Context Length",
            ],
        )
        rename_dict = {
            "Prefill (t/s)": "Avg Prefill (t/s)",
            "Decode (t/s)": "Avg Decode (t/s)",
            "TTFT (s)": "Avg TTFT (s)",
            "PPL": "Avg PPL",
            "KLD": "Avg KLD",
            "Needle": "Needle Pass Rate",
            "RULER": "RULER Pass Rate",
            "LongBench": "LongBench Pass Rate",
            "SWE-bench": "SWE-bench Pass Rate",
            "Agentic Pass Rate": "Avg Agentic Pass Rate",
            "Agentic Turns": "Avg Agentic Turns",
            "Agentic Tool Calls": "Avg Agentic Tool Calls",
        }
        merged_grouped = merged_grouped.rename(columns=rename_dict)

        grouped_format = {
            "Avg Prefill (t/s)": "{:.2f}",
            "Avg Decode (t/s)": "{:.2f}",
            "Avg TTFT (s)": "{:.3f}",
            "Avg PPL": "{:.4f}",
            "Avg KLD": "{:.6f}",
            "Needle Pass Rate": "{:.0%}",
            "RULER Pass Rate": "{:.0%}",
            "LongBench Pass Rate": "{:.0%}",
            "SWE-bench Pass Rate": "{:.0%}",
        }
        if "Avg Agentic Pass Rate" in merged_grouped.columns:
            grouped_format["Avg Agentic Pass Rate"] = (
                lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
            )
        if "Avg Agentic Turns" in merged_grouped.columns:
            grouped_format["Avg Agentic Turns"] = (
                lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            )
        if "Avg Agentic Tool Calls" in merged_grouped.columns:
            grouped_format["Avg Agentic Tool Calls"] = (
                lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            )

        st.dataframe(
            merged_grouped.style.format(grouped_format),
        )

        st.markdown("### 🗂️ Detailed Flat Logs")

        # Display clean browser dataframe
        display_cols = [
            "Timestamp",
            "Model",
            "Base Quant",
            "KV Quant",
            "Prefill (t/s)",
            "Decode (t/s)",
            "TTFT (s)",
            "Needle",
            "RULER",
            "LongBench",
            "SWE-bench",
            "Agentic Suite",
            "Agentic Pass Rate",
            "Agentic Turns",
            "Agentic Tool Calls",
            "PPL",
            "KLD",
        ]
        active_display_cols = [c for c in display_cols if c in unified_df.columns]
        flat_format = {
            "Prefill (t/s)": "{:.2f}",
            "Decode (t/s)": "{:.2f}",
            "TTFT (s)": "{:.3f}",
            "PPL": "{:.4f}",
            "KLD": "{:.6f}",
        }
        if "Agentic Pass Rate" in unified_df.columns:
            flat_format["Agentic Pass Rate"] = (
                lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
            )
        if "Agentic Turns" in unified_df.columns:
            flat_format["Agentic Turns"] = (
                lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            )
        if "Agentic Tool Calls" in unified_df.columns:
            flat_format["Agentic Tool Calls"] = (
                lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
            )

        st.dataframe(
            unified_df[active_display_cols].style.format(flat_format),
        )
    else:
        st.info("No runs match the filter criteria.")


def build_throughput_figure(tp_df, model_colors=None):
    """Build Plotly figure for throughput (PP vs TG) across models and quantization formats."""
    from plotly.subplots import make_subplots

    # Create subplot figure with secondary y-axis
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    if (
        tp_df is None
        or getattr(tp_df, "empty", True)
        or not hasattr(tp_df, "columns")
        or not set(REQUIRED_THROUGHPUT_COLS).issubset(tp_df.columns)
    ):
        return fig1

    if model_colors is None:
        model_colors = {
            "Qwen3.6-27B": "#3b82f6",  # Blue
            "Qwen3.6-27B-spec3": "#10b981",  # Green
            "Qwen3.6-27B-spec4": "#8b5cf6",  # Purple
            "Qwen3.6-35B-A3B-spec": "#f97316",  # Orange
            "Qwen3.6-35B-A3B": "#ef4444",  # Red
        }

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

    # Pre-sort and group by ("Model", "KV Quant") to eliminate redundant O(N) filtering in nested loops
    tp_sorted = tp_df.dropna(subset=["KV Quant"]).sort_values(by="Context Length")
    seen_models = set()

    for (model, quant), q_df in tp_sorted.groupby(["Model", "KV Quant"], sort=True):
        if q_df.empty:  # pragma: no cover
            continue

        color = model_colors.get(model, "#94a3b8")
        show_legend = model not in seen_models  # Show in legend only once per model
        seen_models.add(model)

        customdata = tuple(zip(q_df["Model"], q_df["KV Quant"]))

        # Select dash style based on quant format
        dash_style = (
            "solid"
            if quant == "f16"
            else (
                "dash" if quant == "q8_0" else ("dot" if quant == "q5_1" else "dashdot")
            )
        )

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
                    line=dict(width=1, color="#1e293b"),
                ),
                line=dict(color=color, width=1.5, dash=dash_style),
                customdata=customdata,
                hovertemplate=hover_template_pp,
            ),
            secondary_y=False,
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
                    line=dict(width=1, color="#1e293b"),
                ),
                line=dict(color=color, width=1.5, dash=dash_style),
                customdata=customdata,
                hovertemplate=hover_template_tg,
            ),
            secondary_y=True,
        )

    # Update layout, axes titles, log scale, and dark template
    fig1.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title="Read/Write (PP vs TG) Generation Speeds across Cache Formats",
        xaxis_title="Context Length (tokens, log scale)",
        xaxis_type="log",
    )

    # Left Y-axis (PP)
    fig1.update_yaxes(
        title_text="Prompt Processing (PP) Speed (tokens/sec)", secondary_y=False
    )
    # Right Y-axis (TG)
    fig1.update_yaxes(
        title_text="Token Generation (TG) Speed (tokens/sec)", secondary_y=True
    )

    return fig1


def build_context_scaling_figure(df, model_colors=None):
    """Build Plotly figure for context scaling (Prefill, Decode, TTFT vs. Context Length up to 240k)."""
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if (
        df is None
        or getattr(df, "empty", True)
        or not hasattr(df, "columns")
        or not set(REQUIRED_CONTEXT_SCALING_COLS).issubset(df.columns)
    ):
        return fig

    clean_df = df.copy()
    for col in REQUIRED_CONTEXT_SCALING_COLS:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    clean_df = clean_df.dropna(subset=REQUIRED_CONTEXT_SCALING_COLS)
    clean_df = clean_df[clean_df["Context Length"] > 0].sort_values(by="Context Length")

    if clean_df.empty:
        return fig

    if model_colors is None:
        model_colors = {
            "Qwen3.6-27B": "#3b82f6",
            "Qwen3.6-27B-spec3": "#10b981",
            "Qwen3.6-27B-spec4": "#8b5cf6",
            "Qwen3.6-35B-A3B-spec": "#f97316",
            "Qwen3.6-35B-A3B": "#ef4444",
        }

    models = clean_df["Model"].unique() if "Model" in clean_df.columns else ["Default"]
    seen_models = set()

    for model in sorted(models):
        m_df = (
            clean_df[clean_df["Model"] == model]
            if "Model" in clean_df.columns
            else clean_df
        )
        if m_df.empty:  # pragma: no cover
            continue

        color = model_colors.get(model, "#94a3b8")
        show_legend = model not in seen_models
        seen_models.add(model)

        grouped = (
            m_df.groupby("Context Length", as_index=False)[
                ["Prefill (t/s)", "Decode (t/s)", "TTFT (s)"]
            ]
            .mean()
            .sort_values(by="Context Length")
        )

        prefix = f"{model} " if len(models) > 1 else ""

        # Prefill Speed (t/s) - Left Y-axis (secondary_y=False)
        fig.add_trace(
            go.Scatter(
                x=grouped["Context Length"],
                y=grouped["Prefill (t/s)"],
                mode="lines+markers",
                name=f"{prefix}Prefill (t/s)",
                legendgroup=f"{model}_prefill",
                showlegend=show_legend,
                marker=dict(symbol="square", size=8, color=color),
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{model} Prefill</b><br>Context: %{{x}} tokens<br>Speed: %{{y:.2f}} t/s<extra></extra>",
            ),
            secondary_y=False,
        )

        # Decode Speed (t/s) - Left Y-axis (secondary_y=False)
        fig.add_trace(
            go.Scatter(
                x=grouped["Context Length"],
                y=grouped["Decode (t/s)"],
                mode="lines+markers",
                name=f"{prefix}Decode (t/s)",
                legendgroup=f"{model}_decode",
                showlegend=show_legend,
                marker=dict(symbol="circle", size=8, color=color),
                line=dict(color=color, width=2, dash="dash"),
                hovertemplate=f"<b>{model} Decode</b><br>Context: %{{x}} tokens<br>Speed: %{{y:.2f}} t/s<extra></extra>",
            ),
            secondary_y=False,
        )

        # TTFT (s) - Right Y-axis (secondary_y=True)
        fig.add_trace(
            go.Scatter(
                x=grouped["Context Length"],
                y=grouped["TTFT (s)"],
                mode="lines+markers",
                name=f"{prefix}TTFT (s)",
                legendgroup=f"{model}_ttft",
                showlegend=show_legend,
                marker=dict(symbol="triangle-up", size=8, color=color),
                line=dict(color=color, width=1.5, dash="dot"),
                hovertemplate=f"<b>{model} TTFT</b><br>Context: %{{x}} tokens<br>TTFT: %{{y:.3f}} s<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title="Context Scaling Curves: Throughput & Latency vs. Context Length (up to 240k tokens)",
        xaxis_title="Context Length (tokens, log scale)",
        xaxis_type="log",
    )
    fig.update_yaxes(title_text="Generation Speed (tokens/sec)", secondary_y=False)
    fig.update_yaxes(title_text="Time to First Token (seconds)", secondary_y=True)

    return fig


def build_reasoning_ratio_figure(df):
    """Build Plotly figure for reasoning tokens vs completion tokens across models."""
    fig = go.Figure()

    if (
        df is None
        or getattr(df, "empty", True)
        or not hasattr(df, "columns")
        or not set(REQUIRED_REASONING_RATIO_COLS).issubset(df.columns)
    ):
        fig.update_layout(template="plotly_dark")
        return fig

    clean_df = df.copy()
    for col in ["Reasoning Tokens", "Completion Tokens"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df = clean_df.dropna(
        subset=["Reasoning Tokens", "Completion Tokens"], how="all"
    )
    if clean_df.empty:
        fig.update_layout(template="plotly_dark")
        return fig

    clean_df["Reasoning Tokens"] = clean_df["Reasoning Tokens"].fillna(0).clip(lower=0)
    clean_df["Completion Tokens"] = clean_df["Completion Tokens"].fillna(0).clip(lower=0)

    clean_df = clean_df[
        (clean_df["Reasoning Tokens"] > 0) | (clean_df["Completion Tokens"] > 0)
    ]
    if clean_df.empty:
        fig.update_layout(template="plotly_dark")
        return fig

    if "Model" not in clean_df.columns:
        clean_df["Model"] = "Default"

    grouped = (
        clean_df.groupby("Model", as_index=False)[
            ["Reasoning Tokens", "Completion Tokens"]
        ]
        .mean()
        .sort_values(by="Model")
    )

    fig.add_trace(
        go.Bar(
            name="Reasoning Tokens",
            x=grouped["Model"],
            y=grouped["Reasoning Tokens"],
            marker_color="#8b5cf6",
            hovertemplate="<b>%{x}</b><br>Reasoning: %{y:.0f} tokens<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Completion Tokens",
            x=grouped["Model"],
            y=grouped["Completion Tokens"],
            marker_color="#3b82f6",
            hovertemplate="<b>%{x}</b><br>Completion: %{y:.0f} tokens<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title="Reasoning vs. Completion Token Breakdown Across Models",
        xaxis_title="Model",
        yaxis_title="Average Tokens",
    )

    return fig


def enqueue_output(out, q):
    """Read lines from stream into queue until EOF and close the stream."""
    try:
        for line in iter(out.readline, ""):
            q.put(line)
    finally:
        try:
            out.close()
        except Exception:  # noqa: BLE001, S110
            pass


with tab_plots:
    st.subheader("Performance & Quantization Trade-off Analysis")
    if not filtered_df.empty:
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            st.markdown("#### Throughput (t/s) vs. KV Cache Quantization")
            # Filter rows with throughput values
            tp_df = filtered_df.dropna(subset=["Prefill (t/s)", "Decode (t/s)"])
            if not tp_df.empty:
                fig1 = build_throughput_figure(tp_df)
                st.plotly_chart(fig1)
            else:
                st.info("No throughput metrics available for plots.")

        with col_plot2:
            st.markdown("#### Reasoning Benchmarks Pass Rates")
            # Map Pass/Fail/NA to numeric values for bar charting
            acc_data = extract_reasoning_acc_data(filtered_df)
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
                    labels={"Score": "Pass Rate (0 or 1)"},
                )
                fig2.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig2)
            else:
                st.info("No reasoning accuracy data available for plots.")

        st.divider()

        col_plot3, col_plot4 = st.columns(2)
        with col_plot3:
            st.markdown("#### Quantization Loss: KL Divergence vs. VRAM Savings")
            # Filter rows with KLD values
            loss_df = filtered_df.copy()
            loss_df["KLD"] = pd.to_numeric(loss_df["KLD"], errors="coerce")
            loss_df["PPL"] = pd.to_numeric(loss_df["PPL"], errors="coerce")
            loss_df = loss_df[
                (loss_df["KLD"] > 0) & (loss_df["PPL"].notna()) & (loss_df["PPL"] > 0)
            ].copy()
            if not loss_df.empty:
                # Add VRAM saving percentage
                loss_df["VRAM Savings (%)"] = (
                    loss_df["KV Quant"].map(VRAM_SAVINGS).fillna(0.0)
                )
                fig3 = px.scatter(
                    loss_df,
                    x="VRAM Savings (%)",
                    y="KLD",
                    color="Model",
                    size="PPL",
                    text="KV Quant",
                    title="KL Divergence Distance vs. Estimated VRAM Cache Compression",
                    labels={"KLD": "Kullback-Leibler Divergence (Lower is better)"},
                )
                fig3.update_traces(textposition="top center")
                fig3.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
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
                    title="Perplexity Shift (Lower is better)",
                )
                fig4.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig4)
            else:
                st.info("No Perplexity metrics available for plots.")

        st.divider()

        col_plot5, col_plot6 = st.columns(2)
        with col_plot5:
            st.markdown("#### Context Scaling Curves (1k - 240k tokens)")
            ctx_df = filtered_df.dropna(
                subset=["Context Length", "Prefill (t/s)", "Decode (t/s)", "TTFT (s)"]
            )
            if not ctx_df.empty:
                fig_ctx = build_context_scaling_figure(ctx_df)
                st.plotly_chart(fig_ctx)
            else:  # pragma: no cover
                st.info("No context scaling data available for plots.")

        with col_plot6:
            st.markdown("#### Reasoning vs. Completion Token Breakdown")
            has_tokens = (
                "Reasoning Tokens" in filtered_df.columns
                and filtered_df["Reasoning Tokens"].notna().any()
            )
            if has_tokens:
                fig_ratio = build_reasoning_ratio_figure(filtered_df)
                st.plotly_chart(fig_ratio)
            else:  # pragma: no cover
                st.info("No token breakdown data available for plots.")
    else:
        st.info("No runs logged to plot.")

with tab_compare:
    st.subheader("Side-by-Side Model Comparison")
    if len(filtered_df) >= 2:
        # Group runs by Model and KV Cache Quant to get valid options
        unique_combinations = (
            filtered_df.groupby(["Model", "KV Quant"])
            .size()
            .reset_index()[["Model", "KV Quant"]]
        )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Configuration A")
            models_a = (
                sorted(list(unique_combinations["Model"].unique()))
                if not unique_combinations.empty
                else []
            )
            selected_model_a = (
                st.selectbox(
                    "Select Profile A",
                    models_a,
                    index=0 if models_a else None,
                    key="model_a",
                )
                if models_a
                else None
            )

            quants_a = (
                sorted(
                    list(
                        unique_combinations[
                            unique_combinations["Model"] == selected_model_a
                        ]["KV Quant"].unique()
                    )
                )
                if selected_model_a
                else []
            )
            selected_quant_a = (
                st.selectbox(
                    "Select Quant A",
                    quants_a,
                    index=0 if quants_a else None,
                    key="quant_a",
                )
                if quants_a
                else None
            )

            if selected_model_a and selected_quant_a:
                runA_candidates = filtered_df[
                    (filtered_df["Model"] == selected_model_a)
                    & (filtered_df["KV Quant"] == selected_quant_a)
                ]
                runA = runA_candidates.iloc[0] if not runA_candidates.empty else None
            else:
                runA = None

        with col_c2:
            st.markdown("#### Configuration B")
            models_b = (
                sorted(list(unique_combinations["Model"].unique()))
                if not unique_combinations.empty
                else []
            )
            selected_model_b = (
                st.selectbox(
                    "Select Profile B",
                    models_b,
                    index=0 if models_b else None,
                    key="model_b",
                )
                if models_b
                else None
            )

            quants_b = (
                sorted(
                    list(
                        unique_combinations[
                            unique_combinations["Model"] == selected_model_b
                        ]["KV Quant"].unique()
                    )
                )
                if selected_model_b
                else []
            )
            selected_quant_b = (
                st.selectbox(
                    "Select Quant B",
                    quants_b,
                    index=0 if quants_b else None,
                    key="quant_b",
                )
                if quants_b
                else None
            )

            if selected_model_b and selected_quant_b:
                runB_candidates = filtered_df[
                    (filtered_df["Model"] == selected_model_b)
                    & (filtered_df["KV Quant"] == selected_quant_b)
                ]
                runB = runB_candidates.iloc[0] if not runB_candidates.empty else None
            else:
                runB = None

        if runA is not None and runB is not None:
            # Check for agentic metrics in runA or runB
            has_agentic_comp = (
                pd.notna(runA.get("Agentic Total")) or pd.notna(runB.get("Agentic Total"))
            )
            if has_agentic_comp:
                st.markdown("#### 🤖 Agentic Benchmark Summary")
                col_ag_a, col_ag_b = st.columns(2)
                with col_ag_a:
                    st.markdown(f"**{runA['Model']} Agentic Performance**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Agentic Tasks Run",
                        runA.get("Agentic Total")
                        if pd.notna(runA.get("Agentic Total"))
                        else "N/A",
                    )
                    c2.metric(
                        "Pass Rate %",
                        f"{runA.get('Agentic Pass Rate')}%"
                        if pd.notna(runA.get("Agentic Pass Rate"))
                        else "N/A",
                    )
                    c3.metric(
                        "Average Turns",
                        runA.get("Agentic Turns")
                        if pd.notna(runA.get("Agentic Turns"))
                        else "N/A",
                    )
                    c4.metric(
                        "Tool Calls",
                        runA.get("Agentic Tool Calls")
                        if pd.notna(runA.get("Agentic Tool Calls"))
                        else "N/A",
                    )
                with col_ag_b:
                    st.markdown(f"**{runB['Model']} Agentic Performance**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Agentic Tasks Run",
                        runB.get("Agentic Total")
                        if pd.notna(runB.get("Agentic Total"))
                        else "N/A",
                    )
                    c2.metric(
                        "Pass Rate %",
                        f"{runB.get('Agentic Pass Rate')}%"
                        if pd.notna(runB.get("Agentic Pass Rate"))
                        else "N/A",
                    )
                    c3.metric(
                        "Average Turns",
                        runB.get("Agentic Turns")
                        if pd.notna(runB.get("Agentic Turns"))
                        else "N/A",
                    )
                    c4.metric(
                        "Tool Calls",
                        runB.get("Agentic Tool Calls")
                        if pd.notna(runB.get("Agentic Tool Calls"))
                        else "N/A",
                    )

            st.markdown("### Comparison Table")

            # Build comparison details
            compare_rows = [
                ("Model Name", str(runA["Model"]), str(runB["Model"])),
                ("Endpoint", str(runA["Endpoint"]), str(runB["Endpoint"])),
                ("Base Quant", str(runA["Base Quant"]), str(runB["Base Quant"])),
                ("KV Cache Quant", str(runA["KV Quant"]), str(runB["KV Quant"])),
                (
                    "Context Length (tks)",
                    str(runA["Context Length"]),
                    str(runB["Context Length"]),
                ),
                (
                    "Threads",
                    str(runA["Threads"]) if pd.notna(runA["Threads"]) else "N/A",
                    str(runB["Threads"]) if pd.notna(runB["Threads"]) else "N/A",
                ),
                (
                    "Prefill Speed (t/s)",
                    fmt_num(runA["Prefill (t/s)"], "{:.2f}"),
                    fmt_num(runB["Prefill (t/s)"], "{:.2f}"),
                ),
                (
                    "Decode Speed (t/s)",
                    fmt_num(runA["Decode (t/s)"], "{:.2f}"),
                    fmt_num(runB["Decode (t/s)"], "{:.2f}"),
                ),
                (
                    "TTFT (s)",
                    fmt_num(runA["TTFT (s)"], "{:.3f}"),
                    fmt_num(runB["TTFT (s)"], "{:.3f}"),
                ),
                ("Needle Retrieval", str(runA["Needle"]), str(runB["Needle"])),
                ("RULER Var Tracking", str(runA["RULER"]), str(runB["RULER"])),
                (
                    "LongBench Document QA",
                    str(runA["LongBench"]),
                    str(runB["LongBench"]),
                ),
                (
                    "SWE-bench Toy Debugging",
                    str(runA["SWE-bench"]),
                    str(runB["SWE-bench"]),
                ),
                (
                    "Agentic Suite",
                    str(runA.get("Agentic Suite"))
                    if pd.notna(runA.get("Agentic Suite")) and runA.get("Agentic Suite")
                    else "N/A",
                    str(runB.get("Agentic Suite"))
                    if pd.notna(runB.get("Agentic Suite")) and runB.get("Agentic Suite")
                    else "N/A",
                ),
                (
                    "Agentic Pass Rate",
                    f"{runA.get('Agentic Pass Rate')}%"
                    if pd.notna(runA.get("Agentic Pass Rate"))
                    else "N/A",
                    f"{runB.get('Agentic Pass Rate')}%"
                    if pd.notna(runB.get("Agentic Pass Rate"))
                    else "N/A",
                ),
                (
                    "Agentic Turns",
                    fmt_num(runA.get("Agentic Turns"), "{:.1f}"),
                    fmt_num(runB.get("Agentic Turns"), "{:.1f}"),
                ),
                (
                    "Agentic Tool Calls",
                    fmt_num(runA.get("Agentic Tool Calls"), "{:.0f}"),
                    fmt_num(runB.get("Agentic Tool Calls"), "{:.0f}"),
                ),
                (
                    "Perplexity (PPL)",
                    fmt_num(runA["PPL"], "{:.4f}"),
                    fmt_num(runB["PPL"], "{:.4f}"),
                ),
                (
                    "KL Divergence (KLD)",
                    fmt_num(runA["KLD"], "{:.6f}"),
                    fmt_num(runB["KLD"], "{:.6f}"),
                ),
                (
                    "Same Top Token %",
                    f"{fmt_num(runA['Same Top %'], '{:.2f}')}%"
                    if fmt_num(runA["Same Top %"], "{:.2f}") != "N/A"
                    else "N/A",
                    f"{fmt_num(runB['Same Top %'], '{:.2f}')}%"
                    if fmt_num(runB["Same Top %"], "{:.2f}") != "N/A"
                    else "N/A",
                ),
            ]

            comp_df = pd.DataFrame(
                compare_rows, columns=["Metric", "Configuration A", "Configuration B"]
            )
            st.table(comp_df)
        else:
            st.info("No matching runs found for comparisons.")
    else:
        st.info(
            "Need at least 2 historical runs in the database to perform side-by-side comparison."
        )


def fetch_available_models(endpoint: str, api_key: str | None = None) -> list[str]:
    """Fetch and return sorted list of model IDs from an OpenAI-compatible endpoint."""
    cleaned_endpoint = endpoint.rstrip("/")
    validate_endpoint_url(cleaned_endpoint, allow_private=True)
    url = f"{cleaned_endpoint}/v1/models"
    headers = {}
    key = (
        api_key
        if api_key is not None
        else (os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    )
    if key and key.strip():
        headers["Authorization"] = f"Bearer {key.strip()}"
    resp = requests.get(url, headers=headers, timeout=3)
    resp.raise_for_status()
    data = resp.json()
    models_raw = data.get("data") or [] if isinstance(data, dict) else []
    models = [
        item.get("id")
        for item in models_raw
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and item.get("id").strip()
    ]
    if not models:
        raise ValueError("No models found in response")
    return sorted(models)


with tab_run:
    st.subheader("Execute New Benchmarks")
    st.markdown(
        "Select your configuration and execute the unified runner script (`run_suite.py`) in the background."
    )

    endpoints = load_endpoints()

    if st.session_state.get("endpoint_notice"):
        st.success(st.session_state.pop("endpoint_notice"))

    with st.expander("⚙️ Manage Server Endpoints", expanded=False):
        manage_options = ["➕ Add New Endpoint"] + [e["name"] for e in endpoints]
        selected_manage = st.selectbox(
            "Select Endpoint to Manage",
            manage_options,
            key="manage_ep_target",
        )
        is_new = selected_manage == "➕ Add New Endpoint"
        curr_ep = (
            next((e for e in endpoints if e["name"] == selected_manage), None)
            if not is_new
            else None
        )

        init_name = "" if is_new else (curr_ep["name"] if curr_ep else "")
        init_url = (
            "http://127.0.0.1:8083" if is_new else (curr_ep["url"] if curr_ep else "")
        )
        init_key = "" if is_new else (curr_ep.get("api_key", "") if curr_ep else "")
        init_def = (
            False
            if is_new
            else (curr_ep.get("is_default", False) if curr_ep else False)
        )

        ep_name = st.text_input(
            "Endpoint Label / Name",
            value=init_name,
            key=f"ep_mgmt_name_{selected_manage}",
        )
        ep_url = st.text_input(
            "Endpoint URL", value=init_url, key=f"ep_mgmt_url_{selected_manage}"
        )
        ep_key = st.text_input(
            "API Key (optional)",
            value=init_key,
            type="password",
            key=f"ep_mgmt_key_{selected_manage}",
        )
        ep_default = st.checkbox(
            "Set as default endpoint",
            value=init_def,
            key=f"ep_mgmt_default_{selected_manage}",
        )

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            if st.button("💾 Save Endpoint", key="btn_save_ep"):
                if not ep_name.strip():
                    st.error("Endpoint name cannot be empty.")
                else:
                    try:
                        valid_u = validate_endpoint_url(
                            ep_url.strip(), allow_private=True
                        ).rstrip("/")
                        if is_new:
                            add_endpoint(
                                ep_name.strip(),
                                valid_u,
                                ep_key.strip(),
                                ep_default,
                            )
                            st.session_state["endpoint_notice"] = (
                                f"Endpoint '{ep_name.strip()}' saved successfully."
                            )
                        else:
                            update_endpoint(
                                selected_manage,
                                ep_name.strip(),
                                valid_u,
                                ep_key.strip(),
                                ep_default,
                            )
                            st.session_state["endpoint_notice"] = (
                                f"Endpoint '{ep_name.strip()}' updated successfully."
                            )
                        st.rerun()
                    except Exception as err:
                        st.error(f"Failed to save endpoint: {err}")

        with col_m2:
            if st.button("🔌 Test Connection", key="btn_test_ep"):
                try:
                    test_u = validate_endpoint_url(ep_url.strip(), allow_private=True)
                    tested_models = fetch_available_models(
                        test_u, ep_key.strip() or None
                    )
                    st.success(
                        f"Connection successful! {len(tested_models)} models available: {', '.join(tested_models[:5])}..."
                    )
                except Exception as err:
                    st.error(f"Connection failed: {err}")

        with col_m3:
            if (
                not is_new
                and len(endpoints) > 1
                and st.button("🗑️ Delete Endpoint", key="btn_delete_ep")
            ):
                try:
                    delete_endpoint(selected_manage)
                    st.session_state["endpoint_notice"] = (
                        f"Endpoint '{selected_manage}' deleted successfully."
                    )
                    st.rerun()
                except Exception as err:
                    st.error(f"Failed to delete endpoint: {err}")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        new_mode = st.selectbox(
            "Benchmark Mode",
            ["all", "throughput", "reasoning", "agentic", "kld"],
            key="runner_benchmark_mode",
        )
        agentic_tasks = "all"
        if new_mode in ["agentic", "all"]:
            agentic_tasks = st.selectbox(
                "Agentic Tasks Filter",
                ["all", "fix-syntax", "log-analysis", "git-repair", "env-config"],
                index=0,
                key="runner_agentic_tasks",
            )

        endpoint_options = [e["name"] for e in endpoints] + ["Custom Endpoint..."]
        default_idx = 0
        for i, ep in enumerate(endpoints):
            if ep.get("is_default"):
                default_idx = i
                break

        selected_endpoint = st.selectbox(
            "Server Endpoint",
            endpoint_options,
            index=default_idx if endpoint_options else 0,
            key="runner_ep_select",
        )

        if selected_endpoint == "Custom Endpoint...":
            new_endpoint = st.text_input(
                "Endpoint URL",
                value="http://127.0.0.1:8083",
                key="runner_custom_url",
            )
            runner_api_key = st.text_input(
                "API Key (optional)",
                type="password",
                value="",
                key="runner_custom_api_key",
            )
        else:
            matched_ep = next(
                (e for e in endpoints if e["name"] == selected_endpoint),
                endpoints[0]
                if endpoints
                else {"url": "http://127.0.0.1:8083", "api_key": ""},
            )
            new_endpoint = matched_ep.get("url", "http://127.0.0.1:8083")
            runner_api_key = matched_ep.get("api_key", "")
            st.text_input(
                "Endpoint URL",
                value=new_endpoint,
                disabled=True,
                key=f"runner_saved_url_{selected_endpoint}",
            )
            if runner_api_key:
                st.text_input(
                    "API Key (optional)",
                    value=runner_api_key,
                    type="password",
                    disabled=True,
                    key=f"runner_saved_api_key_{selected_endpoint}",
                )

        # Load available models from the endpoint dynamically
        available_models = []
        try:
            available_models = fetch_available_models(
                new_endpoint, runner_api_key or None
            )
        except Exception as err:  # noqa: BLE001
            st.warning(
                f"Could not retrieve models from endpoint ({err}). You can enter a model identifier manually below."
            )

        if available_models:
            new_model = st.selectbox("Model ID / Endpoint Alias", available_models)
        else:
            new_model = st.text_input("Model ID / Endpoint Alias", value="Qwen3.6-27B")
        st.button("🔄 Refresh Models")
    with col_r2:
        selected_tier = st.selectbox(
            "Context Tier",
            CONTEXT_TIER_OPTIONS,
            index=1,
            key="runner_context_tier",
        )

        tier_token_val = TIER_TO_TOKENS.get(selected_tier)
        if (
            "prev_context_tier" not in st.session_state
            or st.session_state["prev_context_tier"] != selected_tier
        ):
            st.session_state["prev_context_tier"] = selected_tier
            if tier_token_val is not None:
                st.session_state["runner_context_tokens_input"] = tier_token_val
                st.session_state["runner_context_tokens_val"] = tier_token_val

        current_tokens_default = st.session_state.get(
            "runner_context_tokens_val",
            tier_token_val if tier_token_val is not None else 32768,
        )

        new_tokens = st.number_input(
            "Context Length Tokens (for reasoning)",
            min_value=1,
            max_value=262144,
            value=current_tokens_default,
            step=1000,
            key="runner_context_tokens_input",
        )
        st.session_state["runner_context_tokens_val"] = new_tokens

        new_max_tokens = st.number_input(
            "Max Output Tokens (completion limit)",
            min_value=256,
            max_value=32768,
            value=16384,
            step=512,
            key="runner_max_tokens",
        )
        new_gguf = st.text_input(
            "Local GGUF Path (for KLD mode, auto-detects if blank)", value=""
        )
        new_corpus = st.text_input(
            "Corpus Text File (for KLD mode)", value="kld_corpus.txt"
        )

    if st.button("▶️ Launch Benchmark Process", type="primary", width="stretch"):
        try:
            valid_endpoint = validate_endpoint_url(new_endpoint, allow_private=True)
            valid_model = validate_model_name(new_model)
            valid_corpus = validate_corpus_name(new_corpus)
            valid_gguf = validate_gguf_path(new_gguf) if new_gguf else ""
            valid_tokens = validate_new_tokens(new_tokens)
            valid_max_tokens = validate_max_tokens(new_max_tokens)
            valid_api_key = (
                runner_api_key.strip()
                if runner_api_key
                else (os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
            )
        except ValueError as e:
            st.error(f"Input validation error: {e}")
            st.stop()

        st.session_state.bench_running = True
        st.session_state.bench_output = []

        # Build arguments list
        cmd = build_runner_cmd(
            mode=new_mode,
            endpoint=valid_endpoint,
            model=valid_model,
            tokens=valid_tokens,
            corpus=valid_corpus,
            gguf_path=valid_gguf,
            api_key=valid_api_key,
            max_tokens=valid_max_tokens,
            agentic_tasks=agentic_tasks,
        )

        st.info(f"Running command: {' '.join(redact_cli_args(cmd))}")

        # Execute with real-time feedback
        log_placeholder = st.empty()

        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        line_queue = queue.Queue()

        reader_thread = threading.Thread(
            target=enqueue_output, args=(proc.stdout, line_queue), daemon=True
        )
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
            st.success(
                "Benchmark completed successfully! Refreshing historical registry runs..."
            )
            # Re-read historical runs
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Benchmark process failed with exit code: {proc.returncode}")
