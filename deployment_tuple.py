"""Deployment provenance tuple capture and engine inspection utilities."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from utils import validate_endpoint_url


@dataclass
class DeploymentTuple:
    """Canonical description of an evaluated engine/model deployment."""

    engine: str = "unknown"
    engine_version: str = ""
    model_alias: str = ""
    weight_quant: str = "Unknown"
    kv_quant: str = "Unknown"
    routing_mode: str = "unknown"
    speculative_mode: str = "none"
    thinking_default: bool | None = None
    parallel_slots: int | None = None
    max_context: int | None = None
    host: str = ""
    hardware_profile: str = ""
    launch_env: dict[str, str] = field(default_factory=dict)
    extra_meta: dict[str, Any] = field(default_factory=dict)


def parse_extra_meta(raw: str | None) -> dict[str, Any]:
    """Parse ``--extra-meta`` JSON inline string or ``@file.json`` reference."""
    if not raw or not raw.strip():
        return {}
    text = raw.strip()
    if text.startswith("@"):
        path = Path(text[1:]).expanduser()
        text = path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("extra_meta must decode to a JSON object")
    return parsed


def _auth_headers(api_key: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def _probe_json(
    url: str,
    headers: dict[str, str],
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None
    except (requests.RequestException, ValueError):
        return None


def _infer_host_profile(endpoint: str) -> tuple[str, str]:
    parsed = urlparse(endpoint)
    host = parsed.hostname or "unknown"
    if host in {"127.0.0.1", "localhost", "::1"}:
        return "x570", "Dual RTX 3090 48GB NVLink + Ryzen 9 5950X"
    if host == "192.168.0.31":
        return "gx10", "DGX Spark / Asus Ascend GX10 GB10 sm_121"
    return host, f"remote:{host}"


def _infer_quant_from_strings(*values: str) -> str:
    joined = " ".join(v for v in values if v)
    patterns = (
        r"(NVFP4-DQ8|NVFP4-QAD|NVFP4|FP4-Local|SQ-\d+|UD-Q[0-9]+_K_[A-Z]+|"
        r"UD-IQ[0-9]+_[A-Z]+|IQ[0-9]+_[A-Z]+|Q[0-9]+_K_[A-Z]+|Q[0-9]+_[0-9]+|BF16)"
    )
    match = re.search(patterns, joined, re.IGNORECASE)
    return match.group(1) if match else "Unknown"


def detect_engine_from_endpoint(
    endpoint: str,
    model_alias: str,
    *,
    engine_override: str | None = None,
    api_key: str | None = None,
    health_payload: dict[str, Any] | None = None,
    models_payload: dict[str, Any] | None = None,
    props_payload: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> DeploymentTuple:
    """Inspect OpenAI-compatible server metadata and assemble a DeploymentTuple."""
    validate_endpoint_url(endpoint, allow_private=True)
    base = endpoint.rstrip("/")
    headers = _auth_headers(api_key)

    health = (
        health_payload
        if health_payload is not None
        else _probe_json(f"{base}/health", headers)
    )
    models = (
        models_payload
        if models_payload is not None
        else _probe_json(f"{base}/v1/models", headers)
    )
    props = (
        props_payload
        if props_payload is not None
        else _probe_json(f"{base}/props", headers)
    )

    meta = dict(extra_meta or {})
    host, hw_profile = _infer_host_profile(base)

    engine = (engine_override or meta.get("engine") or "").strip().lower()
    engine_version = str(meta.get("engine_version") or "")
    kv_quant = str(meta.get("kv_quant") or "Unknown")
    weight_quant = str(meta.get("weight_quant") or "Unknown")
    routing_mode = str(meta.get("routing_mode") or "unknown")
    spec_mode = str(meta.get("speculative_mode") or "none")
    parallel_slots = meta.get("parallel_slots")
    max_context = meta.get("max_context")
    thinking_default = meta.get("thinking_default")

    model_entry: dict[str, Any] = {}
    if models and isinstance(models.get("data"), list):
        for item in models["data"]:
            if isinstance(item, dict) and item.get("id") == model_alias:
                model_entry = item
                break
        if not model_entry and models["data"]:
            first = models["data"][0]
            if isinstance(first, dict):
                model_entry = first

    if not engine:
        owned_by = str(model_entry.get("owned_by") or "").lower()
        if health and ("max_context" in health or "loaded" in health):
            engine = "basalt" if "basalt" in owned_by or host == "gx10" else "strata"
        elif props or "status" in model_entry or "llamacpp" in owned_by:
            engine = "llama.cpp"
        elif "strata" in owned_by:
            engine = "strata"
        else:
            engine = "unknown"

    if not engine_version and health:
        engine_version = str(
            health.get("version") or health.get("build") or engine_version
        )
    if max_context is None and health and isinstance(health.get("max_context"), int):
        max_context = health["max_context"]
    if (
        max_context is None
        and props
        and isinstance(props.get("default_generation_settings"), dict)
    ):
        n_ctx = props["default_generation_settings"].get("n_ctx")
        if isinstance(n_ctx, int):
            max_context = n_ctx
    if parallel_slots is None and props and isinstance(props.get("total_slots"), int):
        parallel_slots = props["total_slots"]
    if parallel_slots is None and health and isinstance(health.get("parallel"), int):
        parallel_slots = health["parallel"]

    if kv_quant == "Unknown" and health and isinstance(health.get("kv"), str):
        kv_quant = health["kv"]
    if kv_quant == "Unknown" and isinstance(model_entry.get("status"), dict):
        args = model_entry["status"].get("args") or []
        for idx, token in enumerate(args):
            if token in ("-ctk", "--cache-type-k") and idx + 1 < len(args):
                kv_quant = str(args[idx + 1])
                break

    if weight_quant == "Unknown":
        model_path_str = str(
            model_entry.get("meta", {}).get("path", "")
            if isinstance(model_entry.get("meta"), dict)
            else ""
        )
        weight_quant = _infer_quant_from_strings(model_alias, model_path_str)

    launch_env = {
        k: v
        for k, v in (meta.get("launch_env") or {}).items()
        if isinstance(k, str) and isinstance(v, str)
    }
    if not launch_env:
        for env_key in ("STRATA_ROUTE_TAIL_SKIP", "CUDA_VISIBLE_DEVICES"):
            if env_key in os.environ:
                launch_env[env_key] = os.environ[env_key]

    return DeploymentTuple(
        engine=engine,
        engine_version=engine_version,
        model_alias=model_alias,
        weight_quant=weight_quant,
        kv_quant=kv_quant,
        routing_mode=routing_mode,
        speculative_mode=spec_mode,
        thinking_default=thinking_default,
        parallel_slots=parallel_slots,
        max_context=max_context,
        host=str(meta.get("host") or host),
        hardware_profile=str(meta.get("hardware_profile") or hw_profile),
        launch_env=launch_env,
        extra_meta=meta,
    )


def deployment_tuple_to_dict(dt: DeploymentTuple) -> dict[str, Any]:
    """Convert DeploymentTuple to a plain JSON-serializable dictionary."""
    return asdict(dt)
