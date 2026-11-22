"""Normalized KLD and perplexity adapters for llama.cpp, Basalt, and OpenAI endpoints."""

from __future__ import annotations

import json
import math
import re
import subprocess
from typing import Any

import requests

from bench_throughput import auth_headers
from utils import validate_endpoint_url


def parse_llama_kld_output(stdout: str) -> dict[str, Any]:
    """Parse ``llama-perplexity --kl-divergence`` text output."""
    result: dict[str, Any] = {}
    mean_m = re.search(r"Mean KLD\s*[:|]\s*([0-9.]+)", stdout, re.IGNORECASE)
    if mean_m:
        result["mean_kld"] = float(mean_m.group(1))
    else:
        for line in stdout.splitlines():
            if (
                line.strip().startswith("|")
                and "Mean" not in line
                and "---" not in line
            ):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 4:
                    try:
                        result["mean_kld"] = float(parts[0])
                        result["max_kld"] = float(parts[1])
                        result["same_top_match_percent"] = float(parts[-1].rstrip("%"))
                        break
                    except ValueError:
                        continue
    ppl_m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", stdout, re.IGNORECASE)
    if ppl_m:
        result["perplexity"] = float(ppl_m.group(1))
    base_ppl_m = re.search(r"Final\s+base\s+PPL\s*=\s*([0-9.]+)", stdout, re.IGNORECASE)
    if base_ppl_m:
        result["baseline_perplexity"] = float(base_ppl_m.group(1))
    return result


def parse_basalt_bench_output(stdout: str) -> dict[str, Any]:
    """Parse ``basalt-bench --quality`` JSON or table output into normalized metrics."""
    stripped = stdout.strip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
        out: dict[str, Any] = {}
        for key_src, key_dst in (
            ("mean_kld", "mean_kld"),
            ("kld", "mean_kld"),
            ("max_kld", "max_kld"),
            ("ppl", "perplexity"),
            ("perplexity", "perplexity"),
            ("baseline_ppl", "baseline_perplexity"),
            ("same_top_pct", "same_top_match_percent"),
        ):
            if isinstance(data.get(key_src), (int, float)):
                out[key_dst] = float(data[key_src])
        return out

    out = {}
    kld_m = re.search(r"KLD\s*[:=]\s*([0-9.]+)", stdout, re.IGNORECASE)
    ppl_m = re.search(r"PPL\s*[:=]\s*([0-9.]+)", stdout, re.IGNORECASE)
    top_m = re.search(r"Same[- ]Top\s*[:=]\s*([0-9.]+)%", stdout, re.IGNORECASE)
    if kld_m:
        out["mean_kld"] = float(kld_m.group(1))
    if ppl_m:
        out["perplexity"] = float(ppl_m.group(1))
    if top_m:
        out["same_top_match_percent"] = float(top_m.group(1))
    return out


def run_kld_subprocess(
    engine: str, command: list[str], *, timeout: int = 1800
) -> dict[str, Any]:
    """Run an offline KLD/perplexity command and normalize stdout metrics."""
    if not command:
        raise ValueError("command must be non-empty")
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = (
        parse_basalt_bench_output(combined)
        if engine.lower() == "basalt"
        else parse_llama_kld_output(combined)
    )
    return {
        "engine": engine,
        "returncode": proc.returncode,
        **parsed,
    }


def run_reference_ppl_probe(
    endpoint: str,
    model: str,
    corpus_text: str,
    *,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Compute held-out prompt perplexity via ``/v1/completions`` echo logprobs when supported."""
    validate_endpoint_url(endpoint, allow_private=True)
    url = f"{endpoint.rstrip('/')}/v1/completions"
    body = {
        "model": model,
        "prompt": corpus_text,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
    }
    try:
        resp = requests.post(
            url, headers=auth_headers(api_key), json=body, timeout=timeout
        )
        if resp.status_code != 200:
            return {"supported": False, "status": resp.status_code}
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        logprobs = (choice.get("logprobs") or {}).get("token_logprobs")
        if not isinstance(logprobs, list):
            return {"supported": False, "status": resp.status_code}
        vals = [float(v) for v in logprobs if isinstance(v, (int, float))]
        if not vals:
            return {"supported": False, "status": resp.status_code}
        nll = -sum(vals) / len(vals)
        return {
            "supported": True,
            "tokens_scored": len(vals),
            "perplexity": round(math.exp(min(nll, 20.0)), 4),
        }
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        return {"supported": False, "error": str(exc)}
