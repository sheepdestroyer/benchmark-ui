"""Throughput, prefill/decode curves, prefix-cache speedup, concurrency, and HASS probes."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from utils import validate_endpoint_url

_SYNTHETIC_BLOCK = (
    "river stone bridge lantern harvest copper meadow signal orbit glacier "
    "amber cedar harbor marble willow falcon summit canyon lantern prism "
)


def auth_headers(api_key: str | None = None) -> dict[str, str]:
    """Build HTTP headers including optional Bearer authentication."""
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def build_synthetic_prompt(target_tokens: int, seed_tag: str = "probe") -> str:
    """Construct a deterministic prompt roughly ``target_tokens`` tokens long."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    words = _SYNTHETIC_BLOCK.split()
    header = (
        f"Context marker [{seed_tag}]. Read quietly and follow only the final line.\n\n"
    )
    repeats = max(1, target_tokens // len(words))
    body_parts = [
        f"{seed_tag}-{idx % 997}: " + _SYNTHETIC_BLOCK for idx in range(repeats)
    ]
    footer = (
        "\n\nIgnore the filler tokens above and write a concise 40-word technical "
        "summary explaining why deterministic KV cache quantization helps local LLM serving."
    )
    return header + "".join(body_parts)[: max(200, target_tokens * 4)] + footer


def _stream_completion(
    endpoint: str,
    body: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = dict(body, stream=True, stream_options={"include_usage": True})
    t0 = time.monotonic()
    ttft: float | None = None
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    finish_reason: str | None = None

    with requests.post(
        url,
        headers=auth_headers(api_key),
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            chunk_str = line[5:].strip()
            if chunk_str == "[DONE]":
                break
            try:
                obj = json.loads(chunk_str)
            except ValueError:
                continue
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                if (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("tool_calls")
                ) and ttft is None:
                    ttft = time.monotonic() - t0
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    wall = max(time.monotonic() - t0, 1e-6)
    if ttft is None:
        ttft = wall

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    decode_elapsed = max(wall - ttft, 1e-6)

    prefill_tok_s = timings.get("prompt_per_second")
    if not isinstance(prefill_tok_s, (int, float)) or prefill_tok_s <= 0:
        prefill_tok_s = prompt_tokens / max(ttft, 1e-6) if prompt_tokens else 0.0

    decode_tok_s = timings.get("predicted_per_second")
    if not isinstance(decode_tok_s, (int, float)) or decode_tok_s <= 0:
        gen_count = max(completion_tokens - 1, 1) if completion_tokens else 0
        decode_tok_s = gen_count / decode_elapsed if gen_count else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 4),
        "wall_s": round(wall, 4),
        "prefill_tok_s": round(float(prefill_tok_s), 2),
        "decode_tok_s": round(float(decode_tok_s), 2),
        "finish_reason": finish_reason,
        "content": "".join(content_parts),
    }


def run_prefill_decode_curve(
    endpoint: str,
    model: str,
    *,
    contexts: tuple[int, ...] = (4096, 16384),
    max_tokens: int = 128,
    api_key: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Measure prefill and decode throughput across target context sizes."""
    validate_endpoint_url(endpoint, allow_private=True)
    curves: list[dict[str, Any]] = []
    errors: list[str] = []
    for idx, ctx in enumerate(contexts):
        prompt = build_synthetic_prompt(ctx, seed_tag=f"curve-{ctx}-{idx}")
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            metrics = _stream_completion(
                endpoint, body, api_key=api_key, timeout=timeout
            )
            curves.append({"target_context": ctx, **metrics})
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"ctx={ctx}: {exc}")

    valid_pp = [c["prefill_tok_s"] for c in curves if c["prefill_tok_s"] > 0]
    valid_tg = [c["decode_tok_s"] for c in curves if c["decode_tok_s"] > 0]
    ttfts = [c["ttft_s"] for c in curves]

    return {
        "prefill_speed": round(sum(valid_pp) / len(valid_pp), 2) if valid_pp else 0.0,
        "decode_speed": round(sum(valid_tg) / len(valid_tg), 2) if valid_tg else 0.0,
        "ttft": round(sum(ttfts) / len(ttfts), 4) if ttfts else 0.0,
        "curves": curves,
        "errors": errors,
    }


def run_prefix_cache_probe(
    endpoint: str,
    model: str,
    *,
    prefix_tokens: int = 4096,
    api_key: str | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Measure cold vs warm TTFT speedup on a shared multi-turn system prefix."""
    validate_endpoint_url(endpoint, allow_private=True)
    shared_sys = "System specification cache block:\n" + (
        "tool_policy=strict audit_mode=enabled " * max(10, prefix_tokens // 5)
    )
    cold_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": shared_sys},
            {"role": "user", "content": "Reply concisely: step one ready."},
        ],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    warm_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": shared_sys},
            {"role": "user", "content": "Reply concisely: step two ready."},
        ],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        cold = _stream_completion(endpoint, cold_body, api_key=api_key, timeout=timeout)
        warm = _stream_completion(endpoint, warm_body, api_key=api_key, timeout=timeout)
        speedup = round(cold["ttft_s"] / max(warm["ttft_s"], 1e-4), 2)
        return {
            "status": "ok",
            "cold_ttft_s": cold["ttft_s"],
            "warm_ttft_s": warm["ttft_s"],
            "speedup": speedup,
            "prompt_tokens": warm["prompt_tokens"],
        }
    except (requests.RequestException, ValueError) as exc:
        return {"status": "error", "error": str(exc), "speedup": 0.0}


def run_concurrency_probe(
    endpoint: str,
    model: str,
    *,
    concurrency: int = 2,
    prompt_tokens: int = 512,
    max_tokens: int = 96,
    api_key: str | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Run ``concurrency`` overlapping requests and report aggregate decode rate."""
    validate_endpoint_url(endpoint, allow_private=True)
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    def _one(slot_idx: int) -> dict[str, Any]:
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": build_synthetic_prompt(
                        prompt_tokens, seed_tag=f"c{concurrency}-{slot_idx}"
                    ),
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return _stream_completion(endpoint, body, api_key=api_key, timeout=timeout)

    t0 = time.monotonic()
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, idx) for idx in range(concurrency)]
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

    wall = max(time.monotonic() - t0, 1e-6)
    total_tokens = sum(r["completion_tokens"] for r in results)
    per_slot = [r["decode_tok_s"] for r in results if r["decode_tok_s"] > 0]
    ttfts = [r["ttft_s"] for r in results]
    agg_tok_s = (
        round(total_tokens / wall, 2) if total_tokens > 0 else round(sum(per_slot), 2)
    )
    return {
        "concurrency": concurrency,
        "ok_count": len(results),
        "error_count": len(errors),
        "aggregate_decode_tok_s": agg_tok_s,
        "per_slot_decode_tok_s": round(sum(per_slot) / len(per_slot), 2)
        if per_slot
        else 0.0,
        "mean_ttft_s": round(sum(ttfts) / len(ttfts), 4) if ttfts else 0.0,
        "wall_s": round(wall, 3),
        "errors": errors,
    }


def run_hass_probe(
    endpoint: str,
    model: str,
    *,
    thinking: bool = False,
    iterations: int = 3,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Measure short-turn Home Assistant voice/intent TTFT with thinking disabled."""
    validate_endpoint_url(endpoint, allow_private=True)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    prompts = [
        "Turn off the kitchen ceiling lights and set living room brightness to 40 percent. Answer briefly.",
        "What is the indoor thermostat setpoint right now? Reply in one short sentence.",
        "Confirm garage door status check and schedule porch light off at 23:00.",
    ]
    ttfts: list[float] = []
    walls: list[float] = []
    decodes: list[float] = []
    errors: list[str] = []

    for idx in range(iterations):
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise Home Assistant voice assistant.",
                },
                {"role": "user", "content": prompts[idx % len(prompts)]},
            ],
            "max_tokens": 96,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        try:
            res = _stream_completion(endpoint, body, api_key=api_key, timeout=timeout)
            ttfts.append(res["ttft_s"])
            walls.append(res["wall_s"])
            if res["decode_tok_s"] > 0:
                decodes.append(res["decode_tok_s"])
        except (requests.RequestException, ValueError) as exc:
            errors.append(str(exc))

    sorted_ttfts = sorted(ttfts)
    p95_idx = min(len(sorted_ttfts) - 1, math.ceil(0.95 * len(sorted_ttfts)) - 1)
    return {
        "iterations": iterations,
        "ok_count": len(ttfts),
        "error_count": len(errors),
        "thinking": thinking,
        "mean_ttft_s": round(sum(ttfts) / len(ttfts), 4) if ttfts else 0.0,
        "p95_ttft_s": sorted_ttfts[p95_idx] if sorted_ttfts else 0.0,
        "mean_wall_s": round(sum(walls) / len(walls), 4) if walls else 0.0,
        "decode_tok_s": round(sum(decodes) / len(decodes), 2) if decodes else 0.0,
        "errors": errors,
    }


def measure_throughput_suite(
    endpoint: str,
    model: str,
    *,
    profile: str = "quick",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run unified throughput, prefix cache, concurrency=2, and HASS probes."""
    contexts = (8192, 32768, 65536) if profile in ("full", "gate") else (4096, 16384)
    curve = run_prefill_decode_curve(
        endpoint, model, contexts=contexts, max_tokens=128, api_key=api_key
    )
    prefix = run_prefix_cache_probe(
        endpoint,
        model,
        prefix_tokens=8192 if profile in ("full", "gate") else 4096,
        api_key=api_key,
    )
    c2 = run_concurrency_probe(
        endpoint,
        model,
        concurrency=2,
        prompt_tokens=512,
        max_tokens=96,
        api_key=api_key,
    )
    hass = run_hass_probe(
        endpoint, model, thinking=False, iterations=3, api_key=api_key
    )
    return {
        "throughput_metrics": {
            "prefill_speed": curve["prefill_speed"],
            "decode_speed": curve["decode_speed"],
            "ttft": curve["ttft"],
            "curves": curve["curves"],
            "errors": curve["errors"],
        },
        "prefix_cache_metrics": prefix,
        "concurrency_metrics": c2,
        "hass_probe": hass,
    }
