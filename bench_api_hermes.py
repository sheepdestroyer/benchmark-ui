"""OpenAI/Hermes API compatibility checks, streaming replay, and disconnect recovery."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from bench_throughput import auth_headers
from utils import validate_endpoint_url

_RAW_LEAK_TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>")


def synthetic_hermes_fixtures() -> list[tuple[str, dict[str, Any]]]:
    """Built-in minimal synthetic Hermes-shaped tool requests for CI and smoke checks."""
    tool = {
        "type": "function",
        "function": {
            "name": "get_service_status",
            "description": "Check systemd unit status.",
            "parameters": {
                "type": "object",
                "properties": {"unit": {"type": "string"}},
                "required": ["unit"],
            },
        },
    }
    return [
        (
            "synthetic_h01.json",
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Hermes Agent. Call tools when asked to inspect a service.",
                    },
                    {
                        "role": "user",
                        "content": "Check llama-router.service using get_service_status.",
                    },
                ],
                "tools": [tool],
                "tool_choice": "auto",
                "max_tokens": 512,
                "temperature": 0,
            },
        ),
        (
            "synthetic_h02.json",
            {
                "messages": [
                    {"role": "system", "content": "You are Hermes Agent. Be direct."},
                    {"role": "user", "content": "Reply with OK and nothing else."},
                ],
                "max_tokens": 64,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
    ]


def load_hermes_fixtures(
    fixtures_dir: str | Path | None,
    limit: int | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Load Hermes JSON payloads from disk or fall back to synthetic cases."""
    if not fixtures_dir:
        items = synthetic_hermes_fixtures()
    else:
        path = Path(fixtures_dir)
        files = sorted(path.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No Hermes JSON fixtures found in {path}")
        items = [(f.name, json.loads(f.read_text(encoding="utf-8"))) for f in files]
    return items[:limit] if limit is not None and limit > 0 else items


def stream_chat_completion(
    endpoint: str,
    body: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout: float = 600.0,
    stop_after_events: int | None = None,
) -> dict[str, Any]:
    """Stream ``/v1/chat/completions`` SSE frames and merge deltas."""
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = dict(body, stream=True)
    t0 = time.monotonic()
    res: dict[str, Any] = {
        "status": 0,
        "events": 0,
        "bad": 0,
        "content": "",
        "reasoning": "",
        "tool_calls": {},
        "finish": None,
        "usage": None,
        "ttft": None,
        "wall": 0.0,
        "error": None,
    }
    try:
        with requests.post(
            url,
            headers=auth_headers(api_key),
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            res["status"] = resp.status_code
            if resp.status_code != 200:
                res["error"] = resp.text[:300]
                return res
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except ValueError:
                    res["bad"] += 1
                    continue
                res["events"] += 1
                if obj.get("error"):
                    res["error"] = json.dumps(obj["error"])[:300]
                if obj.get("usage"):
                    res["usage"] = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if (
                        d.get("content")
                        or d.get("reasoning_content")
                        or d.get("tool_calls")
                    ) and res["ttft"] is None:
                        res["ttft"] = round(time.monotonic() - t0, 4)
                    res["content"] += d.get("content") or ""
                    res["reasoning"] += d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        slot = res["tool_calls"].setdefault(
                            tc.get("index", 0),
                            {"id": None, "name": "", "arguments": ""},
                        )
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    res["finish"] = ch.get("finish_reason") or res["finish"]
                if stop_after_events and res["events"] >= stop_after_events:
                    break
    except requests.RequestException as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        res["wall"] = round(time.monotonic() - t0, 4)
    return res


def check_hermes_stream_response(
    name: str,
    res: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    require_usage: bool = True,
) -> dict[str, Any]:
    """Validate one streamed Hermes response for SSE hygiene and tool-call structure."""
    problems: list[str] = []
    if res["status"] != 200:
        problems.append(f"HTTP {res['status']} {res.get('error')}")
    if res["bad"]:
        problems.append(f"{res['bad']} bad SSE lines")
    if res["error"] and res["status"] == 200:
        problems.append(str(res["error"]))
    if res["finish"] not in ("stop", "tool_calls", "length"):
        problems.append(f"finish={res['finish']!r}")

    leaked = [tag for tag in _RAW_LEAK_TAGS if tag in res["content"]]
    if leaked:
        problems.append(f"raw tags in content: {leaked}")

    allowed_names = {
        t["function"]["name"]
        for t in (tools or [])
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    malformed_tools = 0
    for tc in res["tool_calls"].values():
        if allowed_names and tc["name"] not in allowed_names:
            problems.append(f"unknown tool {tc['name']!r}")
            malformed_tools += 1
        if not tc["id"]:
            problems.append(f"tool {tc['name']} missing id")
            malformed_tools += 1
        if res["finish"] != "length":
            try:
                json.loads(tc["arguments"] or "{}")
            except ValueError:
                problems.append(f"tool {tc['name']} invalid JSON arguments")
                malformed_tools += 1

    if require_usage and res["status"] == 200 and not problems and not res["usage"]:
        problems.append("missing usage block with include_usage=True")

    return {
        "name": name,
        "passed": not problems,
        "problems": problems,
        "malformed_tools": malformed_tools,
        "raw_tag_leaks": len(leaked),
        "tool_call_count": len(res["tool_calls"]),
        "finish": res["finish"],
        "ttft_s": res["ttft"],
        "wall_s": res["wall"],
    }


def _call_json(
    endpoint: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    raw: bytes | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> tuple[int, Any]:
    url = f"{endpoint.rstrip('/')}{path}"
    headers = auth_headers(api_key)
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=timeout)
        elif raw is not None:
            resp = requests.post(url, headers=headers, data=raw, timeout=timeout)
        else:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        try:
            parsed = resp.json()
        except ValueError:
            parsed = resp.text
        return resp.status_code, parsed
    except requests.RequestException as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def _extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def run_api_edges_suite(
    endpoint: str,
    model: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Verify core OpenAI/Anthropic-compatible API contracts relied on by clients."""
    validate_endpoint_url(endpoint, allow_private=True)
    checks: list[dict[str, Any]] = []

    st, d = _call_json(endpoint, "GET", "/v1/models", api_key=api_key)
    ids = [m.get("id") for m in d.get("data", [])] if isinstance(d, dict) else []
    checks.append(
        {
            "name": "models_list",
            "passed": st == 200 and model in ids,
            "detail": f"HTTP {st}",
        }
    )

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    st, d = _call_json(
        endpoint,
        "POST",
        "/v1/chat/completions",
        body={
            "model": model,
            "messages": [{"role": "user", "content": "Invent a person: name and age."}],
            "max_tokens": 256,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": schema, "strict": True},
            },
        },
        api_key=api_key,
    )
    schema_ok = False
    if st == 200:
        try:
            obj = json.loads(_extract_text(d))
            schema_ok = set(obj) == {"name", "age"} and isinstance(obj["age"], int)
        except (ValueError, TypeError):
            schema_ok = False
    checks.append(
        {"name": "json_schema_strict", "passed": schema_ok, "detail": f"HTTP {st}"}
    )

    st, d = _call_json(
        endpoint,
        "POST",
        "/v1/chat/completions",
        body={
            "model": model,
            "messages": [
                {"role": "user", "content": "Count 1 to 15 separated by commas."}
            ],
            "max_tokens": 128,
            "temperature": 0,
            "stop": [", 7"],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        api_key=api_key,
    )
    txt = _extract_text(d)
    checks.append(
        {
            "name": "stop_sequence",
            "passed": st == 200 and "6" in txt and "7" not in txt,
            "detail": f"HTTP {st}",
        }
    )

    st, d = _call_json(
        endpoint,
        "POST",
        "/v1/chat/completions",
        body={
            "model": model,
            "messages": [{"role": "user", "content": "What is 17 times 3?"}],
            "max_tokens": 128,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        api_key=api_key,
    )
    msg = (
        (d.get("choices") or [{}])[0].get("message") or {}
        if isinstance(d, dict)
        else {}
    )
    checks.append(
        {
            "name": "thinking_off",
            "passed": st == 200
            and not msg.get("reasoning_content")
            and "51" in _extract_text(d),
            "detail": f"HTTP {st}",
        }
    )

    st_bad, _ = _call_json(
        endpoint, "POST", "/v1/chat/completions", raw=b"{not json", api_key=api_key
    )
    st_empty, _ = _call_json(
        endpoint,
        "POST",
        "/v1/chat/completions",
        body={"model": model, "messages": [], "max_tokens": 8},
        api_key=api_key,
    )
    st_rec, d_rec = _call_json(
        endpoint,
        "POST",
        "/v1/chat/completions",
        body={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with only 42."}],
            "max_tokens": 16,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        api_key=api_key,
    )
    checks.append(
        {
            "name": "error_handling_and_recovery",
            "passed": st_bad == 400
            and 400 <= st_empty < 500
            and st_rec == 200
            and "42" in _extract_text(d_rec),
            "detail": f"bad={st_bad} empty={st_empty} rec={st_rec}",
        }
    )

    passed = sum(1 for c in checks if c["passed"])
    return {"passed": passed, "total": len(checks), "checks": checks}


def run_disconnect_recovery_probe(
    endpoint: str,
    model: str,
    *,
    api_key: str | None = None,
    settle_s: float = 0.5,
) -> dict[str, Any]:
    """Ensure aborted SSE streams release server slots immediately."""
    validate_endpoint_url(endpoint, allow_private=True)
    aborted = stream_chat_completion(
        endpoint,
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Write a 2000-word essay about canals."}
            ],
            "max_tokens": 2048,
            "temperature": 0,
        },
        api_key=api_key,
        stop_after_events=5,
    )
    if settle_s > 0:
        time.sleep(settle_s)

    def _followup(idx: int) -> dict[str, Any]:
        return stream_chat_completion(
            endpoint,
            {
                "model": model,
                "messages": [{"role": "user", "content": f"Reply OK-{idx}."}],
                "max_tokens": 32,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            api_key=api_key,
            timeout=30.0,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_followup, (0, 1)))
    passed = all(r["status"] == 200 for r in results)
    return {
        "passed": passed,
        "aborted_events": aborted["events"],
        "followup_statuses": [r["status"] for r in results],
    }


def run_hermes_replay_suite(
    endpoint: str,
    model: str,
    *,
    fixtures_dir: str | Path | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    workers: int = 2,
    limit: int | None = None,
) -> dict[str, Any]:
    """Replay Hermes request fixtures with concurrent streaming and SSE validation."""
    validate_endpoint_url(endpoint, allow_private=True)
    fixtures = load_hermes_fixtures(fixtures_dir, limit=limit)

    def _run_one(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        name, payload = item
        body = dict(payload)
        body["model"] = model
        body["max_tokens"] = min(int(body.get("max_tokens") or max_tokens), max_tokens)
        body["stream_options"] = {"include_usage": True}
        streamed = stream_chat_completion(endpoint, body, api_key=api_key)
        return check_hermes_stream_response(
            name, streamed, tools=body.get("tools"), require_usage=True
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(_run_one, fixtures))

    passed = sum(1 for r in results if r["passed"])
    ttfts = [r["ttft_s"] for r in results if isinstance(r["ttft_s"], (int, float))]
    walls = [r["wall_s"] for r in results]
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "malformed_tool_count": sum(r["malformed_tools"] for r in results),
        "raw_tag_leak_count": sum(r["raw_tag_leaks"] for r in results),
        "total_tool_calls": sum(r["tool_call_count"] for r in results),
        "mean_ttft_s": round(sum(ttfts) / len(ttfts), 4) if ttfts else 0.0,
        "mean_wall_s": round(sum(walls) / len(walls), 4) if walls else 0.0,
        "results": results,
    }


def run_api_and_hermes_suite(
    endpoint: str,
    model: str,
    *,
    profile: str = "quick",
    fixtures_dir: str | Path | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run API edge checks, Hermes fixture replays, and disconnect recovery."""
    edges = run_api_edges_suite(endpoint, model, api_key=api_key)
    limit = None if profile in ("full", "gate") else 8
    replay = run_hermes_replay_suite(
        endpoint, model, fixtures_dir=fixtures_dir, api_key=api_key, limit=limit
    )
    disconnect = (
        run_disconnect_recovery_probe(endpoint, model, api_key=api_key)
        if profile in ("full", "gate")
        else None
    )
    return {
        "api_edges": edges,
        "hermes_replay": replay,
        "disconnect_recovery": disconnect,
    }
