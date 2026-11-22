"""Multi-round co-reference (MRCR) and long-context quality benchmark harness."""

from __future__ import annotations

import csv
import difflib
import io
import json
import re
from pathlib import Path
from typing import Any

import requests

from bench_throughput import auth_headers
from utils import validate_endpoint_url


def synthetic_mrcr_tasks() -> list[dict[str, Any]]:
    """Built-in synthetic MRCR/extract/facts tasks for offline tests and fallbacks."""
    return [
        {
            "id": "mrcr8-64k-0",
            "kind": "mrcr",
            "target": {"prefix": "TAG_A ", "answer": "TAG_A alpha beta gamma"},
            "messages": [
                {
                    "role": "user",
                    "content": "Start with 'TAG_A ' then repeat alpha beta gamma.",
                }
            ],
            "max_tokens": 128,
        },
        {
            "id": "extract-32k-2rows",
            "kind": "extract",
            "target": [
                ["K17-1001", "Alice Moreau", "2024-01-10", "1200.00", "Lyon"],
                ["K17-1002", "Bastien Lefevre", "2024-02-14", "850.50", "Nancy"],
            ],
            "messages": [{"role": "user", "content": "Output CSV rows."}],
            "max_tokens": 256,
        },
        {
            "id": "facts-32k",
            "kind": "facts",
            "target": {
                "C-2040": {"date": "2027-03-15", "counterparty": "Chloé Garnier"}
            },
            "messages": [{"role": "user", "content": "Output contract JSON."}],
            "max_tokens": 256,
        },
    ]


def _mrcr_bin(task_id: str) -> str:
    match = re.match(r"mrcr8?-(64k|128k|240k)-", task_id)
    return f"<{match.group(1)}" if match else "other"


def load_mrcr_tasks(
    tasks_path: str | Path | None,
    *,
    profile: str = "quick",
    limit_per_bin: int | None = None,
) -> list[dict[str, Any]]:
    """Load MRCR/LCQ tasks from JSON file and apply tier-appropriate sampling."""
    if not tasks_path:
        tasks = synthetic_mrcr_tasks()
    else:
        path = Path(tasks_path)
        tasks = json.loads(path.read_text(encoding="utf-8"))

    if limit_per_bin is None:
        if profile == "quick":
            limit_per_bin = 2
        else:
            return tasks

    if limit_per_bin <= 0:
        return []

    counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for task in tasks:
        key = (
            f"mrcr:{_mrcr_bin(task['id'])}"
            if task.get("kind") == "mrcr"
            else f"kind:{task.get('kind', 'other')}"
        )
        seen = counts.get(key, 0)
        if seen < limit_per_bin:
            counts[key] = seen + 1
            selected.append(task)
    return selected


def _norm_cell(val: Any) -> str:
    return re.sub(r"\s+", " ", str(val)).strip().strip("'\"").strip()


def score_mrcr_task(task: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Score one MRCR, extract CSV, facts JSON, or agentic tool-recall completion."""
    kind = task.get("kind") or (
        "mrcr" if str(task.get("id", "")).startswith("mrcr") else "other"
    )
    if "error" in response and response["error"]:
        total = (
            1.0
            if kind == "mrcr"
            else float(len(task["target"]) * (2 if kind == "facts" else 1))
        )
        return {
            "id": task["id"],
            "kind": kind,
            "score": 0.0,
            "total": total,
            "ratio": 0.0,
            "detail": f"ERROR: {response['error']}",
        }

    msg = (response.get("choices") or [{}])[0].get("message") or {}
    content = str(msg.get("content") or "")
    tool_calls = msg.get("tool_calls") or []

    if kind == "mrcr":
        target = task["target"]
        prefix = target["prefix"]
        if not content.startswith(prefix):
            return {
                "id": task["id"],
                "kind": kind,
                "score": 0.0,
                "total": 1.0,
                "ratio": 0.0,
                "detail": "missing prefix",
            }
        sim = difflib.SequenceMatcher(
            None,
            content.removeprefix(prefix),
            target["answer"].removeprefix(prefix),
        ).ratio()
        return {
            "id": task["id"],
            "kind": kind,
            "score": round(sim, 6),
            "total": 1.0,
            "ratio": round(sim, 6),
            "detail": "",
        }

    if kind == "extract":
        txt = re.sub(r"^```\w*\n|```$", "", content.strip(), flags=re.M)
        rows = [
            [_norm_cell(c) for c in row] for row in csv.reader(io.StringIO(txt)) if row
        ]
        rows = [row for row in rows if row and row[0] != "Reference"]
        want = [[_norm_cell(c) for c in row] for row in task["target"]]
        got = float(sum(1 for w in want if w in rows))
        total = float(max(len(want), 1))
        inorder = sum(1 for a, b in zip(rows, want) if a == b)
        return {
            "id": task["id"],
            "kind": kind,
            "score": got,
            "total": float(len(want)),
            "ratio": round(got / total, 6),
            "detail": f"rows={len(rows)}, inorder={inorder}",
        }

    if kind == "facts":
        m = re.search(r"\{.*\}", content, re.S)
        detail = ""
        try:
            parsed = json.loads(m.group(0)) if m else {}
        except ValueError:
            parsed = {}
            detail = "bad JSON"
        got = 0.0
        total = 0.0
        for cid, expected in task["target"].items():
            total += 2.0
            actual = parsed.get(cid)
            if isinstance(actual, dict):
                got += float(_norm_cell(actual.get("date")) == expected["date"])
                got += float(
                    _norm_cell(actual.get("counterparty")) == expected["counterparty"]
                )
        return {
            "id": task["id"],
            "kind": kind,
            "score": got,
            "total": total,
            "ratio": round(got / max(total, 1.0), 6),
            "detail": detail,
        }

    # Agentic tool-call recall
    vals: dict[str, Any] = {}
    detail = ""
    if (
        tool_calls
        and (tool_calls[0].get("function") or {}).get("name") == "report_values"
    ):
        try:
            args = json.loads(tool_calls[0]["function"]["arguments"])
            vals = args.get("values", args) if isinstance(args, dict) else {}
        except ValueError:
            detail = "bad args"
    else:
        detail = "no tool call"
    target = task["target"]
    got = float(
        sum(1 for k, v in target.items() if _norm_cell(vals.get(k)) == _norm_cell(v))
    )
    total = float(len(target))
    return {
        "id": task["id"],
        "kind": kind,
        "score": got,
        "total": total,
        "ratio": round(got / max(total, 1.0), 6),
        "detail": detail,
    }


def run_mrcr_suite(
    endpoint: str,
    model: str,
    *,
    tasks_path: str | Path | None = None,
    profile: str = "quick",
    api_key: str | None = None,
    limit_per_bin: int | None = None,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Run MRCR and LCQ tasks against ``/v1/chat/completions`` and aggregate scores."""
    validate_endpoint_url(endpoint, allow_private=True)
    tasks = load_mrcr_tasks(tasks_path, profile=profile, limit_per_bin=limit_per_bin)
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    headers = auth_headers(api_key)

    per_task: list[dict[str, Any]] = []
    errors = 0
    for task in tasks:
        body: dict[str, Any] = {
            "model": model,
            "messages": task["messages"],
            "max_tokens": int(task.get("max_tokens") or 1024),
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if task.get("tools"):
            body["tools"] = task["tools"]
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            errors += 1
            payload = {"error": str(exc)}
        scored = score_mrcr_task(task, payload)
        usage = payload.get("usage") if isinstance(payload, dict) else {}
        scored["prompt_tokens"] = (
            usage.get("prompt_tokens") if isinstance(usage, dict) else None
        )
        per_task.append(scored)

    mrcr_rows = [r for r in per_task if r["kind"] == "mrcr"]
    custom_rows = [r for r in per_task if r["kind"] != "mrcr"]

    by_bin: dict[str, float] = {}
    for bin_name in ("<64k", "<128k", "<240k"):
        vals = [r["ratio"] for r in mrcr_rows if _mrcr_bin(r["id"]) == bin_name]
        if vals:
            by_bin[bin_name] = round(sum(vals) / len(vals), 4)

    custom_score = sum(r["score"] for r in custom_rows)
    custom_total = sum(r["total"] for r in custom_rows)

    return {
        "tasks_run": len(per_task),
        "errors": errors,
        "mrcr_mean": round(sum(r["ratio"] for r in mrcr_rows) / len(mrcr_rows), 4)
        if mrcr_rows
        else None,
        "mrcr_by_bin": by_bin,
        "custom_score": custom_score,
        "custom_total": custom_total,
        "custom_ratio": round(custom_score / custom_total, 4)
        if custom_total > 0
        else None,
        "per_task": per_task,
    }
