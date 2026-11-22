"""Unit tests for bench_mrcr.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

import bench_mrcr as bm


def test_load_mrcr_tasks_variants():
    quick_syn = bm.load_mrcr_tasks(None, profile="quick")
    assert len(quick_syn) == 3
    assert bm.load_mrcr_tasks(None, limit_per_bin=0) == []

    tasks = [
        {
            "id": "mrcr8-64k-0",
            "kind": "mrcr",
            "target": {"prefix": "P ", "answer": "P x"},
            "messages": [],
        },
        {
            "id": "mrcr8-64k-1",
            "kind": "mrcr",
            "target": {"prefix": "P ", "answer": "P y"},
            "messages": [],
        },
        {
            "id": "mrcr8-64k-2",
            "kind": "mrcr",
            "target": {"prefix": "P ", "answer": "P z"},
            "messages": [],
        },
        {"id": "extract-32k", "kind": "extract", "target": [], "messages": []},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp) / "tasks.json"
        tp.write_text(json.dumps(tasks), encoding="utf-8")
        assert len(bm.load_mrcr_tasks(tp, profile="quick")) == 3
        assert len(bm.load_mrcr_tasks(tp, profile="full")) == 4


def test_score_mrcr_task_all_kinds():
    mrcr_task = {
        "id": "mrcr8-128k-0",
        "kind": "mrcr",
        "target": {"prefix": "TAG ", "answer": "TAG alpha beta"},
    }
    assert (
        bm.score_mrcr_task(
            mrcr_task, {"choices": [{"message": {"content": "no prefix"}}]}
        )["score"]
        == 0.0
    )
    assert (
        bm.score_mrcr_task(
            mrcr_task, {"choices": [{"message": {"content": "TAG alpha beta"}}]}
        )["score"]
        == 1.0
    )
    assert bm.score_mrcr_task(mrcr_task, {"error": "boom"})["score"] == 0.0

    ext_task = {
        "id": "extract-32k",
        "kind": "extract",
        "target": [["K17-1", "A", "2024-01-01", "10.00", "Lyon"]],
    }
    ext_resp = {
        "choices": [
            {
                "message": {
                    "content": "```csv\nReference,Holder,Date,Amount (EUR),City\nK17-1,A,2024-01-01,10.00,Lyon\n```"
                }
            }
        ]
    }
    assert bm.score_mrcr_task(ext_task, ext_resp)["ratio"] == 1.0
    assert bm.score_mrcr_task(ext_task, {"error": "timeout"})["score"] == 0.0

    facts_task = {
        "id": "facts-32k",
        "kind": "facts",
        "target": {"C-1": {"date": "2027-01-01", "counterparty": "Jean Doe"}},
    }
    assert (
        bm.score_mrcr_task(
            facts_task,
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"C-1": {"date": "2027-01-01", "counterparty": "Jean Doe"}}'
                        }
                    }
                ]
            },
        )["ratio"]
        == 1.0
    )
    assert (
        bm.score_mrcr_task(
            facts_task, {"choices": [{"message": {"content": "{bad}"}}]}
        )["detail"]
        == "bad JSON"
    )
    assert (
        bm.score_mrcr_task(
            facts_task, {"choices": [{"message": {"content": "no braces"}}]}
        )["score"]
        == 0.0
    )
    assert bm.score_mrcr_task(facts_task, {"error": "fail"})["total"] == 2.0

    agent_task = {
        "id": "agent-32k",
        "kind": "agent",
        "target": {"MAX_TOKENS": "12000"},
    }
    agent_ok = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "report_values",
                                "arguments": '{"values": {"MAX_TOKENS": "12000"}}',
                            }
                        }
                    ],
                }
            }
        ]
    }
    assert bm.score_mrcr_task(agent_task, agent_ok)["ratio"] == 1.0
    agent_bad_args = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "report_values", "arguments": "{"}}
                    ],
                }
            }
        ]
    }
    assert bm.score_mrcr_task(agent_task, agent_bad_args)["detail"] == "bad args"
    assert (
        bm.score_mrcr_task(agent_task, {"choices": [{"message": {"content": "hi"}}]})[
            "detail"
        ]
        == "no tool call"
    )


def test_run_mrcr_suite_paths():
    ok_resp = MagicMock()
    ok_resp.raise_for_status = MagicMock()
    ok_resp.json.side_effect = [
        {
            "choices": [{"message": {"content": "TAG_A alpha beta gamma"}}],
            "usage": {"prompt_tokens": 20},
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "K17-1001,Alice Moreau,2024-01-10,1200.00,Lyon\n"
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": '{"C-2040":{"date":"2027-03-15","counterparty":"Chloé Garnier"}}'
                    }
                }
            ]
        },
    ]
    with patch("bench_mrcr.requests.post", return_value=ok_resp):
        res = bm.run_mrcr_suite("http://127.0.0.1:8083", "m", profile="quick")
    assert res["mrcr_mean"] == 1.0
    assert res["mrcr_by_bin"] == {"<64k": 1.0}
    assert res["errors"] == 0

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp) / "t.json"
        tp.write_text(
            json.dumps(
                [
                    {
                        "id": "agent-32k",
                        "kind": "agent",
                        "target": {"A": "1"},
                        "messages": [],
                        "tools": [{"type": "function"}],
                    }
                ]
            ),
            encoding="utf-8",
        )
        with patch(
            "bench_mrcr.requests.post",
            side_effect=requests.RequestException("down"),
        ):
            res_err = bm.run_mrcr_suite(
                "http://127.0.0.1:8083", "m", tasks_path=tp, profile="full"
            )
    assert res_err["errors"] == 1
    assert res_err["mrcr_mean"] is None

    empty = bm.run_mrcr_suite("http://127.0.0.1:8083", "m", limit_per_bin=0)
    assert empty["tasks_run"] == 0 and empty["custom_ratio"] is None
