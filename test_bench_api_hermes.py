"""Unit tests for bench_api_hermes.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import bench_api_hermes as bah


def _mock_stream(lines, status=200, text="err"):
    resp = MagicMock(status_code=status, text=text)
    resp.iter_lines.return_value = iter(lines)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_load_hermes_fixtures():
    syn = bah.load_hermes_fixtures(None)
    assert len(syn) == 2
    assert len(bah.load_hermes_fixtures(None, limit=1)) == 1

    with tempfile.TemporaryDirectory() as tmp:
        empty_dir = Path(tmp) / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            bah.load_hermes_fixtures(empty_dir)

        f1 = Path(tmp) / "h01.json"
        f1.write_text(json.dumps({"messages": []}), encoding="utf-8")
        loaded = bah.load_hermes_fixtures(tmp)
        assert loaded == [("h01.json", {"messages": []})]


def test_stream_chat_completion_variants():
    lines = [
        "",
        "event: ping",
        "data: {bad",
        'data: {"error":{"message":"engine died"}}',
        'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"bash","arguments":"{\\"cmd\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"ls\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"total_tokens":10}}',
        "data: [DONE]",
    ]
    with patch("bench_api_hermes.requests.post", return_value=_mock_stream(lines)):
        res = bah.stream_chat_completion("http://127.0.0.1:8083", {"model": "m"})
    assert res["status"] == 200
    assert res["bad"] == 1
    assert "engine died" in res["error"]
    assert res["tool_calls"][0]["arguments"] == '{"cmd":"ls"}'

    stop_lines = [
        'data: {"choices":[{"delta":{"content":"a"}}]}',
        'data: {"choices":[{"delta":{"content":"b"}}]}',
    ]
    with patch("bench_api_hermes.requests.post", return_value=_mock_stream(stop_lines)):
        res_stop = bah.stream_chat_completion(
            "http://127.0.0.1:8083", {"model": "m"}, stop_after_events=1
        )
    assert res_stop["events"] == 1

    no_done_lines = ['data: {"choices":[{"delta":{"content":"only"}}]}']
    with patch(
        "bench_api_hermes.requests.post", return_value=_mock_stream(no_done_lines)
    ):
        res_nodone = bah.stream_chat_completion("http://127.0.0.1:8083", {"model": "m"})
    assert res_nodone["content"] == "only"

    with patch(
        "bench_api_hermes.requests.post",
        return_value=_mock_stream([], status=500, text="server error"),
    ):
        res_500 = bah.stream_chat_completion("http://127.0.0.1:8083", {"model": "m"})
    assert res_500["status"] == 500 and "server error" in res_500["error"]

    with patch(
        "bench_api_hermes.requests.post",
        side_effect=requests.RequestException("conn reset"),
    ):
        res_exc = bah.stream_chat_completion("http://127.0.0.1:8083", {"model": "m"})
    assert res_exc["status"] == 0 and "RequestException" in res_exc["error"]


def test_check_hermes_stream_response():
    tools = [{"type": "function", "function": {"name": "bash"}}]
    ok_res = {
        "status": 200,
        "bad": 0,
        "error": None,
        "finish": "tool_calls",
        "content": "",
        "tool_calls": {0: {"id": "c1", "name": "bash", "arguments": '{"cmd":"ls"}'}},
        "usage": {"prompt_tokens": 10},
        "ttft": 0.1,
        "wall": 0.5,
    }
    chk_ok = bah.check_hermes_stream_response("h01", ok_res, tools=tools)
    assert chk_ok["passed"] is True and chk_ok["malformed_tools"] == 0

    bad_res = {
        "status": 500,
        "bad": 1,
        "error": "oops",
        "finish": "weird",
        "content": "<think>leak</think>",
        "tool_calls": {
            0: {"id": None, "name": "unknown_tool", "arguments": "{not json"}
        },
        "usage": None,
        "ttft": 0.1,
        "wall": 0.5,
    }
    chk_bad = bah.check_hermes_stream_response("h02", bad_res, tools=tools)
    assert chk_bad["passed"] is False and chk_bad["malformed_tools"] == 3

    len_res = {
        "status": 200,
        "bad": 0,
        "error": "mid-stream err",
        "finish": "length",
        "content": "",
        "tool_calls": {0: {"id": "c1", "name": "bash", "arguments": "{partial"}},
        "usage": None,
        "ttft": 0.1,
        "wall": 0.5,
    }
    chk_len = bah.check_hermes_stream_response(
        "h03", len_res, tools=["non-dict"], require_usage=True
    )
    assert chk_len["passed"] is False

    no_usage_res = dict(ok_res, usage=None, finish="stop", tool_calls={})
    chk_nu = bah.check_hermes_stream_response("h04", no_usage_res, require_usage=True)
    assert chk_nu["passed"] is False


def test_call_json_and_extract_text():
    get_ok = MagicMock(status_code=200)
    get_ok.json.return_value = {"data": []}
    post_text = MagicMock(status_code=400, text="bad json")
    post_text.json.side_effect = ValueError("not json")

    with patch("bench_api_hermes.requests.get", return_value=get_ok):
        assert bah._call_json("http://127.0.0.1:8083", "GET", "/v1/models") == (
            200,
            {"data": []},
        )
    with patch("bench_api_hermes.requests.post", return_value=post_text):
        assert bah._call_json(
            "http://127.0.0.1:8083", "POST", "/v1/chat/completions", raw=b"{"
        ) == (400, "bad json")
    with patch(
        "bench_api_hermes.requests.post",
        side_effect=requests.RequestException("down"),
    ):
        st, _ = bah._call_json(
            "http://127.0.0.1:8083", "POST", "/v1/chat/completions", body={}
        )
        assert st == 0

    assert bah._extract_text("bad") == ""
    assert bah._extract_text({"choices": []}) == ""
    assert bah._extract_text({"choices": [{"message": {"content": "hi"}}]}) == "hi"


def test_edges_disconnect_replay_and_suite():
    seq_ok = [
        (200, {"data": [{"id": "m"}]}),
        (200, {"choices": [{"message": {"content": '{"name":"A","age":30}'}}]}),
        (200, {"choices": [{"message": {"content": "1, 2, 3, 4, 5, 6"}}]}),
        (200, {"choices": [{"message": {"content": "51"}}]}),
        (400, "bad json"),
        (422, "empty"),
        (200, {"choices": [{"message": {"content": "42"}}]}),
    ]
    with patch("bench_api_hermes._call_json", side_effect=seq_ok):
        edges = bah.run_api_edges_suite("http://127.0.0.1:8083", "m")
    assert edges["passed"] == 5 and edges["total"] == 5

    seq_fail = [
        (500, {}),
        (500, {}),
        (500, {}),
        (500, {}),
        (500, {}),
        (500, {}),
        (500, {}),
    ]
    with patch("bench_api_hermes._call_json", side_effect=seq_fail):
        edges_fail = bah.run_api_edges_suite("http://127.0.0.1:8083", "m")
    assert edges_fail["passed"] == 0

    seq_bad_schema = [
        (200, {"data": []}),
        (200, {"choices": [{"message": {"content": "{not-json"}}]}),
        (500, {}),
        (500, {}),
        (500, {}),
        (500, {}),
        (500, {}),
    ]
    with patch("bench_api_hermes._call_json", side_effect=seq_bad_schema):
        edges_schema_bad = bah.run_api_edges_suite("http://127.0.0.1:8083", "m")
    assert edges_schema_bad["checks"][1]["passed"] is False

    with patch(
        "bench_api_hermes.stream_chat_completion",
        side_effect=[
            {"events": 5, "status": 200},
            {"status": 200},
            {"status": 200},
        ],
    ):
        disc = bah.run_disconnect_recovery_probe(
            "http://127.0.0.1:8083", "m", settle_s=0.0
        )
    assert disc["passed"] is True

    with patch(
        "bench_api_hermes.stream_chat_completion",
        side_effect=[
            {"events": 1, "status": 200},
            {"status": 503},
            {"status": 200},
        ],
    ):
        disc_s = bah.run_disconnect_recovery_probe(
            "http://127.0.0.1:8083", "m", settle_s=0.001
        )
    assert disc_s["passed"] is False

    ok_stream = {
        "status": 200,
        "bad": 0,
        "error": None,
        "finish": "stop",
        "content": "OK",
        "tool_calls": {},
        "usage": {"total_tokens": 5},
        "ttft": 0.05,
        "wall": 0.1,
    }
    with patch("bench_api_hermes.stream_chat_completion", return_value=ok_stream):
        rep = bah.run_hermes_replay_suite("http://127.0.0.1:8083", "m")
    assert rep["passed"] == 2 and rep["failed"] == 0

    with (
        patch(
            "bench_api_hermes.run_api_edges_suite",
            return_value={"passed": 5, "total": 5},
        ),
        patch(
            "bench_api_hermes.run_hermes_replay_suite",
            return_value={"passed": 2, "total": 2},
        ),
        patch(
            "bench_api_hermes.run_disconnect_recovery_probe",
            return_value={"passed": True},
        ) as mock_disc,
    ):
        q = bah.run_api_and_hermes_suite("http://127.0.0.1:8083", "m", profile="quick")
        g = bah.run_api_and_hermes_suite("http://127.0.0.1:8083", "m", profile="gate")
    assert q["disconnect_recovery"] is None
    assert g["disconnect_recovery"]["passed"] is True
    mock_disc.assert_called_once()
