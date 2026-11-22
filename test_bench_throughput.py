"""Unit tests for bench_throughput.py."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

import bench_throughput as bt


def _make_sse_response(lines, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError("bad status")
    resp.iter_lines.return_value = iter(lines)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_auth_headers_and_prompt_builder():
    assert bt.auth_headers(None) == {"Content-Type": "application/json"}
    assert bt.auth_headers(" ") == {"Content-Type": "application/json"}
    assert bt.auth_headers("tok")["Authorization"] == "Bearer tok"

    with pytest.raises(ValueError, match="positive"):
        bt.build_synthetic_prompt(0)
    prompt = bt.build_synthetic_prompt(64, seed_tag="s1")
    assert "s1" in prompt and len(prompt) > 200


def test_stream_completion_with_timings_and_fallbacks():
    lines_with_timings = [
        "",
        "event: message",
        "data: not-json",
        'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}',
        'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10},"timings":{"prompt_per_second":900.0,"predicted_per_second":55.5}}',
        "data: [DONE]",
    ]
    with patch(
        "bench_throughput.requests.post",
        return_value=_make_sse_response(lines_with_timings),
    ):
        res = bt._stream_completion("http://127.0.0.1:8083", {"model": "m"})
    assert res["content"] == "Hello world"
    assert res["prefill_tok_s"] == 900.0
    assert res["decode_tok_s"] == 55.5
    assert res["finish_reason"] == "stop"

    lines_no_timings = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}],"usage":{"prompt_tokens":200,"completion_tokens":21}}',
        "data: [DONE]",
    ]
    with (
        patch(
            "bench_throughput.requests.post",
            return_value=_make_sse_response(lines_no_timings),
        ),
        patch("bench_throughput.time.monotonic", side_effect=[0.0, 0.5, 1.5]),
    ):
        res2 = bt._stream_completion("http://127.0.0.1:8083", {"model": "m"})
    assert res2["prefill_tok_s"] == 400.0
    assert res2["decode_tok_s"] == 20.0

    with (
        patch(
            "bench_throughput.requests.post",
            return_value=_make_sse_response(
                ['data: {"choices":[{"delta":{"content":"x"}}]}']
            ),
        ),
        patch("bench_throughput.time.monotonic", side_effect=[10.0, 10.5, 11.0]),
    ):
        res3 = bt._stream_completion("http://127.0.0.1:8083", {"model": "m"})
    assert res3["prompt_tokens"] == 0
    assert res3["decode_tok_s"] == 0.0

    with (
        patch(
            "bench_throughput.requests.post",
            return_value=_make_sse_response(["data: [DONE]"]),
        ),
        patch("bench_throughput.time.monotonic", side_effect=[10.0, 11.0]),
    ):
        res4 = bt._stream_completion("http://127.0.0.1:8083", {"model": "m"})
    assert res4["prompt_tokens"] == 0
    assert res4["ttft_s"] == 1.0


def test_curve_prefix_concurrency_hass_and_suite():
    with (
        patch(
            "bench_throughput._stream_completion",
            side_effect=[
                {
                    "prompt_tokens": 4000,
                    "completion_tokens": 128,
                    "ttft_s": 1.0,
                    "wall_s": 3.0,
                    "prefill_tok_s": 4000.0,
                    "decode_tok_s": 60.0,
                    "finish_reason": "stop",
                    "content": "a",
                },
                requests.RequestException("timeout"),
            ],
        ),
    ):
        curve = bt.run_prefill_decode_curve(
            "http://127.0.0.1:8083", "m", contexts=(4096, 16384)
        )
    assert curve["prefill_speed"] == 4000.0
    assert len(curve["errors"]) == 1

    with patch(
        "bench_throughput._stream_completion",
        side_effect=requests.RequestException("down"),
    ):
        empty_curve = bt.run_prefill_decode_curve(
            "http://127.0.0.1:8083", "m", contexts=(4096,)
        )
    assert empty_curve["prefill_speed"] == 0.0
    assert empty_curve["decode_speed"] == 0.0

    with patch(
        "bench_throughput._stream_completion",
        side_effect=[
            {"ttft_s": 2.0, "prompt_tokens": 4096},
            {"ttft_s": 0.25, "prompt_tokens": 4096},
        ],
    ):
        pref = bt.run_prefix_cache_probe("http://127.0.0.1:8083", "m")
    assert pref["status"] == "ok" and pref["speedup"] == 8.0

    with patch(
        "bench_throughput._stream_completion",
        side_effect=ValueError("bad"),
    ):
        pref_err = bt.run_prefix_cache_probe("http://127.0.0.1:8083", "m")
    assert pref_err["status"] == "error"

    with pytest.raises(ValueError, match=">= 1"):
        bt.run_concurrency_probe("http://127.0.0.1:8083", "m", concurrency=0)

    with patch(
        "bench_throughput._stream_completion",
        side_effect=[
            {"completion_tokens": 50, "decode_tok_s": 30.0, "ttft_s": 0.2},
            {"completion_tokens": 50, "decode_tok_s": 32.0, "ttft_s": 0.3},
        ],
    ):
        c2 = bt.run_concurrency_probe("http://127.0.0.1:8083", "m", concurrency=2)
    assert c2["ok_count"] == 2 and c2["per_slot_decode_tok_s"] == 31.0

    with patch(
        "bench_throughput._stream_completion",
        side_effect=[
            {"completion_tokens": 0, "decode_tok_s": 10.0, "ttft_s": 0.1},
            RuntimeError("slot failed"),
        ],
    ):
        c2_partial = bt.run_concurrency_probe(
            "http://127.0.0.1:8083", "m", concurrency=2
        )
    assert c2_partial["error_count"] == 1
    assert c2_partial["aggregate_decode_tok_s"] == 10.0

    with patch(
        "bench_throughput._stream_completion",
        side_effect=RuntimeError("all failed"),
    ):
        c2_fail = bt.run_concurrency_probe("http://127.0.0.1:8083", "m", concurrency=1)
    assert c2_fail["per_slot_decode_tok_s"] == 0.0

    with pytest.raises(ValueError, match=">= 1"):
        bt.run_hass_probe("http://127.0.0.1:8083", "m", iterations=0)

    with patch(
        "bench_throughput._stream_completion",
        side_effect=[
            {"ttft_s": 0.4, "wall_s": 0.8, "decode_tok_s": 50.0},
            {"ttft_s": 0.5, "wall_s": 0.9, "decode_tok_s": 0.0},
            requests.RequestException("oops"),
        ],
    ):
        hass = bt.run_hass_probe("http://127.0.0.1:8083", "m", iterations=3)
    assert hass["ok_count"] == 2 and hass["error_count"] == 1

    with patch(
        "bench_throughput._stream_completion",
        side_effect=requests.RequestException("down"),
    ):
        hass_empty = bt.run_hass_probe("http://127.0.0.1:8083", "m", iterations=1)
    assert hass_empty["mean_ttft_s"] == 0.0

    with (
        patch(
            "bench_throughput.run_prefill_decode_curve",
            return_value={
                "prefill_speed": 100.0,
                "decode_speed": 40.0,
                "ttft": 0.5,
                "curves": [],
                "errors": [],
            },
        ),
        patch(
            "bench_throughput.run_prefix_cache_probe",
            return_value={"status": "ok", "speedup": 5.0},
        ),
        patch(
            "bench_throughput.run_concurrency_probe",
            return_value={"aggregate_decode_tok_s": 60.0},
        ),
        patch(
            "bench_throughput.run_hass_probe",
            return_value={"mean_ttft_s": 0.3},
        ),
    ):
        suite_q = bt.measure_throughput_suite(
            "http://127.0.0.1:8083", "m", profile="quick"
        )
        suite_g = bt.measure_throughput_suite(
            "http://127.0.0.1:8083", "m", profile="gate"
        )
    assert suite_q["prefix_cache_metrics"]["speedup"] == 5.0
    assert suite_g["hass_probe"]["mean_ttft_s"] == 0.3
