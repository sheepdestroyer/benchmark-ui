"""Unit tests for bench_kld_adapters.py."""

import math
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

import bench_kld_adapters as bka


def test_parse_llama_and_basalt_outputs():
    llama_out = (
        "Final estimate: PPL = 5.1234\n"
        "Final base PPL = 5.0100\n"
        "| Mean KLD | Max KLD | Same Top |\n"
        "| 1 | 2 |\n"
        "| bad | row | x | y |\n"
        "| 0.0412 | 0.89 | 99.10% | 42 |\n"
    )
    parsed = bka.parse_llama_kld_output(llama_out)
    assert parsed["mean_kld"] == 0.0412
    assert parsed["perplexity"] == 5.1234
    assert parsed["baseline_perplexity"] == 5.01

    parsed_regex = bka.parse_llama_kld_output("Mean KLD: 0.0521\n")
    assert parsed_regex["mean_kld"] == 0.0521
    assert bka.parse_llama_kld_output("nothing here") == {}

    basalt_json = bka.parse_basalt_bench_output(
        '{"mean_kld": 0.0561, "ppl": 5.2, "same_top_pct": 98.4}'
    )
    assert basalt_json == {
        "mean_kld": 0.0561,
        "perplexity": 5.2,
        "same_top_match_percent": 98.4,
    }
    basalt_txt = bka.parse_basalt_bench_output("KLD: 0.061 PPL=5.3 Same-Top: 97.5%")
    assert basalt_txt["mean_kld"] == 0.061
    assert basalt_txt["same_top_match_percent"] == 97.5
    assert bka.parse_basalt_bench_output("empty") == {}


def test_run_kld_subprocess():
    with pytest.raises(ValueError, match="non-empty"):
        bka.run_kld_subprocess("llama.cpp", [])

    proc = subprocess.CompletedProcess(
        args=["basalt-bench"],
        returncode=0,
        stdout='{"kld": 0.05, "perplexity": 5.1}',
        stderr="",
    )
    with patch("bench_kld_adapters.subprocess.run", return_value=proc):
        res = bka.run_kld_subprocess("basalt", ["basalt-bench"])
    assert res["mean_kld"] == 0.05 and res["perplexity"] == 5.1

    proc2 = subprocess.CompletedProcess(
        args=["llama-perplexity"],
        returncode=0,
        stdout="Mean KLD: 0.03",
        stderr="",
    )
    with patch("bench_kld_adapters.subprocess.run", return_value=proc2):
        res2 = bka.run_kld_subprocess("llama.cpp", ["llama-perplexity"])
    assert res2["mean_kld"] == 0.03


def test_run_reference_ppl_probe():
    ok = MagicMock(status_code=200)
    ok.json.return_value = {
        "choices": [{"logprobs": {"token_logprobs": [None, -1.0, -2.0]}}]
    }
    with patch("bench_kld_adapters.requests.post", return_value=ok):
        res = bka.run_reference_ppl_probe("http://127.0.0.1:8083", "m", "hello world")
    assert res["supported"] is True
    assert res["tokens_scored"] == 2
    assert res["perplexity"] == round(math.exp(1.5), 4)

    with patch(
        "bench_kld_adapters.requests.post",
        return_value=MagicMock(status_code=400),
    ):
        assert bka.run_reference_ppl_probe("http://127.0.0.1:8083", "m", "hi") == {
            "supported": False,
            "status": 400,
        }

    no_lp = MagicMock(status_code=200)
    no_lp.json.return_value = {"choices": [{}]}
    with patch("bench_kld_adapters.requests.post", return_value=no_lp):
        assert (
            bka.run_reference_ppl_probe("http://127.0.0.1:8083", "m", "hi")["supported"]
            is False
        )

    empty_vals = MagicMock(status_code=200)
    empty_vals.json.return_value = {
        "choices": [{"logprobs": {"token_logprobs": [None]}}]
    }
    with patch("bench_kld_adapters.requests.post", return_value=empty_vals):
        assert (
            bka.run_reference_ppl_probe("http://127.0.0.1:8083", "m", "hi")["supported"]
            is False
        )

    with patch(
        "bench_kld_adapters.requests.post",
        side_effect=requests.RequestException("down"),
    ):
        res_err = bka.run_reference_ppl_probe("http://127.0.0.1:8083", "m", "hi")
    assert res_err["supported"] is False and "error" in res_err
