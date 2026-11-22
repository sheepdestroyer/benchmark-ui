"""Unit tests for deployment_tuple.py."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import deployment_tuple as dt_mod


def test_parse_extra_meta_variants():
    assert dt_mod.parse_extra_meta(None) == {}
    assert dt_mod.parse_extra_meta("   ") == {}
    assert dt_mod.parse_extra_meta('{"engine": "strata", "a": 1}') == {
        "engine": "strata",
        "a": 1,
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "meta.json"
        p.write_text(json.dumps({"host": "gx10"}), encoding="utf-8")
        assert dt_mod.parse_extra_meta(f"@{p}") == {"host": "gx10"}
    with pytest.raises(ValueError, match="JSON object"):
        dt_mod.parse_extra_meta("[1, 2, 3]")


def test_auth_headers_and_probe_json():
    assert dt_mod._auth_headers(None) == {}
    assert dt_mod._auth_headers("  ") == {}
    assert dt_mod._auth_headers("secret") == {"Authorization": "Bearer secret"}

    ok_resp = MagicMock(status_code=200)
    ok_resp.json.return_value = {"loaded": True}
    list_resp = MagicMock(status_code=200)
    list_resp.json.return_value = [1, 2]
    bad_status = MagicMock(status_code=503)

    with patch("deployment_tuple.requests.get", return_value=ok_resp):
        assert dt_mod._probe_json("http://127.0.0.1:8083/health", {}) == {
            "loaded": True
        }
    with patch("deployment_tuple.requests.get", return_value=list_resp):
        assert dt_mod._probe_json("http://127.0.0.1:8083/health", {}) is None
    with patch("deployment_tuple.requests.get", return_value=bad_status):
        assert dt_mod._probe_json("http://127.0.0.1:8083/health", {}) is None
    with patch(
        "deployment_tuple.requests.get",
        side_effect=requests.RequestException("down"),
    ):
        assert dt_mod._probe_json("http://127.0.0.1:8083/health", {}) is None


def test_infer_host_and_quant():
    assert dt_mod._infer_host_profile("http://127.0.0.1:8083")[0] == "x570"
    assert dt_mod._infer_host_profile("http://192.168.0.31:8080")[0] == "gx10"
    assert dt_mod._infer_host_profile("http://10.0.0.5:8080") == (
        "10.0.0.5",
        "remote:10.0.0.5",
    )
    assert dt_mod._infer_host_profile("http://example") == ("example", "remote:example")
    with patch("deployment_tuple.urlparse") as mock_up:
        mock_up.return_value = MagicMock(hostname=None)
        assert dt_mod._infer_host_profile("http://x")[0] == "unknown"

    assert dt_mod._infer_quant_from_strings("path/UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert dt_mod._infer_quant_from_strings("Qwen3.8-FN-NVFP4-DQ8") == "NVFP4-DQ8"
    assert dt_mod._infer_quant_from_strings("plain-name") == "Unknown"


def test_detect_engine_basalt_strata_llamacpp_and_unknown():
    basalt = dt_mod.detect_engine_from_endpoint(
        "http://192.168.0.31:8080",
        "strata-qwen",
        health_payload={
            "loaded": True,
            "max_context": 262144,
            "kv": "int8",
            "parallel": 2,
            "version": "0.4.1",
        },
        models_payload={
            "data": [
                {"id": "other"},
                {"id": "strata-qwen", "owned_by": "basalt"},
            ]
        },
        props_payload={},
        extra_meta={"launch_env": {"CUDA_VISIBLE_DEVICES": "0", "bad": 123}},
    )
    assert basalt.engine == "basalt"
    assert basalt.kv_quant == "int8"
    assert basalt.parallel_slots == 2
    assert basalt.max_context == 262144
    assert basalt.launch_env == {"CUDA_VISIBLE_DEVICES": "0"}
    d = dt_mod.deployment_tuple_to_dict(basalt)
    assert d["host"] == "gx10"

    llama = dt_mod.detect_engine_from_endpoint(
        "http://127.0.0.1:8083",
        "locallama-qwen",
        health_payload={},
        models_payload={
            "data": [
                {
                    "id": "other-model",
                    "owned_by": "llamacpp",
                    "status": {"args": ["--foo", "-ctk", "q8_0"]},
                    "meta": {"path": "/models/Qwen3.8-27B-UD-Q5_K_XL.gguf"},
                }
            ]
        },
        props_payload={
            "default_generation_settings": {"n_ctx": 240896},
            "total_slots": 2,
        },
    )
    assert llama.engine == "llama.cpp"
    assert llama.kv_quant == "q8_0"
    assert llama.weight_quant == "UD-Q5_K_XL"
    assert llama.max_context == 240896

    strata_fallback = dt_mod.detect_engine_from_endpoint(
        "http://10.0.0.2:8080",
        "inkling",
        health_payload={},
        models_payload={
            "data": [
                {
                    "id": "inkling",
                    "owned_by": "strata",
                    "meta": "non-dict",
                }
            ]
        },
        props_payload={},
    )
    assert strata_fallback.engine == "strata"
    assert strata_fallback.kv_quant == "Unknown"

    llama_status_nokv = dt_mod.detect_engine_from_endpoint(
        "http://10.0.0.2:8080",
        "inkling",
        health_payload={},
        models_payload={
            "data": [
                {
                    "id": "inkling",
                    "owned_by": "llamacpp",
                    "status": {"args": ["--no-kv"]},
                }
            ]
        },
        props_payload={"default_generation_settings": {"n_ctx": "bad"}},
    )
    assert llama_status_nokv.engine == "llama.cpp"
    assert llama_status_nokv.max_context is None

    strata_health = dt_mod.detect_engine_from_endpoint(
        "http://10.0.0.3:8080",
        "fn",
        health_payload={"loaded": True},
        models_payload={"data": ["not-a-dict"]},
        props_payload={},
    )
    assert strata_health.engine == "strata"

    explicit_override = dt_mod.detect_engine_from_endpoint(
        "http://10.0.0.5:8080",
        "m",
        engine_override="basalt",
        health_payload={},
        models_payload={"data": []},
        props_payload={},
        extra_meta={"weight_quant": "NVFP4-DQ8"},
    )
    assert explicit_override.engine == "basalt"
    assert explicit_override.weight_quant == "NVFP4-DQ8"

    with patch.dict(os.environ, {"STRATA_ROUTE_TAIL_SKIP": "0"}, clear=False):
        with patch("deployment_tuple._probe_json", return_value=None):
            unk = dt_mod.detect_engine_from_endpoint(
                "http://10.0.0.4:8080", "unknown-m"
            )
            assert unk.engine == "unknown"
            assert unk.launch_env["STRATA_ROUTE_TAIL_SKIP"] == "0"

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("STRATA_ROUTE_TAIL_SKIP", "CUDA_VISIBLE_DEVICES")
    }
    with patch.dict(os.environ, clean_env, clear=True):
        unk2 = dt_mod.detect_engine_from_endpoint(
            "http://10.0.0.6:8080",
            "m",
            health_payload={},
            models_payload={"data": []},
            props_payload={},
        )
        assert unk2.launch_env == {}
