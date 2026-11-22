"""Unit tests for bench_ocr.py."""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import bench_ocr as bo


def test_png_fence_runaway_and_validation():
    png = bo.minimal_png_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")

    assert bo.strip_code_fence("plain") == "plain"
    assert bo.strip_code_fence('```json\n{"a":1}\n```').strip() == '{"a":1}'
    assert bo.strip_code_fence("```json\npartial") == "partial"

    assert bo.runaway_problem("normal text") is None
    assert bo.runaway_problem("a" + ("\n" * 25) + "b") == "runaway_blank_lines"
    assert bo.runaway_problem(("abcd1234" * 200)) == "runaway_repetition"
    assert bo.runaway_problem("XXXX" * 300) is None

    with pytest.raises(ValueError, match="transcript"):
        bo.validate_vision_payload("bad")
    with pytest.raises(ValueError, match="completeness"):
        bo.validate_vision_payload({"text": "ok", "complete": "yes"})
    with pytest.raises(ValueError, match="visual_description"):
        bo.validate_vision_payload(
            {
                "text": "ok",
                "complete": True,
                "blank_verified": False,
                "visual_description": 1,
            }
        )
    with pytest.raises(ValueError, match="uncertainties"):
        bo.validate_vision_payload(
            {
                "text": "ok",
                "complete": True,
                "blank_verified": False,
                "visual_description": "doc",
                "uncertainties": [123],
            }
        )
    valid = {
        "text": "ok",
        "complete": True,
        "blank_verified": False,
        "visual_description": "doc",
        "uncertainties": [],
    }
    assert bo.validate_vision_payload(valid) == valid


def test_normalization_edit_distance_and_scoring():
    ref = "\u27e6h:Bonjour\u27e7  l\u2019ami [illegible]\n\nSuite"
    assert bo.normalize_ocr_ref(ref) == "Bonjour l'ami\nSuite"
    assert bo.normalize_ocr_hyp("Line  1\n\n") == "Line 1"

    assert bo.levenshtein_distance("abc", "abc") == 0
    assert bo.levenshtein_distance("abc", "") == 3
    assert bo.levenshtein_distance("kitten", "sitting") == 3
    # Exercise > 4096 char DP fallback branch
    long_a = "x" * 4100 + "abc"
    long_b = "x" * 4100 + "adc"
    assert bo.levenshtein_distance(long_a, long_b) == 1

    assert bo._word_error_rate([], []) == 0.0
    assert bo._word_error_rate(["one", "two"], ["one", "three"]) == 0.5

    metrics = bo.score_ocr_page(
        "Premiere ligne longue\nDeuxieme ligne longue",
        "Premiere ligne longue\nLigne inventee ici",
    )
    assert metrics["omitted"] == 1
    assert metrics["invented"] == 1
    assert metrics["cer"] > 0


def test_load_ocr_gold_cases():
    syn = bo.load_ocr_gold_cases(None)
    assert len(syn) == 1

    with tempfile.TemporaryDirectory() as tmp:
        gdir = Path(tmp)
        png_bytes = bo.minimal_png_bytes()
        png_sha = hashlib.sha256(png_bytes).hexdigest()
        png_file = gdir / "p1.png"
        png_file.write_bytes(png_bytes)

        manifest = {
            "pages": [
                {"page_id": "p2", "png_path": str(png_file), "png_sha256": png_sha}
            ]
        }
        gold_list = {
            "agreed": [
                {
                    "page_id": "p1",
                    "stratum": "s1",
                    "png_path": str(png_file),
                    "png_sha256": png_sha,
                    "text": "hello",
                },
                {"page_id": "p2", "stratum": "s1", "text": "world"},
                {
                    "page_id": "p3",
                    "stratum": "s2",
                    "png_path": str(png_file),
                    "png_sha256": png_sha,
                    "text": "!",
                },
            ]
        }
        (gdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (gdir / "gold.json").write_text(json.dumps(gold_list), encoding="utf-8")

        q = bo.load_ocr_gold_cases(gdir, profile="quick", limit=2)
        assert len(q) == 2
        q_auto = bo.load_ocr_gold_cases(gdir, profile="quick")
        assert len(q_auto) == 3

        gold_dict = {
            "agreed": {
                "p1": {
                    "page_id": "p1",
                    "stratum": "s1",
                    "png_path": str(png_file),
                    "png_sha256": "deadbeef",
                    "text": "x",
                }
            }
        }
        (gdir / "gold.json").write_text(json.dumps(gold_dict), encoding="utf-8")
        with pytest.raises(ValueError, match="checksum"):
            bo.load_ocr_gold_cases(gdir, profile="full")


def test_run_ocr_suite_paths():
    ok_payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '```json\n{"text":"Article 1. Synthetic OCR smoke check.","visual_description":"clean","continues_beyond_edge":false,"edge_notes":[],"uncertainties":[],"complete":true,"blank_verified":false}\n```'
                },
            }
        ]
    }
    resp_ok = MagicMock(raise_for_status=MagicMock())
    resp_ok.json.return_value = ok_payload
    with patch("bench_ocr.requests.post", return_value=resp_ok):
        res_ok = bo.run_ocr_suite("http://127.0.0.1:8083", "m")
    assert res_ok["cer_mean"] == 0.0 and res_ok["errors"] == 0

    runaway_text = "abc123xy" * 160
    runaway_payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "text": runaway_text,
                            "visual_description": "loop",
                            "uncertainties": [],
                            "complete": True,
                            "blank_verified": False,
                        }
                    )
                },
            }
        ]
    }
    resp_loop = MagicMock(raise_for_status=MagicMock())
    resp_loop.json.return_value = runaway_payload
    with patch("bench_ocr.requests.post", return_value=resp_loop):
        res_loop = bo.run_ocr_suite("http://127.0.0.1:8083", "m")
    assert res_loop["runaway_count"] == 1

    trunc_payload = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "line\\n" + "\\n" * 25},
            }
        ]
    }
    resp_trunc = MagicMock(raise_for_status=MagicMock())
    resp_trunc.json.return_value = trunc_payload
    with patch("bench_ocr.requests.post", return_value=resp_trunc):
        res_trunc = bo.run_ocr_suite("http://127.0.0.1:8083", "m")
    assert res_trunc["runaway_count"] == 1 and res_trunc["cer_mean"] is None

    trunc_noloop = {
        "choices": [{"finish_reason": "length", "message": {"content": "short"}}]
    }
    resp_trunc_noloop = MagicMock(raise_for_status=MagicMock())
    resp_trunc_noloop.json.return_value = trunc_noloop
    with patch("bench_ocr.requests.post", return_value=resp_trunc_noloop):
        res_nl = bo.run_ocr_suite("http://127.0.0.1:8083", "m")
    assert res_nl["runaway_count"] == 0 and res_nl["errors"] == 1

    bad_json_resp = MagicMock(raise_for_status=MagicMock())
    bad_json_resp.json.return_value = {
        "choices": [{"finish_reason": "stop", "message": {"content": "{not json"}}]
    }
    with patch("bench_ocr.requests.post", return_value=bad_json_resp):
        res_bad = bo.run_ocr_suite("http://127.0.0.1:8083", "m")
    assert res_bad["invalid_json_count"] == 1

    with patch(
        "bench_ocr.requests.post", side_effect=requests.RequestException("down")
    ):
        res_req_err = bo.run_ocr_suite("http://127.0.0.1:8083", "m")
    assert res_req_err["errors"] == 1 and res_req_err["pages_scored"] == 0
