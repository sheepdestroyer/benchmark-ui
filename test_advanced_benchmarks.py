import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from advanced_benchmarks import (
    _get_presets_config,
    call_endpoint,
    generate_filler_text,
    get_preset_metadata,
    is_safe_code,
    load_presets_config,
    map_repo_to_preset_alias,
    resolve_presets_path,
)


class TestIsSafeCode(unittest.TestCase):
    def test_valid_code(self):
        code = "print('Hello World')\nx = 1 + 1"
        safe, msg = is_safe_code(code)
        self.assertTrue(safe)
        self.assertIsNone(msg)

    def test_syntax_error(self):
        code = "print('Hello World'"
        safe, msg = is_safe_code(code)
        self.assertFalse(safe)
        self.assertIn("Syntax error", msg)

    def test_forbidden_code_patterns(self):
        cases = [
            ("import subprocess", "import subprocess", "Forbidden"),
            ("import subprocess.Popen", "import sub-module", "Forbidden"),
            ("from subprocess import Popen", "from forbidden module", "Forbidden"),
            ("from os import system", "from forbidden name", "Forbidden"),
            ("import os\nos.system('ls')", "forbidden attribute usage", "Forbidden"),
            ("eval('1 + 1')", "forbidden identifier usage", "Forbidden"),
        ]
        for code, label, expected_keyword in cases:
            with self.subTest(case=label):
                safe, msg = is_safe_code(code)
                self.assertFalse(safe)
                self.assertIsNotNone(msg)
                self.assertIn(expected_keyword, msg)

class TestGenerateFillerText(unittest.TestCase):
    def test_zero_target(self):
        res = generate_filler_text(0)
        self.assertEqual(res, [])

    def test_negative_target(self):
        res = generate_filler_text(-10)
        self.assertEqual(res, [])

    def test_positive_target_size(self):
        # target_tokens=10 => target_chars=45
        res = generate_filler_text(10)
        self.assertTrue(len(res) > 0)
        self.assertTrue(all(isinstance(p, str) for p in res))
        total_chars = sum(len(p) + 1 for p in res)
        self.assertGreaterEqual(total_chars, 45)

    @patch('advanced_benchmarks.random.choice')
    def test_paragraph_structure(self, mock_choice):
        mock_choice.return_value = "A sentence."
        # One paragraph of 5 sentences is "A sentence. A sentence. A sentence. A sentence. A sentence."
        # len = 5 * 11 + 4 = 59.
        # Target tokens = 10 => 45 chars.
        # This should generate exactly 1 paragraph.
        res = generate_filler_text(10)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], "A sentence. A sentence. A sentence. A sentence. A sentence.")


class TestPresetsConfigAndAlias(unittest.TestCase):
    def setUp(self):
        load_presets_config.cache_clear()
        map_repo_to_preset_alias.cache_clear()

    def tearDown(self):
        load_presets_config.cache_clear()
        map_repo_to_preset_alias.cache_clear()

    def test_load_presets_config_nonexistent_path(self):
        config = load_presets_config("/nonexistent/file/path/model_presets.ini")
        self.assertIsNone(config)

    def test_load_presets_config_default_nonexistent(self):
        with patch.dict(os.environ, {"PRESETS_FILE": "/nonexistent/default/model_presets.ini"}):
            load_presets_config.cache_clear()
            config = load_presets_config(None)
            self.assertIsNone(config)

    def test_load_presets_config_caching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[TestModel]\nalias = TestAlias\n", encoding="utf-8")

            c1 = load_presets_config(str(presets_file))
            c2 = load_presets_config(str(presets_file))
            self.assertIsNotNone(c1)
            self.assertIs(c1, c2)
            self.assertEqual(load_presets_config.cache_info().hits, 1)

    def test_load_presets_config_corrupt_ini(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "corrupt.ini"
            presets_file.write_text("corrupt header without closing bracket\nkey = val\n", encoding="utf-8")

            config = load_presets_config(str(presets_file))
            self.assertIsNone(config)

    def test_load_presets_config_alias_pointer(self):
        self.assertIs(_get_presets_config, load_presets_config)

    def test_map_repo_to_preset_alias_invalid_input(self):
        self.assertIsNone(map_repo_to_preset_alias(None))
        self.assertEqual(map_repo_to_preset_alias(""), "")
        self.assertEqual(map_repo_to_preset_alias(12345), 12345)

    def test_map_repo_to_preset_alias_nonexistent_file(self):
        res = map_repo_to_preset_alias("my-model", presets_file="/nonexistent/model_presets.ini")
        self.assertEqual(res, "my-model")

    def test_map_repo_to_preset_alias_matching_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[Qwen3.6-27B]\nthreads = 8\n", encoding="utf-8")

            # Exact match case-insensitive
            res1 = map_repo_to_preset_alias("qwen3.6-27b", presets_file=str(presets_file))
            self.assertEqual(res1, "Qwen3.6-27B")

            res2 = map_repo_to_preset_alias("QWEN3.6-27B", presets_file=str(presets_file))
            self.assertEqual(res2, "Qwen3.6-27B")

    def test_map_repo_to_preset_alias_fallback_hf_repo_and_alias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("""[*]
flash-attn = true

[Qwen3.6-27B-spec3]
hf-repo = unsloth/Qwen3.6-27B-spec3-GGUF
alias = qwen-spec-alias
""", encoding="utf-8")

            # hf-repo match
            res_repo = map_repo_to_preset_alias("unsloth/qwen3.6-27b-spec3-gguf", presets_file=str(presets_file))
            self.assertEqual(res_repo, "Qwen3.6-27B-spec3")

            # alias match
            res_alias = map_repo_to_preset_alias("QWEN-SPEC-ALIAS", presets_file=str(presets_file))
            self.assertEqual(res_alias, "Qwen3.6-27B-spec3")

    def test_map_repo_to_preset_alias_fallback_no_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[Qwen3.6-27B]\nthreads = 8\n", encoding="utf-8")

            res = map_repo_to_preset_alias("completely-unknown-repo", presets_file=str(presets_file))
            self.assertEqual(res, "completely-unknown-repo")

    def test_map_repo_to_preset_alias_caching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[CachedModel]\nalias = CM\n", encoding="utf-8")

            res1 = map_repo_to_preset_alias("CachedModel", presets_file=str(presets_file))
            res2 = map_repo_to_preset_alias("CachedModel", presets_file=str(presets_file))
            self.assertEqual(res1, "CachedModel")
            self.assertEqual(res2, "CachedModel")
            self.assertEqual(map_repo_to_preset_alias.cache_info().hits, 1)

    def test_map_repo_to_preset_alias_corrupt_ini_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "corrupt.ini"
            presets_file.write_text("[invalid ini", encoding="utf-8")

            res = map_repo_to_preset_alias("fallback-model", presets_file=str(presets_file))
            self.assertEqual(res, "fallback-model")

    def test_resolve_presets_path(self):
        # 1. Explicit path
        self.assertEqual(resolve_presets_path("/explicit/path.ini"), "/explicit/path.ini")
        self.assertEqual(resolve_presets_path(Path("/explicit/path2.ini")), "/explicit/path2.ini")

        # 2. PRESETS_FILE environment variable
        with patch.dict(os.environ, {"PRESETS_FILE": "/env/path.ini"}):
            self.assertEqual(resolve_presets_path(None), "/env/path.ini")

        # 3. Default fallback logic when PRESETS_FILE is not set
        primary = os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini"))
        fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "../llama.cpp/model_presets.ini"))

        # Primary exists
        with patch.dict(os.environ, {}, clear=True), patch("os.path.exists", side_effect=lambda p: str(p) == primary):
            self.assertEqual(resolve_presets_path(None), primary)

        # Primary does not exist, fallback exists
        with patch.dict(os.environ, {}, clear=True), patch("os.path.exists", side_effect=lambda p: str(p) == fallback):
            self.assertEqual(resolve_presets_path(None), fallback)

        # Neither exists -> returns primary
        with patch.dict(os.environ, {}, clear=True), patch("os.path.exists", return_value=False):
            self.assertEqual(resolve_presets_path(None), primary)

    def test_normalize_repo_id_and_repo_id_matches_helpers(self):
        from advanced_benchmarks import _normalize_repo_id, _repo_id_matches
        self.assertEqual(_normalize_repo_id(None), "")
        self.assertEqual(_normalize_repo_id(""), "")
        self.assertEqual(_normalize_repo_id(123), "")
        self.assertEqual(_normalize_repo_id("  unsloth/foo  "), "foo")

        self.assertFalse(_repo_id_matches(None, "bar"))
        self.assertFalse(_repo_id_matches("bar", None))
        self.assertFalse(_repo_id_matches(123, "bar"))
        self.assertFalse(_repo_id_matches("  ", "bar"))
        self.assertFalse(_repo_id_matches("bar", "   "))
        self.assertTrue(_repo_id_matches("unsloth/bar", "bar"))
        self.assertTrue(_repo_id_matches("bar", "unsloth/bar"))
        self.assertTrue(_repo_id_matches("bar", "bar"))
        self.assertFalse(_repo_id_matches("bar", "baz"))

    def test_map_repo_to_preset_alias_full_alias_and_exception(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("""[ProfileX]
alias = unsloth/alias1, alias2
""", encoding="utf-8")

            # Match full alias with unsloth prefix
            self.assertEqual(
                map_repo_to_preset_alias("unsloth/alias1, alias2", presets_file=str(presets_file)),
                "ProfileX"
            )

        mock_cfg = MagicMock()
        mock_cfg.sections.side_effect = RuntimeError("Mock error")
        with patch("advanced_benchmarks.load_presets_config", return_value=mock_cfg):
            self.assertEqual(map_repo_to_preset_alias("any-model"), "any-model")

    def test_load_presets_config_fallback_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fallback_ini = Path(tmp_dir) / "model_presets.ini"
            fallback_ini.write_text("[FallbackProfile]\nthreads = 12\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True), patch("advanced_benchmarks.resolve_presets_path", return_value=str(fallback_ini)):
                load_presets_config.cache_clear()
                cfg = load_presets_config(None)
                self.assertIsNotNone(cfg)
                self.assertIn("FallbackProfile", cfg.sections())

    def test_map_repo_to_preset_alias_unsloth_prefix_normalization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("""[*]
flash-attn = true

[Qwen3.8-27B-spec]
hf-repo = unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL
alias = unsloth/locallama-qwen, local-qwen

[PlainSection]
hf-repo = PlainRepo:latest
alias = plain-alias

[unsloth/PrefixSection]
hf-repo = PrefixRepo:latest
""", encoding="utf-8")

            # 1. Query has prefix, hf-repo has prefix
            self.assertEqual(
                map_repo_to_preset_alias("unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL", presets_file=str(presets_file)),
                "Qwen3.8-27B-spec"
            )
            # 2. Query has NO prefix, hf-repo HAS prefix
            self.assertEqual(
                map_repo_to_preset_alias("Qwen3.8-27B-GGUF:UD-Q5_K_XL", presets_file=str(presets_file)),
                "Qwen3.8-27B-spec"
            )
            # 3. Query has prefix, hf-repo has NO prefix
            self.assertEqual(
                map_repo_to_preset_alias("unsloth/PlainRepo:latest", presets_file=str(presets_file)),
                "PlainSection"
            )
            # 4. Query has NO prefix, section HAS prefix
            self.assertEqual(
                map_repo_to_preset_alias("PrefixSection", presets_file=str(presets_file)),
                "unsloth/PrefixSection"
            )
            # 5. Query has prefix, section has NO prefix
            self.assertEqual(
                map_repo_to_preset_alias("unsloth/PlainSection", presets_file=str(presets_file)),
                "PlainSection"
            )
            # 6. Alias normalization: query has no prefix, alias has prefix
            self.assertEqual(
                map_repo_to_preset_alias("locallama-qwen", presets_file=str(presets_file)),
                "Qwen3.8-27B-spec"
            )
            # 7. Alias normalization: query has prefix, alias has no prefix
            self.assertEqual(
                map_repo_to_preset_alias("unsloth/local-qwen", presets_file=str(presets_file)),
                "Qwen3.8-27B-spec"
            )
            # 8. Single alias with prefix
            self.assertEqual(
                map_repo_to_preset_alias("unsloth/plain-alias", presets_file=str(presets_file)),
                "PlainSection"
            )

    def test_get_preset_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("""[*]
flash-attn = true
parallel = 2

[MyProfile]
parallel = 4
n-gpu-layers = 33
custom-param = hello
""", encoding="utf-8")

            meta = get_preset_metadata("MyProfile", presets_file=str(presets_file))
            self.assertEqual(meta["flash_attn"], "true")
            self.assertEqual(meta["parallel"], "4")
            self.assertEqual(meta["n_gpu_layers"], "33")
            self.assertEqual(meta["custom_param"], "hello")

            # Nonexistent section still gets globals and defaults
            meta_default = get_preset_metadata("UnknownProfile", presets_file=str(presets_file))
            self.assertEqual(meta_default["flash_attn"], "true")
            self.assertEqual(meta_default["parallel"], "2")
            self.assertEqual(meta_default["n_gpu_layers"], "99")

    def test_get_preset_metadata_nonexistent_file(self):
        meta = get_preset_metadata("AnyProfile", presets_file="/nonexistent/model_presets.ini")
        self.assertEqual(meta["parallel"], "1")
        self.assertEqual(meta["n_gpu_layers"], "99")
        self.assertEqual(meta["spec_type"], "None")

class TestASTSafetyCheck(unittest.TestCase):
    def test_safe_code_snippets(self):
        safe_snippets = [
            # Arithmetic & math
            ("1 + 2 * 3 - 4 / 2 ** 2", "arithmetic precedence"),
            ("x = 10 % 3\ny = 2 ** 5\nz = x + y", "basic math operators"),
            ("import math\nval = math.sqrt(16) + math.sin(math.pi / 2)", "math module functions"),
            # String formatting & manipulation
            ("name = 'world'\nmsg = f'hello {name}'\ns = msg.upper().strip()", "f-strings and string methods"),
            ("formatted = 'Value: {:.2f}'.format(3.14159)", "str.format"),
            ("parts = 'a,b,c'.split(',')\njoined = '-'.join(parts)", "split and join"),
            # Loops & control flow
            ("""
def sum_evens(n):
    total = 0
    for i in range(n):
        if i % 2 == 0:
            total += i
        else:
            continue
    return total
""", "for loop with if/else/continue"),
            ("""
count = 0
while count < 5:
    count += 1
""", "while loop"),
            ("""
try:
    x = 10 / 2
except ZeroDivisionError:
    x = 0
finally:
    done = True
""", "try-except-finally block"),
            # Dict, list, set, tuple manipulation
            ("d = {'a': 1, 'b': 2}\nd['c'] = 3\nval = d.get('a')", "dict manipulation"),
            ("items = [1, 2, 3]\nitems.append(4)\npopped = items.pop()", "list methods"),
            ("squares = [x**2 for x in range(10) if x % 2 == 0]", "list comprehension"),
            ("unique = {x % 3 for x in range(10)}", "set comprehension"),
            ("lookup = {k: v for k, v in [('a', 1), ('b', 2)]}", "dict comprehension"),
            # Algorithms & recursion
            ("""
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
""", "recursive fibonacci"),
            ("""
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""", "binary search algorithm"),
            # Safe imports
            ("""
import json
data = json.loads('{"a": 1}')
""", "json import"),
            ("""
import re
match = re.match(r'\\d+', '123')
""", "re import"),
            ("import collections\nq = collections.deque()", "collections import"),
            ("import itertools\ncomb = list(itertools.combinations([1, 2], 2))", "itertools import"),
            ("import random\nr = random.randint(1, 10)", "random import"),
            ("import time\nt = time.time()", "time import"),
            ("import datetime\nnow = datetime.datetime.now()", "datetime import"),
            # Benign patterns requested by reviewer
            ("import re\npattern = re.compile(r'\\d+')", "re.compile allowed"),
            ("""
class Base:
    def __init__(self):
        self.x = 1
class Sub(Base):
    def __init__(self):
        super().__init__()
""", "super().__init__() allowed"),
            ("""
class Encapsulated:
    def __init__(self):
        self.__secret = 42
    def get_secret(self):
        return self.__secret
""", "private attributes self.__var allowed"),
            ("from __future__ import annotations", "from __future__ import annotations allowed"),
            ("""
if __name__ == '__main__':
    msg = __doc__
""", "if __name__ == '__main__' and __doc__ allowed"),
            ("""
class Container:
    def __len__(self):
        return 0
    def __str__(self):
        return 'container'
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
c = Container()
s = str(c)
l = len(c)
""", "safe dunder methods allowed"),
            # Single underscore variables
            ("_private = 42\nx = _private + 1", "single underscore identifiers"),
        ]

        for code, label in safe_snippets:
            with self.subTest(case=label):
                safe, msg = is_safe_code(code)
                self.assertTrue(safe, f"Expected safe for '{label}', got error: {msg}")
                self.assertIsNone(msg)

    def test_exploit_bypasses(self):
        exploit_snippets = [
            # Prompt-specified exploits
            ("__import__('os').system('ls')", "import dunder call"),
            ("__builtins__['eval']('1+1')", "builtins subscript exploit"),
            ("().__class__.__bases__[0].__subclasses__()", "dunder subclass traversal"),
            ("open('/etc/passwd')", "builtin open call"),
            ("eval('1+1')", "builtin eval call"),
            ("exec('1+1')", "builtin exec call"),
            ("import sys", "forbidden sys import"),
            ("import subprocess", "forbidden subprocess import"),
            ("import ctypes", "forbidden ctypes import"),
            ("import posix", "forbidden posix import"),
            ("import shutil", "forbidden shutil import"),
            # Other dangerous modules
            ("import os", "forbidden os import"),
            ("import socket", "forbidden socket import"),
            ("import pty", "forbidden pty import"),
            ("import builtins", "forbidden builtins import"),
            ("import _frozen_importlib", "forbidden _frozen_importlib import"),
            ("import importlib", "forbidden importlib import"),
            ("import inspect", "forbidden inspect import"),
            ("import pickle", "forbidden pickle import"),
            ("import shelve", "forbidden shelve import"),
            ("import multiprocessing", "forbidden multiprocessing import"),
            ("import threading", "forbidden threading import"),
            ("import signal", "forbidden signal import"),
            # Reviewer requested bypass modules
            ("import asyncio", "forbidden asyncio import"),
            ("import pathlib", "forbidden pathlib import"),
            ("import io", "forbidden io import"),
            ("import runpy", "forbidden runpy import"),
            ("import operator", "forbidden operator import"),
            ("import urllib", "forbidden urllib import"),
            ("import http", "forbidden http import"),
            ("import webbrowser", "forbidden webbrowser import"),
            ("import tempfile", "forbidden tempfile import"),
            ("import pdb", "forbidden pdb import"),
            ("import code", "forbidden code import"),
            ("operator.attrgetter('x')", "operator attribute access"),
            ("pathlib.Path('/etc/passwd')", "pathlib attribute access"),
            # Reviewer requested bypass builtins
            ("globals()", "forbidden globals builtin"),
            ("locals()", "forbidden locals builtin"),
            ("vars()", "forbidden vars builtin"),
            ("dir()", "forbidden dir builtin"),
            ("exit()", "forbidden exit builtin"),
            ("quit()", "forbidden quit builtin"),
            ("help()", "forbidden help builtin"),
            # ImportFrom dangerous modules
            ("from os import path", "from os import"),
            ("from sys import argv", "from sys import"),
            ("from subprocess import Popen", "from subprocess import"),
            ("from shutil import rmtree", "from shutil import"),
            ("from ctypes import CDLL", "from ctypes import"),
            ("from builtins import getattr", "from builtins import"),
            ("from importlib import import_module", "from importlib import"),
            ("from pathlib import Path", "from pathlib import"),
            ("from io import StringIO", "from io import"),
            # Dangerous builtins / calls
            ("compile('1+1', '', 'eval')", "builtin compile call"),
            ("getattr(math, 'sin')", "builtin getattr call"),
            ("setattr(obj, 'x', 1)", "builtin setattr call"),
            ("delattr(obj, 'x')", "builtin delattr call"),
            ("input('Enter:')", "builtin input call"),
            ("breakpoint()", "builtin breakpoint call"),
            ("f = open", "builtin open assigned as identifier"),
            ("e = eval", "builtin eval assigned as identifier"),
            ("g = globals", "builtin globals assigned as identifier"),
            ("obj.open()", "attribute call on open"),
            ("obj.eval('1+1')", "attribute call on eval"),
            ("obj.exec('1+1')", "attribute call on exec"),
            ("something.compile('1+1')", "non-re compile call"),
            # Dangerous attribute roots
            ("os.system('ls')", "os.system attribute access"),
            ("os.environ", "os.environ attribute access"),
            ("os.path.join('a', 'b')", "os.path attribute access"),
            ("sys.modules", "sys.modules attribute access"),
            ("sys.exit(0)", "sys.exit attribute access"),
            ("subprocess.run(['ls'])", "subprocess.run attribute access"),
            ("shutil.rmtree('/tmp')", "shutil.rmtree attribute access"),
            ("ctypes.c_char_p(b'test')", "ctypes attribute access"),
            # Reviewer requested danger dunder string constants
            ("x = '__builtins__'", "danger dunder constant __builtins__"),
            ("x = '__globals__'", "danger dunder constant __globals__"),
            ("x = '__subclasses__'", "danger dunder constant __subclasses__"),
            ("x = '__code__'", "danger dunder constant __code__"),
            ("x = '__import__'", "danger dunder constant __import__"),
            ("d = {'key': '__builtins__'}", "danger dunder constant in dict key"),
            # Dangerous dunder attributes
            ("x.__globals__", "dunder attribute __globals__"),
            ("x.__code__", "dunder attribute __code__"),
            ("x.__dict__", "dunder attribute __dict__"),
            ("x.__class__", "dunder attribute __class__"),
            ("x.__subclasses__()", "dunder attribute __subclasses__"),
            ("x.__bases__", "dunder attribute __bases__"),
            ("from math import __name__", "dunder import alias"),
            ("import math as __math", "dunder asname import"),
        ]

        for code, label in exploit_snippets:
            with self.subTest(case=label):
                safe, msg = is_safe_code(code)
                self.assertFalse(safe, f"Expected exploit to be blocked for '{label}' ({code})")
                self.assertIsNotNone(msg)
                self.assertTrue(
                    any(kw in msg for kw in ("Forbidden", "Syntax error", "Invalid")),
                    f"Expected security violation keyword in msg for '{label}', got: {msg}"
                )

    def test_syntax_error_handling(self):
        syntax_errors = [
            ("print('unclosed string)", "unclosed string literal"),
            ("def broken(:", "invalid def syntax"),
            ("for i in", "incomplete for loop"),
            ("((((", "unmatched parentheses"),
            ("import", "incomplete import statement"),
        ]
        for code, label in syntax_errors:
            with self.subTest(case=label):
                safe, msg = is_safe_code(code)
                self.assertFalse(safe)
                self.assertIn("Syntax error", msg)

    def test_invalid_input_types(self):
        for bad_input in [None, 123, [1, 2, 3], {"code": "x = 1"}]:
            with self.subTest(bad_input=type(bad_input).__name__):
                safe, msg = is_safe_code(bad_input)
                self.assertFalse(safe)
                self.assertIn("Invalid code input", msg)


class TestCallEndpoint(unittest.TestCase):
    def _create_mock_response(self, status_code=200, lines=None, text=""):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = text
        mock_resp.iter_lines.return_value = lines if lines is not None else []
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        return mock_resp

    @patch("advanced_benchmarks.requests.post")
    def test_call_endpoint_normal_sse_streaming(self, mock_post):
        """Normal SSE streaming response with delta content, delta reasoning_content, usage metrics, and [DONE] token."""
        lines = [
            b"",  # Empty line skipped
            b"event: ping",  # Non-data line skipped
            b'data: {"choices": [{"delta": {"role": "assistant"}}]}',
            b'data: {"choices": [{"delta": {"reasoning_content": "Thinking step 1. "}}]}',
            b'data: {"choices": [{"delta": {"reasoning_content": "Thinking step 2."}}]}',
            b'data: {"choices": [{"delta": {"content": "Hello, "}}]}',
            b'data: {"choices": [{"delta": {"content": "world!"}}]}',
            b'data: {"usage": {"prompt_tokens": 42, "completion_tokens": 18}}',
            b"data: [DONE]",
            b'data: {"choices": [{"delta": {"content": "Should be ignored after [DONE]"}}]}',
        ]
        mock_resp = self._create_mock_response(status_code=200, lines=lines)
        mock_post.return_value = mock_resp

        res = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt", max_tokens=256)

        self.assertIsNotNone(res)
        self.assertEqual(res["response"], "Hello, world!")
        self.assertEqual(res["reasoning"], "Thinking step 1. Thinking step 2.")
        self.assertEqual(res["prompt_tokens"], 42)
        self.assertEqual(res["completion_tokens"], 18)
        self.assertGreater(res["ttft"], 0.0)
        self.assertGreaterEqual(res["decode_time"], 0.0)
        self.assertGreater(res["total_time"], 0.0)
        self.assertGreater(res["prefill_speed"], 0.0)
        self.assertGreaterEqual(res["decode_speed"], 0.0)

        # Verify requests.post payload and parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(kwargs["headers"], {"Content-Type": "application/json"})
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["timeout"], 300)
        self.assertEqual(
            kwargs["json"],
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test prompt"}],
                "max_tokens": 256,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    @patch("advanced_benchmarks.time.time")
    @patch("advanced_benchmarks.requests.post")
    def test_first_token_timing_detection_content_first(self, mock_post, mock_time):
        """Early first token timing detection when content arrives first."""
        # Timeline:
        # Call 1 (start_time): 10.0
        # Call 2 (first_token_time): 12.0 (when content arrives)
        # Call 3 (end_time): 15.0
        mock_time.side_effect = [10.0, 12.0, 15.0]

        lines = [
            b'data: {"choices": [{"delta": {"role": "assistant"}}]}',
            b'data: {"choices": [{"delta": {"content": "First "}}]}',
            b'data: {"choices": [{"delta": {"content": "second."}}]}',
            b"data: [DONE]",
        ]
        mock_resp = self._create_mock_response(status_code=200, lines=lines)
        mock_post.return_value = mock_resp

        res = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")

        self.assertIsNotNone(res)
        self.assertEqual(res["response"], "First second.")
        self.assertEqual(res["reasoning"], "")
        self.assertAlmostEqual(res["ttft"], 2.0)  # 12.0 - 10.0
        self.assertAlmostEqual(res["decode_time"], 3.0)  # 15.0 - 12.0
        self.assertAlmostEqual(res["total_time"], 5.0)  # 15.0 - 10.0

    @patch("advanced_benchmarks.time.time")
    @patch("advanced_benchmarks.requests.post")
    def test_first_token_timing_detection_reasoning_first(self, mock_post, mock_time):
        """Early first token timing detection when reasoning_content arrives first."""
        # Timeline:
        # Call 1 (start_time): 20.0
        # Call 2 (first_token_time): 21.5 (when reasoning_content arrives)
        # Call 3 (end_time): 25.0
        mock_time.side_effect = [20.0, 21.5, 25.0]

        lines = [
            b'data: {"choices": [{"delta": {"role": "assistant"}}]}',
            b'data: {"choices": [{"delta": {"reasoning_content": "Thinking..."}}]}',
            b'data: {"choices": [{"delta": {"content": "Answer"}}]}',
            b"data: [DONE]",
        ]
        mock_resp = self._create_mock_response(status_code=200, lines=lines)
        mock_post.return_value = mock_resp

        res = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")

        self.assertIsNotNone(res)
        self.assertEqual(res["response"], "Answer")
        self.assertEqual(res["reasoning"], "Thinking...")
        self.assertAlmostEqual(res["ttft"], 1.5)  # 21.5 - 20.0
        self.assertAlmostEqual(res["decode_time"], 3.5)  # 25.0 - 21.5
        self.assertAlmostEqual(res["total_time"], 5.0)  # 25.0 - 20.0

    @patch("advanced_benchmarks.time.time")
    @patch("advanced_benchmarks.requests.post")
    def test_first_token_timing_detection_no_tokens(self, mock_post, mock_time):
        """When no content or reasoning tokens arrive, first_token_time defaults to end_time."""
        mock_time.side_effect = [30.0, 32.0]

        lines = [
            b"data: [DONE]",
        ]
        mock_resp = self._create_mock_response(status_code=200, lines=lines)
        mock_post.return_value = mock_resp

        res = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")

        self.assertIsNotNone(res)
        self.assertEqual(res["response"], "")
        self.assertEqual(res["reasoning"], "")
        self.assertAlmostEqual(res["ttft"], 2.0)  # end_time - start_time
        self.assertAlmostEqual(res["decode_time"], 0.0)  # end_time - first_token_time (which is end_time)
        self.assertAlmostEqual(res["total_time"], 2.0)

    @patch("advanced_benchmarks.requests.post")
    def test_call_endpoint_http_non_200(self, mock_post):
        """Non-200 HTTP status (e.g. 500, 404) returns None."""
        for code in [400, 404, 500, 503]:
            with self.subTest(status_code=code):
                mock_resp = self._create_mock_response(status_code=code, text="HTTP error")
                mock_post.return_value = mock_resp
                res = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")
                self.assertIsNone(res)

    @patch("advanced_benchmarks.requests.post")
    def test_call_endpoint_network_exception(self, mock_post):
        """Network/connection exception during requests.post returns None."""
        exceptions = [
            requests.exceptions.ConnectionError("Connection refused"),
            requests.exceptions.Timeout("Read timed out"),
            requests.exceptions.RequestException("Request failure"),
            RuntimeError("Unexpected socket error"),
        ]
        for exc in exceptions:
            with self.subTest(exc=type(exc).__name__):
                mock_post.side_effect = exc
                res = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")
                self.assertIsNone(res)

    @patch("advanced_benchmarks.requests.post")
    def test_call_endpoint_malformed_sse_json(self, mock_post):
        """Malformed SSE JSON chunk is gracefully skipped without interrupting stream."""
        lines = [
            b"data: {invalid-json-chunk",
            b"data: not-json",
            b'data: {"choices": [{"delta": {"content": "Recovered content"}}]}',
            b"data: {broken: json",
            b"data: [DONE]",
        ]
        mock_resp = self._create_mock_response(status_code=200, lines=lines)
        mock_post.return_value = mock_resp

        res = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")

        self.assertIsNotNone(res)
        self.assertEqual(res["response"], "Recovered content")

    @patch("advanced_benchmarks.requests.post")
    def test_call_endpoint_unexpected_chunk_exception(self, mock_post):
        """Unexpected exception during chunk processing is handled gracefully without crashing."""
        lines = [
            # choices is not a subscriptable list -> TypeError
            b'data: {"choices": 12345}',
            # choices[0] is None -> AttributeError on .get()
            b'data: {"choices": [null]}',
            # delta content is non-string (int) -> TypeError
            b'data: {"choices": [{"delta": {"content": 123}}]}',
            # delta reasoning_content is non-string (list) -> TypeError
            b'data: {"choices": [{"delta": {"reasoning_content": [1, 2, 3]}}]}',
            # Valid chunk afterwards
            b'data: {"choices": [{"delta": {"content": "Working content"}}]}',
            b"data: [DONE]",
        ]
        mock_resp = self._create_mock_response(status_code=200, lines=lines)
        mock_post.return_value = mock_resp

        res = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")

        self.assertIsNotNone(res)
        self.assertEqual(res["response"], "Working content")

    @patch("advanced_benchmarks.time.time")
    @patch("advanced_benchmarks.requests.post")
    def test_speed_calculations_and_zero_division_guard(self, mock_post, mock_time):
        """Verify speed calculations for positive times and division by zero guard when times are 0."""
        # Case 1: Positive ttft and decode_time
        mock_time.side_effect = [100.0, 102.0, 107.0]  # ttft = 2.0, decode = 5.0
        lines = [
            b'data: {"choices": [{"delta": {"content": "token"}}]}',
            b'data: {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}',
            b"data: [DONE]",
        ]
        mock_resp = self._create_mock_response(status_code=200, lines=lines)
        mock_post.return_value = mock_resp

        res = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["ttft"], 2.0)
        self.assertAlmostEqual(res["decode_time"], 5.0)
        self.assertAlmostEqual(res["prefill_speed"], 50.0)  # 100 / 2.0
        self.assertAlmostEqual(res["decode_speed"], 10.0)  # 50 / 5.0

        # Case 2: Zero ttft and zero decode_time (start == first_token == end)
        mock_time.side_effect = [50.0, 50.0, 50.0]
        lines = [
            b'data: {"choices": [{"delta": {"content": "instant"}}]}',
            b'data: {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}',
            b"data: [DONE]",
        ]
        mock_resp = self._create_mock_response(status_code=200, lines=lines)
        mock_post.return_value = mock_resp

        res_zero = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")
        self.assertIsNotNone(res_zero)
        self.assertEqual(res_zero["ttft"], 0.0)
        self.assertEqual(res_zero["decode_time"], 0.0)
        self.assertEqual(res_zero["prefill_speed"], 0)
        self.assertEqual(res_zero["decode_speed"], 0)

        # Case 3: Missing usage dictionary
        mock_time.side_effect = [10.0, 12.0, 14.0]
        lines = [
            b'data: {"choices": [{"delta": {"content": "no-usage"}}]}',
            b"data: [DONE]",
        ]
        mock_resp = self._create_mock_response(status_code=200, lines=lines)
        mock_post.return_value = mock_resp

        res_no_usage = call_endpoint("http://127.0.0.1:8080", "test-model", "test prompt")
        self.assertIsNotNone(res_no_usage)
        self.assertEqual(res_no_usage["prompt_tokens"], 0)
        self.assertEqual(res_no_usage["completion_tokens"], 0)
        self.assertEqual(res_no_usage["prefill_speed"], 0)
        self.assertEqual(res_no_usage["decode_speed"], 0)


if __name__ == "__main__":
    unittest.main()
