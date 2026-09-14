import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

import advanced_benchmarks
from advanced_benchmarks import (
    _extract_code_block_from_response,
    _get_presets_config,
    _parse_endpoint_model_args,
    _parse_endpoint_preset_block,
    _print_benchmark_summary,
    _save_run_data,
    call_endpoint,
    generate_filler_text,
    get_model_settings_from_endpoint,
    get_preset_metadata,
    is_safe_code,
    load_presets_config,
    main,
    map_repo_to_preset_alias,
    resolve_presets_path,
    run_swe_test,
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

    @patch("advanced_benchmarks.random.choice")
    def test_paragraph_structure(self, mock_choice):
        mock_choice.return_value = "A sentence."
        # One paragraph of 5 sentences is "A sentence. A sentence. A sentence. A sentence. A sentence."
        # len = 5 * 11 + 4 = 59.
        # Target tokens = 10 => 45 chars.
        # This should generate exactly 1 paragraph.
        res = generate_filler_text(10)
        self.assertEqual(len(res), 1)
        self.assertEqual(
            res[0], "A sentence. A sentence. A sentence. A sentence. A sentence."
        )

    def test_invalid_types(self):
        invalid_types = [None, "invalid", [100], {"tokens": 100}, True, False]
        for val in invalid_types:
            with self.subTest(val=val), self.assertRaises(TypeError) as ctx:
                generate_filler_text(val)
            self.assertEqual(
                str(ctx.exception), "target_tokens must be an integer or float"
            )

    def test_float_targets(self):
        self.assertEqual(generate_filler_text(0.0), [])
        self.assertEqual(generate_filler_text(-5.5), [])
        res = generate_filler_text(10.5)
        self.assertTrue(len(res) > 0)


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
        with patch.dict(
            os.environ, {"PRESETS_FILE": "/nonexistent/default/model_presets.ini"}
        ):
            load_presets_config.cache_clear()
            config = load_presets_config(None)
            self.assertIsNone(config)

    def test_load_presets_config_caching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text(
                "[TestModel]\nalias = TestAlias\n", encoding="utf-8"
            )

            c1 = load_presets_config(str(presets_file))
            c2 = load_presets_config(str(presets_file))
            self.assertIsNotNone(c1)
            self.assertIs(c1, c2)
            self.assertEqual(load_presets_config.cache_info().hits, 1)

    def test_load_presets_config_corrupt_ini(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "corrupt.ini"
            presets_file.write_text(
                "corrupt header without closing bracket\nkey = val\n", encoding="utf-8"
            )

            config = load_presets_config(str(presets_file))
            self.assertIsNone(config)

    def test_load_presets_config_alias_pointer(self):
        self.assertIs(_get_presets_config, load_presets_config)

    def test_map_repo_to_preset_alias_invalid_input(self):
        self.assertIsNone(map_repo_to_preset_alias(None))
        self.assertEqual(map_repo_to_preset_alias(""), "")
        self.assertEqual(map_repo_to_preset_alias(12345), 12345)

    def test_map_repo_to_preset_alias_nonexistent_file(self):
        res = map_repo_to_preset_alias(
            "my-model", presets_file="/nonexistent/model_presets.ini"
        )
        self.assertEqual(res, "my-model")

    def test_map_repo_to_preset_alias_matching_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[Qwen3.6-27B]\nthreads = 8\n", encoding="utf-8")

            # Exact match case-insensitive
            res1 = map_repo_to_preset_alias(
                "qwen3.6-27b", presets_file=str(presets_file)
            )
            self.assertEqual(res1, "Qwen3.6-27B")

            res2 = map_repo_to_preset_alias(
                "QWEN3.6-27B", presets_file=str(presets_file)
            )
            self.assertEqual(res2, "Qwen3.6-27B")

    def test_map_repo_to_preset_alias_fallback_hf_repo_and_alias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text(
                """[*]
flash-attn = true

[Qwen3.6-27B-spec3]
hf-repo = unsloth/Qwen3.6-27B-spec3-GGUF
alias = qwen-spec-alias
""",
                encoding="utf-8",
            )

            # hf-repo match
            res_repo = map_repo_to_preset_alias(
                "unsloth/qwen3.6-27b-spec3-gguf", presets_file=str(presets_file)
            )
            self.assertEqual(res_repo, "Qwen3.6-27B-spec3")

            # alias match
            res_alias = map_repo_to_preset_alias(
                "QWEN-SPEC-ALIAS", presets_file=str(presets_file)
            )
            self.assertEqual(res_alias, "Qwen3.6-27B-spec3")

    def test_map_repo_to_preset_alias_fallback_no_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[Qwen3.6-27B]\nthreads = 8\n", encoding="utf-8")

            res = map_repo_to_preset_alias(
                "completely-unknown-repo", presets_file=str(presets_file)
            )
            self.assertEqual(res, "completely-unknown-repo")

    def test_map_repo_to_preset_alias_caching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text("[CachedModel]\nalias = CM\n", encoding="utf-8")

            res1 = map_repo_to_preset_alias(
                "CachedModel", presets_file=str(presets_file)
            )
            res2 = map_repo_to_preset_alias(
                "CachedModel", presets_file=str(presets_file)
            )
            self.assertEqual(res1, "CachedModel")
            self.assertEqual(res2, "CachedModel")
            self.assertEqual(map_repo_to_preset_alias.cache_info().hits, 1)

    def test_map_repo_to_preset_alias_corrupt_ini_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "corrupt.ini"
            presets_file.write_text("[invalid ini", encoding="utf-8")

            res = map_repo_to_preset_alias(
                "fallback-model", presets_file=str(presets_file)
            )
            self.assertEqual(res, "fallback-model")

    def test_resolve_presets_path(self):
        # 1. Explicit path
        self.assertEqual(
            resolve_presets_path("/explicit/path.ini"), "/explicit/path.ini"
        )
        self.assertEqual(
            resolve_presets_path(Path("/explicit/path2.ini")), "/explicit/path2.ini"
        )

        # 2. PRESETS_FILE environment variable
        with patch.dict(os.environ, {"PRESETS_FILE": "/env/path.ini"}):
            self.assertEqual(resolve_presets_path(None), "/env/path.ini")

        # 3. Default fallback logic when PRESETS_FILE is not set
        primary = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../llama.cpp/profiles/model_presets.ini"
            )
        )
        fallback = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../llama.cpp/model_presets.ini")
        )

        # Primary exists
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", side_effect=lambda p: str(p) == primary),
        ):
            self.assertEqual(resolve_presets_path(None), primary)

        # Primary does not exist, fallback exists
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", side_effect=lambda p: str(p) == fallback),
        ):
            self.assertEqual(resolve_presets_path(None), fallback)

        # Neither exists -> returns primary
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", return_value=False),
        ):
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
            presets_file.write_text(
                """[ProfileX]
alias = unsloth/alias1, alias2
""",
                encoding="utf-8",
            )

            # Match full alias with unsloth prefix
            self.assertEqual(
                map_repo_to_preset_alias(
                    "unsloth/alias1, alias2", presets_file=str(presets_file)
                ),
                "ProfileX",
            )

        mock_cfg = MagicMock()
        mock_cfg.sections.side_effect = RuntimeError("Mock error")
        with patch("advanced_benchmarks.load_presets_config", return_value=mock_cfg):
            self.assertEqual(map_repo_to_preset_alias("any-model"), "any-model")

    def test_load_presets_config_fallback_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fallback_ini = Path(tmp_dir) / "model_presets.ini"
            fallback_ini.write_text(
                "[FallbackProfile]\nthreads = 12\n", encoding="utf-8"
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "advanced_benchmarks.resolve_presets_path",
                    return_value=str(fallback_ini),
                ),
            ):
                load_presets_config.cache_clear()
                cfg = load_presets_config(None)
                self.assertIsNotNone(cfg)
                self.assertIn("FallbackProfile", cfg.sections())

    def test_map_repo_to_preset_alias_unsloth_prefix_normalization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text(
                """[*]
flash-attn = true

[Qwen3.8-27B-spec]
hf-repo = unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL
alias = unsloth/locallama-qwen, local-qwen

[PlainSection]
hf-repo = PlainRepo:latest
alias = plain-alias

[unsloth/PrefixSection]
hf-repo = PrefixRepo:latest
""",
                encoding="utf-8",
            )

            # 1. Query has prefix, hf-repo has prefix
            self.assertEqual(
                map_repo_to_preset_alias(
                    "unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL",
                    presets_file=str(presets_file),
                ),
                "Qwen3.8-27B-spec",
            )
            # 2. Query has NO prefix, hf-repo HAS prefix
            self.assertEqual(
                map_repo_to_preset_alias(
                    "Qwen3.8-27B-GGUF:UD-Q5_K_XL", presets_file=str(presets_file)
                ),
                "Qwen3.8-27B-spec",
            )
            # 3. Query has prefix, hf-repo has NO prefix
            self.assertEqual(
                map_repo_to_preset_alias(
                    "unsloth/PlainRepo:latest", presets_file=str(presets_file)
                ),
                "PlainSection",
            )
            # 4. Query has NO prefix, section HAS prefix
            self.assertEqual(
                map_repo_to_preset_alias(
                    "PrefixSection", presets_file=str(presets_file)
                ),
                "unsloth/PrefixSection",
            )
            # 5. Query has prefix, section has NO prefix
            self.assertEqual(
                map_repo_to_preset_alias(
                    "unsloth/PlainSection", presets_file=str(presets_file)
                ),
                "PlainSection",
            )
            # 6. Alias normalization: query has no prefix, alias has prefix
            self.assertEqual(
                map_repo_to_preset_alias(
                    "locallama-qwen", presets_file=str(presets_file)
                ),
                "Qwen3.8-27B-spec",
            )
            # 7. Alias normalization: query has prefix, alias has no prefix
            self.assertEqual(
                map_repo_to_preset_alias(
                    "unsloth/local-qwen", presets_file=str(presets_file)
                ),
                "Qwen3.8-27B-spec",
            )
            # 8. Single alias with prefix
            self.assertEqual(
                map_repo_to_preset_alias(
                    "unsloth/plain-alias", presets_file=str(presets_file)
                ),
                "PlainSection",
            )

    def test_get_preset_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            presets_file = Path(tmp_dir) / "model_presets.ini"
            presets_file.write_text(
                """[*]
flash-attn = true
parallel = 2

[MyProfile]
parallel = 4
n-gpu-layers = 33
custom-param = hello
""",
                encoding="utf-8",
            )

            meta = get_preset_metadata("MyProfile", presets_file=str(presets_file))
            self.assertEqual(meta["flash_attn"], "true")
            self.assertEqual(meta["parallel"], "4")
            self.assertEqual(meta["n_gpu_layers"], "33")
            self.assertEqual(meta["custom_param"], "hello")

            # Nonexistent section still gets globals and defaults
            meta_default = get_preset_metadata(
                "UnknownProfile", presets_file=str(presets_file)
            )
            self.assertEqual(meta_default["flash_attn"], "true")
            self.assertEqual(meta_default["parallel"], "2")
            self.assertEqual(meta_default["n_gpu_layers"], "99")

    def test_get_preset_metadata_nonexistent_file(self):
        meta = get_preset_metadata(
            "AnyProfile", presets_file="/nonexistent/model_presets.ini"
        )
        self.assertEqual(meta["parallel"], "1")
        self.assertEqual(meta["n_gpu_layers"], "99")
        self.assertEqual(meta["spec_type"], "None")


class TestASTSafetyCheck(unittest.TestCase):
    def test_safe_code_snippets(self):
        safe_snippets = [
            # Arithmetic & math
            ("1 + 2 * 3 - 4 / 2 ** 2", "arithmetic precedence"),
            ("x = 10 % 3\ny = 2 ** 5\nz = x + y", "basic math operators"),
            (
                "import math\nval = math.sqrt(16) + math.sin(math.pi / 2)",
                "math module functions",
            ),
            # String formatting & manipulation
            (
                "name = 'world'\nmsg = f'hello {name}'\ns = msg.upper().strip()",
                "f-strings and string methods",
            ),
            ("formatted = 'Value: {:.2f}'.format(3.14159)", "str.format"),
            ("parts = 'a,b,c'.split(',')\njoined = '-'.join(parts)", "split and join"),
            # Loops & control flow
            (
                """
def sum_evens(n):
    total = 0
    for i in range(n):
        if i % 2 == 0:
            total += i
        else:
            continue
    return total
""",
                "for loop with if/else/continue",
            ),
            (
                """
count = 0
while count < 5:
    count += 1
""",
                "while loop",
            ),
            (
                """
try:
    x = 10 / 2
except ZeroDivisionError:
    x = 0
finally:
    done = True
""",
                "try-except-finally block",
            ),
            # Dict, list, set, tuple manipulation
            ("d = {'a': 1, 'b': 2}\nd['c'] = 3\nval = d.get('a')", "dict manipulation"),
            (
                "items = [1, 2, 3]\nitems.append(4)\npopped = items.pop()",
                "list methods",
            ),
            ("squares = [x**2 for x in range(10) if x % 2 == 0]", "list comprehension"),
            ("unique = {x % 3 for x in range(10)}", "set comprehension"),
            ("lookup = {k: v for k, v in [('a', 1), ('b', 2)]}", "dict comprehension"),
            # Algorithms & recursion
            (
                """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
""",
                "recursive fibonacci",
            ),
            (
                """
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
""",
                "binary search algorithm",
            ),
            # Safe imports
            (
                """
import json
data = json.loads('{"a": 1}')
""",
                "json import",
            ),
            (
                """
import re
match = re.match(r'\\d+', '123')
""",
                "re import",
            ),
            ("import collections\nq = collections.deque()", "collections import"),
            (
                "import itertools\ncomb = list(itertools.combinations([1, 2], 2))",
                "itertools import",
            ),
            ("import random\nr = random.randint(1, 10)", "random import"),
            ("import time\nt = time.time()", "time import"),
            ("import datetime\nnow = datetime.datetime.now()", "datetime import"),
            # Benign patterns requested by reviewer
            ("import re\npattern = re.compile(r'\\d+')", "re.compile allowed"),
            (
                """
class Base:
    def __init__(self):
        self.x = 1
class Sub(Base):
    def __init__(self):
        super().__init__()
""",
                "super().__init__() allowed",
            ),
            (
                """
class Encapsulated:
    def __init__(self):
        self.__secret = 42
    def get_secret(self):
        return self.__secret
""",
                "private attributes self.__var allowed",
            ),
            (
                "from __future__ import annotations",
                "from __future__ import annotations allowed",
            ),
            (
                """
if __name__ == '__main__':
    msg = __doc__
""",
                "if __name__ == '__main__' and __doc__ allowed",
            ),
            (
                """
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
""",
                "safe dunder methods allowed",
            ),
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
                self.assertFalse(
                    safe, f"Expected exploit to be blocked for '{label}' ({code})"
                )
                self.assertIsNotNone(msg)
                self.assertTrue(
                    any(kw in msg for kw in ("Forbidden", "Syntax error", "Invalid")),
                    f"Expected security violation keyword in msg for '{label}', got: {msg}",
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

        res = call_endpoint(
            "http://127.0.0.1:8080", "test-model", "test prompt", max_tokens=256
        )

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
        self.assertAlmostEqual(
            res["decode_time"], 0.0
        )  # end_time - first_token_time (which is end_time)
        self.assertAlmostEqual(res["total_time"], 2.0)

    @patch("advanced_benchmarks.requests.post")
    def test_call_endpoint_http_non_200(self, mock_post):
        """Non-200 HTTP status (e.g. 500, 404) returns None."""
        for code in [400, 404, 500, 503]:
            with self.subTest(status_code=code):
                mock_resp = self._create_mock_response(
                    status_code=code, text="HTTP error"
                )
                mock_post.return_value = mock_resp
                res = call_endpoint(
                    "http://127.0.0.1:8080", "test-model", "test prompt"
                )
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
                res = call_endpoint(
                    "http://127.0.0.1:8080", "test-model", "test prompt"
                )
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

        res_no_usage = call_endpoint(
            "http://127.0.0.1:8080", "test-model", "test prompt"
        )
        self.assertIsNotNone(res_no_usage)
        self.assertEqual(res_no_usage["prompt_tokens"], 0)
        self.assertEqual(res_no_usage["completion_tokens"], 0)
        self.assertEqual(res_no_usage["prefill_speed"], 0)
        self.assertEqual(res_no_usage["decode_speed"], 0)


class TestGetModelSettingsFromEndpoint(unittest.TestCase):
    """Comprehensive test suite for get_model_settings_from_endpoint and its helpers."""

    def _create_mock_response(self, status_code=200, json_data=None):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        if json_data is not None:
            mock_resp.json.return_value = json_data
        else:
            mock_resp.json.return_value = {}
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        return mock_resp

    def test_parse_endpoint_model_args_helpers(self):
        """Test _parse_endpoint_model_args helper with various valid, invalid, and empty inputs."""
        # Empty or non-sequence inputs
        self.assertEqual(_parse_endpoint_model_args(None), {})
        self.assertEqual(_parse_endpoint_model_args([]), {})
        self.assertEqual(_parse_endpoint_model_args("not-a-list"), {})
        self.assertEqual(_parse_endpoint_model_args(123), {})

        # Trailing flags with no following value
        self.assertEqual(_parse_endpoint_model_args(["--threads"]), {})
        self.assertEqual(_parse_endpoint_model_args(["--batch-size"]), {})
        self.assertEqual(_parse_endpoint_model_args(["--ubatch-size"]), {})
        self.assertEqual(_parse_endpoint_model_args(["--cache-type-k"]), {})

        # Non-integer arguments
        invalid_args = [
            "--threads",
            "abc",
            "--batch-size",
            "xyz",
            "--ubatch-size",
            "bad",
        ]
        parsed = _parse_endpoint_model_args(invalid_args)
        self.assertIsNone(parsed["threads"])
        self.assertIsNone(parsed["batch_size"])
        self.assertIsNone(parsed["ubatch_size"])

        # Type error cases (e.g. non-string / None values in args)
        type_err_args = ["--threads", None, "--batch-size", [1], "--ubatch-size", {2}]
        parsed_types = _parse_endpoint_model_args(type_err_args)
        self.assertIsNone(parsed_types["threads"])
        self.assertIsNone(parsed_types["batch_size"])
        self.assertIsNone(parsed_types["ubatch_size"])

        # Valid arguments
        valid_args = [
            "--threads",
            "8",
            "--batch-size",
            "512",
            "--ubatch-size",
            "128",
            "--cache-type-k",
            "q8_0",
        ]
        expected = {
            "threads": 8,
            "batch_size": 512,
            "ubatch_size": 128,
            "kv_cache_quant": "q8_0",
        }
        self.assertEqual(_parse_endpoint_model_args(valid_args), expected)

        # Cache type v override
        args_with_v = ["--cache-type-k", "q8_0", "--cache-type-v", "q4_0"]
        self.assertEqual(
            _parse_endpoint_model_args(args_with_v)["kv_cache_quant"], "q4_0"
        )

    def test_parse_endpoint_preset_block_helpers(self):
        """Test _parse_endpoint_preset_block helper with various valid, invalid, and empty inputs."""
        # Empty or non-string inputs
        self.assertEqual(_parse_endpoint_preset_block(None), {})
        self.assertEqual(_parse_endpoint_preset_block(""), {})
        self.assertEqual(_parse_endpoint_preset_block(123), {})
        self.assertEqual(_parse_endpoint_preset_block(["not", "str"]), {})

        # Lines without '=' or empty lines
        unstructured = "\n# Just a comment\nno_equal_sign\n   \n"
        self.assertEqual(_parse_endpoint_preset_block(unstructured), {})

        # Invalid integer values
        invalid_preset = "threads=invalid\nbatch-size=foo\nubatch-size=bar"
        parsed = _parse_endpoint_preset_block(invalid_preset)
        self.assertIsNone(parsed["threads"])
        self.assertIsNone(parsed["batch_size"])
        self.assertIsNone(parsed["ubatch_size"])

        # Valid preset block with whitespace and both cache types
        valid_preset = """
        threads = 16
        batch-size = 1024
        ubatch-size = 256
        cache-type-k = q5_1
        cache-type-v = q8_0
        other-setting = ignored
        """
        expected = {
            "threads": 16,
            "batch_size": 1024,
            "ubatch_size": 256,
            "kv_cache_quant": "q8_0",
        }
        self.assertEqual(_parse_endpoint_preset_block(valid_preset), expected)

    @patch("advanced_benchmarks.requests.get")
    def test_successful_fetch_exact_match(self, mock_get):
        """Successful /v1/models fetch with exact target_model match."""
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "id": "qwen3.6-27b",
                        "status": {
                            "args": ["--threads", "8", "--batch-size", "512"],
                            "preset": "ubatch-size=128",
                        },
                    },
                    {
                        "id": "qwen3.6-27b-extended",
                        "status": {"args": ["--threads", "16"]},
                    },
                ]
            },
        )
        settings = get_model_settings_from_endpoint(
            "http://127.0.0.1:8081", "qwen3.6-27b"
        )
        self.assertEqual(settings["model_name"], "qwen3.6-27b")
        self.assertEqual(settings["threads"], 8)
        self.assertEqual(settings["batch_size"], 512)
        self.assertEqual(settings["ubatch_size"], 128)
        self.assertEqual(settings["speculative_draft_type"], "None")

    @patch("advanced_benchmarks.requests.get")
    def test_successful_fetch_partial_match(self, mock_get):
        """Successful /v1/models fetch with partial target_model match."""
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "id": "unsloth/Qwen3.6-27B-Instruct-GGUF",
                        "status": {
                            "args": ["--threads", "12", "--batch-size", "256"],
                            "preset": "cache-type-k=q4_k_m",
                        },
                    }
                ]
            },
        )
        settings = get_model_settings_from_endpoint(
            "http://127.0.0.1:8081", "Qwen3.6-27B"
        )
        self.assertEqual(settings["model_name"], "Qwen3.6-27B")
        self.assertEqual(settings["threads"], 12)
        self.assertEqual(settings["batch_size"], 256)
        self.assertEqual(settings["kv_cache_quant"], "q4_k_m")

    @patch("advanced_benchmarks.requests.get")
    def test_successful_fetch_no_matching_model(self, mock_get):
        """Successful /v1/models fetch but no item matches target_model."""
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={
                "data": [{"id": "llama-3-8b", "status": {"args": ["--threads", "8"]}}]
            },
        )
        settings = get_model_settings_from_endpoint(
            "http://127.0.0.1:8081", "unmatched-model"
        )
        self.assertEqual(settings["model_name"], "unmatched-model")
        self.assertIsNone(settings["threads"])
        self.assertIsNone(settings["batch_size"])
        self.assertIsNone(settings["ubatch_size"])
        self.assertEqual(settings["kv_cache_quant"], "Unknown")
        self.assertEqual(settings["speculative_draft_type"], "None")

    @patch("advanced_benchmarks.requests.get")
    def test_parsing_args_from_status_args(self, mock_get):
        """Parsing args from status.args (--threads, --batch-size, --ubatch-size, --cache-type-k, etc.)."""
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "id": "test-model",
                        "status": {
                            "args": [
                                "--threads",
                                "6",
                                "--batch-size",
                                "1024",
                                "--ubatch-size",
                                "512",
                                "--cache-type-k",
                                "q8_0",
                            ],
                            "preset": "",
                        },
                    }
                ]
            },
        )
        settings = get_model_settings_from_endpoint(
            "http://127.0.0.1:8081", "test-model"
        )
        self.assertEqual(settings["threads"], 6)
        self.assertEqual(settings["batch_size"], 1024)
        self.assertEqual(settings["ubatch_size"], 512)
        self.assertEqual(settings["kv_cache_quant"], "q8_0")

    @patch("advanced_benchmarks.requests.get")
    def test_parsing_preset_string_block_overrides_args(self, mock_get):
        """Parsing preset string block with valid values overriding status.args and invalid values resetting to None."""
        # Case 1: Preset overrides args
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "id": "test-model",
                        "status": {
                            "args": [
                                "--threads",
                                "4",
                                "--batch-size",
                                "256",
                                "--ubatch-size",
                                "64",
                                "--cache-type-k",
                                "q4_0",
                            ],
                            "preset": "threads=12\nbatch-size=512\nubatch-size=128\ncache-type-k=q5_1",
                        },
                    }
                ]
            },
        )
        settings = get_model_settings_from_endpoint(
            "http://127.0.0.1:8081", "test-model"
        )
        self.assertEqual(settings["threads"], 12)
        self.assertEqual(settings["batch_size"], 512)
        self.assertEqual(settings["ubatch_size"], 128)
        self.assertEqual(settings["kv_cache_quant"], "q5_1")

        # Case 2: Preset invalid values reset to None
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "id": "test-model",
                        "status": {
                            "args": [
                                "--threads",
                                "4",
                                "--batch-size",
                                "256",
                                "--ubatch-size",
                                "64",
                            ],
                            "preset": "threads=invalid\nbatch-size=invalid\nubatch-size=invalid",
                        },
                    }
                ]
            },
        )
        settings_invalid = get_model_settings_from_endpoint(
            "http://127.0.0.1:8081", "test-model"
        )
        self.assertIsNone(settings_invalid["threads"])
        self.assertIsNone(settings_invalid["batch_size"])
        self.assertIsNone(settings_invalid["ubatch_size"])

    @patch("advanced_benchmarks.requests.get")
    def test_speculative_draft_type_detection(self, mock_get):
        """Speculative draft type detection: ngram if 'mtp' in ID or 'spec' in args, otherwise None."""
        test_cases = [
            (
                "mtp in ID, no spec in args",
                "deepseek-v3-mtp-q4_k_m",
                ["--threads", "8"],
                "ngram",
            ),
            (
                "spec in args, no mtp in ID",
                "qwen3.6-27b",
                ["--speculative-draft", "model"],
                "ngram",
            ),
            ("spec flag in args", "llama-3", ["--spec-draft", "true"], "ngram"),
            ("both mtp in ID and spec in args", "model-mtp", ["--spec", "on"], "ngram"),
            (
                "neither mtp in ID nor spec in args",
                "standard-model",
                ["--threads", "8", "--batch-size", "512"],
                "None",
            ),
        ]
        for label, model_id, args, expected_draft in test_cases:
            with self.subTest(case=label):
                mock_get.return_value = self._create_mock_response(
                    status_code=200,
                    json_data={"data": [{"id": model_id, "status": {"args": args}}]},
                )
                settings = get_model_settings_from_endpoint(
                    "http://127.0.0.1:8081", model_id
                )
                self.assertEqual(settings["speculative_draft_type"], expected_draft)

    @patch("advanced_benchmarks.requests.get")
    def test_base_quantization_resolution(self, mock_get):
        """Base quantization resolution from model ID / name."""
        quant_cases = [
            ("q4_k_s", "Q4_K_S"),
            ("q4_k_m", "Q4_K_M"),
            ("q4_k_l", "Q4_K_L"),
            ("q4_k_xl", "Q4_K_XL"),
            ("q5_k_s", "Q5_K_S"),
            ("q5_k_m", "Q5_K_M"),
            ("q8_0", "Q8_0"),
            ("f16", "f16"),
        ]
        for q_token, expected_quant in quant_cases:
            with self.subTest(quant=q_token):
                # 1. Resolved from target_model name directly when endpoint is offline
                mock_get.side_effect = requests.exceptions.ConnectionError("Offline")
                settings = get_model_settings_from_endpoint(
                    "http://127.0.0.1:8081", f"model-{q_token}"
                )
                self.assertEqual(settings["base_quantization"], expected_quant)

        # 2. Resolved from returned model ID when target_model has no quant substring
        mock_get.side_effect = None
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={"data": [{"id": "model-q8_0", "status": {}}]},
        )
        settings = get_model_settings_from_endpoint("http://127.0.0.1:8081", "model")
        self.assertEqual(settings["base_quantization"], "Q8_0")

        # 3. Model ID overrides target_model if both have quantization
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={"data": [{"id": "model-q8_0-variant-q4_k_m", "status": {}}]},
        )
        settings = get_model_settings_from_endpoint(
            "http://127.0.0.1:8081", "model-q8_0"
        )
        self.assertEqual(settings["base_quantization"], "Q4_K_M")

        # 4. Unknown when neither has quantization
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={"data": [{"id": "model-plain", "status": {}}]},
        )
        settings = get_model_settings_from_endpoint(
            "http://127.0.0.1:8081", "model-plain"
        )
        self.assertEqual(settings["base_quantization"], "Unknown")

    @patch("advanced_benchmarks.requests.get")
    def test_fallback_non_200_status_code(self, mock_get):
        """Fallback when endpoint returns non-200 status code."""
        for code in (404, 500, 502):
            with self.subTest(status_code=code):
                mock_get.return_value = self._create_mock_response(status_code=code)
                settings = get_model_settings_from_endpoint(
                    "http://127.0.0.1:8081", "target-q4_k_m"
                )
                self.assertEqual(settings["model_name"], "target-q4_k_m")
                self.assertEqual(settings["base_quantization"], "Q4_K_M")
                self.assertEqual(settings["kv_cache_quant"], "Unknown")
                self.assertIsNone(settings["threads"])
                self.assertIsNone(settings["batch_size"])
                self.assertIsNone(settings["ubatch_size"])
                self.assertEqual(settings["speculative_draft_type"], "None")
                self.assertIn("profile_alias", settings)

    @patch("advanced_benchmarks.requests.get")
    def test_fallback_request_exception(self, mock_get):
        """Fallback when endpoint request raises an exception (requests.exceptions.RequestException)."""
        exceptions = [
            requests.exceptions.RequestException("Generic request exception"),
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.Timeout("Request timed out"),
            ValueError("Malformed JSON response"),
        ]
        for exc in exceptions:
            with self.subTest(exc=type(exc).__name__):
                if isinstance(exc, ValueError):
                    mock_resp = self._create_mock_response(status_code=200)
                    mock_resp.json.side_effect = exc
                    mock_get.return_value = mock_resp
                else:
                    mock_get.side_effect = exc

                settings = get_model_settings_from_endpoint(
                    "http://127.0.0.1:8081", "custom-model"
                )
                self.assertEqual(settings["model_name"], "custom-model")
                self.assertEqual(settings["base_quantization"], "Unknown")
                self.assertIsNone(settings["threads"])
                self.assertEqual(settings["speculative_draft_type"], "None")

    @patch("advanced_benchmarks.get_preset_metadata")
    @patch("advanced_benchmarks.map_repo_to_preset_alias")
    @patch("advanced_benchmarks.requests.get")
    def test_merging_with_preset_metadata_and_profile_alias(
        self, mock_get, mock_alias, mock_meta
    ):
        """Merging with preset metadata and profile_alias, respecting setdefault semantics."""
        mock_alias.return_value = "custom-alias"
        mock_meta.return_value = {
            "flash_attn": "true",
            "parallel": "4",
            "n_gpu_layers": "33",
            "fit": "false",
            "threads": 99,  # Should NOT overwrite threads parsed from endpoint
            "extra_custom_param": "preset_val",
        }
        mock_get.return_value = self._create_mock_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "id": "custom-alias",
                        "status": {"args": ["--threads", "16", "--batch-size", "256"]},
                    }
                ]
            },
        )

        settings = get_model_settings_from_endpoint(
            "http://127.0.0.1:8081", "custom-alias"
        )
        self.assertEqual(settings["profile_alias"], "custom-alias")
        # threads should be 16 from args, not overwritten by 99 from preset_metadata
        self.assertEqual(settings["threads"], 16)
        self.assertEqual(settings["batch_size"], 256)
        # Preset metadata fields should be merged in
        self.assertEqual(settings["flash_attn"], "true")
        self.assertEqual(settings["parallel"], "4")
        self.assertEqual(settings["n_gpu_layers"], "33")
        self.assertEqual(settings["fit"], "false")
        self.assertEqual(settings["extra_custom_param"], "preset_val")


class TestExtractCodeBlockFromResponse(unittest.TestCase):
    def test_extract_from_raw_response(self):
        raw = (
            "Thought process here.\n```python\ndef solve():\n    return 42\n```\nDone."
        )
        res = _extract_code_block_from_response(raw, "")
        self.assertEqual(res, "def solve():\n    return 42")

    def test_extract_from_reasoning_fallback(self):
        raw = "No code block in raw response."
        reasoning = "Trace:\n```python\ndef fix():\n    return 100\n```\nExplanation."
        res = _extract_code_block_from_response(raw, reasoning)
        self.assertEqual(res, "def fix():\n    return 100")

    def test_no_code_block_returns_empty(self):
        raw = "Plain text response with no blocks."
        reasoning = "Plain text thought trace."
        res = _extract_code_block_from_response(raw, reasoning)
        self.assertEqual(res, "")

    def test_none_or_empty_inputs(self):
        self.assertEqual(_extract_code_block_from_response(None, None), "")
        self.assertEqual(_extract_code_block_from_response("", ""), "")


class TestRunSweTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        self.toy_repo_dir = os.path.join(self.tmpdir, "toy_repo")
        os.makedirs(self.toy_repo_dir, exist_ok=True)
        self.code_path = os.path.join(self.toy_repo_dir, "calculator.py")
        self.test_path = os.path.join(self.toy_repo_dir, "test_calculator.py")
        self.orig_code = "def parse_and_eval(expr):\n    return eval(expr)\n"
        self.orig_test = (
            "import unittest\nclass TestCalc(unittest.TestCase):\n    pass\n"
        )
        with open(self.code_path, "w", encoding="utf-8") as f:
            f.write(self.orig_code)
        with open(self.test_path, "w", encoding="utf-8") as f:
            f.write(self.orig_test)
        self.fake_script = os.path.join(self.tmpdir, "advanced_benchmarks.py")
        self.file_patcher = patch.object(
            advanced_benchmarks, "__file__", self.fake_script
        )
        self.file_patcher.start()

    def tearDown(self):
        self.file_patcher.stop()
        self.tmpdir_obj.cleanup()

    def test_toy_repo_files_missing(self):
        # Missing calculator.py
        os.remove(self.code_path)
        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNone(res)

        # Restore code_path, remove test_path
        with open(self.code_path, "w", encoding="utf-8") as f:
            f.write(self.orig_code)
        os.remove(self.test_path)
        res2 = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNone(res2)

    @patch("advanced_benchmarks.call_endpoint")
    def test_call_endpoint_failure(self, mock_call):
        mock_call.return_value = None
        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNone(res)

    @patch("advanced_benchmarks.call_endpoint")
    def test_no_python_code_block_in_response(self, mock_call):
        mock_call.return_value = {
            "response": "Here is how you fix it: simply change + to *.",
            "reasoning": "Reasoning without code block.",
            "ttft": 0.2,
            "prefill_speed": 80.0,
            "decode_time": 0.6,
            "decode_speed": 40.0,
        }
        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res)
        self.assertFalse(res["passed"])
        self.assertEqual(res["benchmark"], "SWE-bench")

    @patch("advanced_benchmarks.call_endpoint")
    def test_code_block_fails_is_safe_code_ast_check(self, mock_call):
        mock_call.return_value = {
            "response": "```python\nimport subprocess\nsubprocess.run(['rm', '-rf', '/'])\n```",
            "reasoning": "",
            "ttft": 0.2,
            "prefill_speed": 80.0,
            "decode_time": 0.6,
            "decode_speed": 40.0,
        }
        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res)
        self.assertFalse(res["passed"])
        backup_path = self.code_path + ".bak"
        self.assertFalse(os.path.exists(backup_path))
        with open(self.code_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), self.orig_code)

    @patch("advanced_benchmarks.subprocess.run")
    @patch("advanced_benchmarks.call_endpoint")
    def test_code_block_passes_ast_and_tests_pass(self, mock_call, mock_run):
        mock_call.return_value = {
            "response": "```python\ndef parse_and_eval(expr):\n    return 42\n```",
            "reasoning": "Simple solution",
            "ttft": 0.15,
            "prefill_speed": 110.0,
            "decode_time": 0.4,
            "decode_speed": 55.0,
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Ran 1 test in 0.001s\n\nOK", stderr=""
        )

        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res)
        self.assertTrue(res["passed"])
        self.assertEqual(res["benchmark"], "SWE-bench")
        backup_path = self.code_path + ".bak"
        self.assertFalse(os.path.exists(backup_path))
        with open(self.code_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), self.orig_code)

    @patch("advanced_benchmarks.subprocess.run")
    @patch("advanced_benchmarks.call_endpoint")
    def test_code_block_passes_ast_and_tests_fail(self, mock_call, mock_run):
        mock_call.return_value = {
            "response": "```python\ndef parse_and_eval(expr):\n    return 0\n```",
            "reasoning": "",
            "ttft": 0.15,
            "prefill_speed": 110.0,
            "decode_time": 0.4,
            "decode_speed": 55.0,
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="FAILED (failures=1)"
        )

        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res)
        self.assertFalse(res["passed"])
        backup_path = self.code_path + ".bak"
        self.assertFalse(os.path.exists(backup_path))
        with open(self.code_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), self.orig_code)

    @patch("advanced_benchmarks.subprocess.run")
    @patch("advanced_benchmarks.call_endpoint")
    def test_subprocess_timeout_handled_gracefully(self, mock_call, mock_run):
        mock_call.return_value = {
            "response": "```python\ndef parse_and_eval(expr):\n    return 42\n```",
            "reasoning": "",
            "ttft": 0.15,
            "prefill_speed": 110.0,
            "decode_time": 0.4,
            "decode_speed": 55.0,
        }
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=30)

        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res)
        self.assertFalse(res["passed"])
        backup_path = self.code_path + ".bak"
        self.assertFalse(os.path.exists(backup_path))
        with open(self.code_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), self.orig_code)

    @patch("advanced_benchmarks.subprocess.run")
    @patch("advanced_benchmarks.call_endpoint")
    def test_subprocess_exception_and_backup_cleanup(self, mock_call, mock_run):
        mock_call.return_value = {
            "response": "```python\ndef parse_and_eval(expr):\n    return 42\n```",
            "reasoning": "",
            "ttft": 0.15,
            "prefill_speed": 110.0,
            "decode_time": 0.4,
            "decode_speed": 55.0,
        }
        mock_run.side_effect = RuntimeError("Uncaught subprocess failure")

        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res)
        self.assertFalse(res["passed"])
        backup_path = self.code_path + ".bak"
        self.assertFalse(os.path.exists(backup_path))
        with open(self.code_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), self.orig_code)

    @patch("advanced_benchmarks.subprocess.run")
    @patch("advanced_benchmarks.call_endpoint")
    def test_reasoning_trace_fallback_success(self, mock_call, mock_run):
        mock_call.return_value = {
            "response": "Here is my reasoning below.",
            "reasoning": "Thinking:\n```python\ndef parse_and_eval(expr):\n    return 42\n```",
            "ttft": 0.2,
            "prefill_speed": 90.0,
            "decode_time": 0.5,
            "decode_speed": 45.0,
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK", stderr=""
        )

        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res)
        self.assertTrue(res["passed"])

    @patch("advanced_benchmarks.subprocess.run")
    @patch("advanced_benchmarks.call_endpoint")
    def test_uses_buggy_calculator_fixture_when_available(self, mock_call, mock_run):
        fixtures_dir = os.path.join(self.toy_repo_dir, "fixtures")
        os.makedirs(fixtures_dir, exist_ok=True)
        buggy_path = os.path.join(fixtures_dir, "buggy_calculator.py")
        buggy_code = "# BUGGY FIXTURE CODE\ndef parse_and_eval(expr):\n    return 0\n"
        with open(buggy_path, "w", encoding="utf-8") as f:
            f.write(buggy_code)

        captured_prompt = {}

        def fake_call(endpoint, model, prompt, max_tokens=16384, api_key=None):
            captured_prompt["prompt"] = prompt
            return {
                "response": "```python\ndef parse_and_eval(expr):\n    return 42\n```",
                "reasoning": "",
                "ttft": 0.1,
                "prefill_speed": 100.0,
                "decode_time": 0.1,
                "decode_speed": 50.0,
            }

        mock_call.side_effect = fake_call
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK", stderr=""
        )

        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res)
        self.assertTrue(res["passed"])

        # Verify prompt contained the buggy fixture code and prompt instruction
        self.assertIn("# BUGGY FIXTURE CODE", captured_prompt.get("prompt", ""))
        self.assertIn(
            "Fix the order-of-operations bug in the file calculator.py so that all tests pass. "
            "If all tests already pass or once fixed, output the complete python code block "
            "immediately without exhaustive verification.",
            captured_prompt.get("prompt", ""),
        )

        # Verify calculator.py is safely restored to original clean state
        backup_path = self.code_path + ".bak"
        self.assertFalse(os.path.exists(backup_path))
        with open(self.code_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), self.orig_code)

    @patch("advanced_benchmarks.call_endpoint")
    def test_restores_clean_state_when_exception_in_endpoint(self, mock_call):
        fixtures_dir = os.path.join(self.toy_repo_dir, "fixtures")
        os.makedirs(fixtures_dir, exist_ok=True)
        buggy_path = os.path.join(fixtures_dir, "buggy_calculator.py")
        with open(buggy_path, "w", encoding="utf-8") as f:
            f.write("# BUGGY\n")

        mock_call.side_effect = RuntimeError("Endpoint network failure")

        with self.assertRaises(RuntimeError):
            run_swe_test("http://127.0.0.1:8081", "test-model")

        # Verify calculator.py was safely restored even on exception
        backup_path = self.code_path + ".bak"
        self.assertFalse(os.path.exists(backup_path))
        with open(self.code_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), self.orig_code)

    @patch("advanced_benchmarks.subprocess.run")
    @patch("advanced_benchmarks.call_endpoint")
    def test_restores_clean_state_when_tests_fail_with_fixture(self, mock_call, mock_run):
        fixtures_dir = os.path.join(self.toy_repo_dir, "fixtures")
        os.makedirs(fixtures_dir, exist_ok=True)
        buggy_path = os.path.join(fixtures_dir, "buggy_calculator.py")
        with open(buggy_path, "w", encoding="utf-8") as f:
            f.write("# BUGGY\n")

        mock_call.return_value = {
            "response": "```python\ndef parse_and_eval(expr):\n    return 0\n```",
            "reasoning": "",
            "ttft": 0.1,
            "prefill_speed": 100.0,
            "decode_time": 0.1,
            "decode_speed": 50.0,
        }
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="FAILED"
        )

        res = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res)
        self.assertFalse(res["passed"])

        # Verify calculator.py is restored and backup removed
        backup_path = self.code_path + ".bak"
        self.assertFalse(os.path.exists(backup_path))
        with open(self.code_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), self.orig_code)

    @patch("advanced_benchmarks.call_endpoint")
    def test_run_swe_test_forwards_custom_max_tokens(self, mock_call):
        mock_call.return_value = {
            "response": "```python\npass\n```",
            "reasoning": "",
            "ttft": 0.1,
            "prefill_speed": 100.0,
        }
        # Test default max_tokens is 16384
        res_default = run_swe_test("http://127.0.0.1:8081", "test-model")
        self.assertIsNotNone(res_default)
        self.assertEqual(mock_call.call_args.kwargs.get("max_tokens"), 16384)

        # Test custom max_tokens forwarded
        mock_call.reset_mock()
        res_custom = run_swe_test("http://127.0.0.1:8081", "test-model", max_tokens=8192)
        self.assertIsNotNone(res_custom)
        self.assertEqual(mock_call.call_args.kwargs.get("max_tokens"), 8192)


class TestMainRunner(unittest.TestCase):
    @patch("advanced_benchmarks._save_run_data")
    @patch("advanced_benchmarks.run_swe_test")
    @patch("advanced_benchmarks.run_longbench_test")
    @patch("advanced_benchmarks.run_ruler_test")
    @patch("advanced_benchmarks.run_needle_test")
    def test_cli_benchmark_needle(
        self, mock_needle, mock_ruler, mock_long, mock_swe, mock_save
    ):
        mock_needle.return_value = {
            "benchmark": "Needle",
            "passed": True,
            "prompt_tokens": 100,
            "ttft": 0.1,
            "prefill_speed": 50.0,
            "decode_speed": 25.0,
        }
        results = main(
            [
                "--benchmark",
                "needle",
                "--endpoint",
                "http://127.0.0.1:8081",
                "--model",
                "Qwen",
                "--tokens",
                "50000",
            ]
        )
        self.assertEqual(len(results), 1)
        mock_needle.assert_called_once_with(
            "http://127.0.0.1:8081", "Qwen", tokens=50000
        )
        mock_ruler.assert_not_called()
        mock_long.assert_not_called()
        mock_swe.assert_not_called()
        mock_save.assert_called_once()

    @patch("advanced_benchmarks._save_run_data")
    @patch("advanced_benchmarks.run_swe_test")
    @patch("advanced_benchmarks.run_longbench_test")
    @patch("advanced_benchmarks.run_ruler_test")
    @patch("advanced_benchmarks.run_needle_test")
    def test_cli_benchmark_ruler(
        self, mock_needle, mock_ruler, mock_long, mock_swe, mock_save
    ):
        mock_ruler.return_value = {
            "benchmark": "RULER",
            "passed": True,
            "prompt_tokens": 100,
            "ttft": 0.1,
            "prefill_speed": 50.0,
            "decode_speed": 25.0,
        }
        results = main(["--benchmark", "ruler"])
        self.assertEqual(len(results), 1)
        mock_needle.assert_not_called()
        mock_ruler.assert_called_once()
        mock_long.assert_not_called()
        mock_swe.assert_not_called()

    @patch("advanced_benchmarks._save_run_data")
    @patch("advanced_benchmarks.run_swe_test")
    @patch("advanced_benchmarks.run_longbench_test")
    @patch("advanced_benchmarks.run_ruler_test")
    @patch("advanced_benchmarks.run_needle_test")
    def test_cli_benchmark_longbench(
        self, mock_needle, mock_ruler, mock_long, mock_swe, mock_save
    ):
        mock_long.return_value = {
            "benchmark": "LongBench",
            "passed": True,
            "prompt_tokens": 100,
            "ttft": 0.1,
            "prefill_speed": 50.0,
            "decode_speed": 25.0,
        }
        results = main(["--benchmark", "longbench"])
        self.assertEqual(len(results), 1)
        mock_needle.assert_not_called()
        mock_ruler.assert_not_called()
        mock_long.assert_called_once()
        mock_swe.assert_not_called()

    @patch("advanced_benchmarks._save_run_data")
    @patch("advanced_benchmarks.run_swe_test")
    @patch("advanced_benchmarks.run_longbench_test")
    @patch("advanced_benchmarks.run_ruler_test")
    @patch("advanced_benchmarks.run_needle_test")
    def test_cli_benchmark_swe(
        self, mock_needle, mock_ruler, mock_long, mock_swe, mock_save
    ):
        mock_swe.return_value = {
            "benchmark": "SWE-bench",
            "passed": True,
            "prompt_tokens": 100,
            "ttft": 0.1,
            "prefill_speed": 50.0,
            "decode_speed": 25.0,
        }
        results = main(["--benchmark", "swe"])
        self.assertEqual(len(results), 1)
        mock_needle.assert_not_called()
        mock_ruler.assert_not_called()
        mock_long.assert_not_called()
        mock_swe.assert_called_once()

    @patch("advanced_benchmarks._save_run_data")
    @patch("advanced_benchmarks.run_swe_test")
    @patch("advanced_benchmarks.run_longbench_test")
    @patch("advanced_benchmarks.run_ruler_test")
    @patch("advanced_benchmarks.run_needle_test")
    def test_cli_benchmark_all_and_default(
        self, mock_needle, mock_ruler, mock_long, mock_swe, mock_save
    ):
        res_item = {
            "benchmark": "Test",
            "passed": True,
            "prompt_tokens": 100,
            "ttft": 0.1,
            "prefill_speed": 50.0,
            "decode_speed": 25.0,
        }
        mock_needle.return_value = res_item
        mock_ruler.return_value = res_item
        mock_long.return_value = res_item
        mock_swe.return_value = res_item

        # Explicit --benchmark all
        results = main(["--benchmark", "all"])
        self.assertEqual(len(results), 4)
        self.assertEqual(mock_needle.call_count, 1)
        self.assertEqual(mock_ruler.call_count, 1)
        self.assertEqual(mock_long.call_count, 1)
        self.assertEqual(mock_swe.call_count, 1)

        # Default with empty arguments -> executes all
        mock_needle.reset_mock()
        mock_ruler.reset_mock()
        mock_long.reset_mock()
        mock_swe.reset_mock()
        results2 = main([])
        self.assertEqual(len(results2), 4)
        self.assertEqual(mock_needle.call_count, 1)
        self.assertEqual(mock_ruler.call_count, 1)
        self.assertEqual(mock_long.call_count, 1)
        self.assertEqual(mock_swe.call_count, 1)

    @patch("advanced_benchmarks._save_run_data")
    @patch("advanced_benchmarks.run_swe_test")
    @patch("advanced_benchmarks.run_longbench_test")
    @patch("advanced_benchmarks.run_ruler_test")
    @patch("advanced_benchmarks.run_needle_test")
    def test_cli_individual_flags(
        self, mock_needle, mock_ruler, mock_long, mock_swe, mock_save
    ):
        res_item = {
            "benchmark": "B",
            "passed": True,
            "prompt_tokens": 10,
            "ttft": 0.1,
            "prefill_speed": 1.0,
            "decode_speed": 1.0,
        }
        mock_needle.return_value = res_item
        mock_ruler.return_value = res_item
        mock_long.return_value = res_item
        mock_swe.return_value = res_item

        main(["--needle"])
        mock_needle.assert_called_once()
        mock_ruler.assert_not_called()

        mock_needle.reset_mock()
        main(["--ruler"])
        mock_ruler.assert_called_once()

        mock_ruler.reset_mock()
        main(["--longbench"])
        mock_long.assert_called_once()

        mock_long.reset_mock()
        main(["--swe"])
        mock_swe.assert_called_once()

        mock_swe.reset_mock()
        main(["--all"])
        self.assertEqual(mock_needle.call_count, 1)
        self.assertEqual(mock_ruler.call_count, 1)
        self.assertEqual(mock_long.call_count, 1)
        self.assertEqual(mock_swe.call_count, 1)

    @patch("advanced_benchmarks.get_model_settings_from_endpoint")
    @patch("advanced_benchmarks.run_needle_test")
    def test_output_json_file_writing(self, mock_needle, mock_settings):
        mock_settings.return_value = {"model_name": "TestModel", "threads": 4}
        mock_needle.return_value = {
            "benchmark": "Needle",
            "passed": True,
            "prompt_tokens": 1500,
            "ttft": 0.25,
            "prefill_speed": 120.0,
            "decode_speed": 40.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "custom_runs", "test_output.json")
            cli_args = ["--benchmark", "needle", "--output", out_file]
            results = main(cli_args)

            self.assertEqual(len(results), 1)
            self.assertTrue(os.path.exists(out_file))

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(
                data["run_metadata"]["target_endpoint"], "http://127.0.0.1:8083"
            )
            self.assertEqual(data["run_metadata"]["cli_arguments"], cli_args)
            self.assertEqual(data["model_settings"]["model_name"], "TestModel")
            self.assertEqual(data["throughput_metrics"]["prefill_speed"], 120.0)
            self.assertEqual(data["throughput_metrics"]["decode_speed"], 40.0)
            self.assertEqual(data["throughput_metrics"]["ttft"], 0.25)
            self.assertEqual(data["reasoning_accuracy"]["needle"], "Pass")
            self.assertEqual(data["reasoning_accuracy"]["ruler"], "N/A")

    @patch("advanced_benchmarks.run_needle_test")
    def test_empty_results_no_output_written(self, mock_needle):
        mock_needle.return_value = None
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "not_created.json")
            results = main(["--benchmark", "needle", "--output", out_file])
            self.assertEqual(results, [])
            self.assertFalse(os.path.exists(out_file))

    @patch("advanced_benchmarks.get_model_settings_from_endpoint")
    def test_save_run_data_default_history_path_and_metrics(self, mock_settings):
        mock_settings.return_value = {"model_name": "TestModel"}
        results = [
            {
                "benchmark": "Needle",
                "passed": True,
                "prefill_speed": 100.0,
                "decode_speed": 50.0,
                "ttft": 0.1,
            },
            {
                "benchmark": "SWE-bench",
                "passed": False,
                "prefill_speed": 80.0,
                "decode_speed": 30.0,
                "ttft": 0.3,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_script = os.path.join(tmpdir, "advanced_benchmarks.py")
            with patch.object(advanced_benchmarks, "__file__", fake_script):
                out_path = _save_run_data(
                    results, "http://127.0.0.1:8081", "TestModel", ["--all"]
                )
                self.assertTrue(os.path.exists(out_path))
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertAlmostEqual(
                    data["throughput_metrics"]["prefill_speed"], 90.0
                )
                self.assertAlmostEqual(data["throughput_metrics"]["decode_speed"], 40.0)
                self.assertAlmostEqual(data["throughput_metrics"]["ttft"], 0.2)
                self.assertEqual(data["reasoning_accuracy"]["needle"], "Pass")
                self.assertEqual(data["reasoning_accuracy"]["swe_bench"], "Fail")

    def test_print_benchmark_summary(self):
        results = [
            {
                "benchmark": "Needle",
                "passed": True,
                "prompt_tokens": 1000,
                "ttft": 0.2,
                "prefill_speed": 100.0,
                "decode_speed": 40.0,
            },
            {
                "benchmark": "SWE-bench",
                "passed": False,
                "prompt_tokens": 500,
                "ttft": 0.3,
                "prefill_speed": 80.0,
                "decode_speed": 30.0,
            },
        ]
        with patch("builtins.print") as mock_print:
            _print_benchmark_summary(results)
            mock_print.assert_called()


class TestApiKeyAndAuthSupport(unittest.TestCase):
    """Test API key authentication in call_endpoint, get_model_settings_from_endpoint, benchmarks, and CLI."""

    def _create_mock_response(self, status_code=200, lines=None):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = "OK"
        mock_resp.iter_lines.return_value = lines if lines is not None else []
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        return mock_resp

    @patch("advanced_benchmarks.requests.post")
    def test_call_endpoint_with_api_key(self, mock_post):
        mock_resp = self._create_mock_response(status_code=200, lines=[b"data: [DONE]"])
        mock_post.return_value = mock_resp

        call_endpoint(
            "http://127.0.0.1:8080", "test-model", "test prompt", api_key="sk-explicit-token"
        )
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-explicit-token")

    @patch("advanced_benchmarks.requests.post")
    def test_call_endpoint_with_env_var_api_key(self, mock_post):
        mock_resp = self._create_mock_response(status_code=200, lines=[b"data: [DONE]"])
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-token"}):
            call_endpoint(
                "http://127.0.0.1:8080", "test-model", "test prompt"
            )
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-env-token")

    @patch("advanced_benchmarks.requests.get")
    def test_get_model_settings_from_endpoint_with_api_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "test-model", "status": {}}]}
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_get.return_value = mock_resp

        get_model_settings_from_endpoint(
            "http://127.0.0.1:8080", "test-model", api_key="sk-settings-key"
        )
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-settings-key")

    @patch("advanced_benchmarks.requests.get")
    def test_get_model_settings_from_endpoint_with_env_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "test-model", "status": {}}]}
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-settings-key"}):
            get_model_settings_from_endpoint(
                "http://127.0.0.1:8080", "test-model"
            )
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-env-settings-key")

    @patch("advanced_benchmarks.call_endpoint")
    def test_run_needle_test_forwards_api_key(self, mock_call):
        mock_call.return_value = {
            "response": "The code is BANANA_SPLIT",
            "reasoning": "",
            "ttft": 0.1,
            "decode_time": 0.2,
            "prefill_speed": 100.0,
            "decode_speed": 50.0,
        }
        res = advanced_benchmarks.run_needle_test(
            "http://127.0.0.1:8080", "test-model", tokens=100, api_key="needle-key"
        )
        self.assertIsNotNone(res)
        mock_call.assert_called_once()
        self.assertEqual(mock_call.call_args.kwargs.get("api_key"), "needle-key")

    @patch("advanced_benchmarks.call_endpoint")
    def test_run_ruler_test_forwards_api_key(self, mock_call):
        mock_call.return_value = {
            "response": "The value is 93",
            "reasoning": "",
            "ttft": 0.1,
            "decode_time": 0.2,
            "prefill_speed": 100.0,
            "decode_speed": 50.0,
        }
        res = advanced_benchmarks.run_ruler_test(
            "http://127.0.0.1:8080", "test-model", tokens=100, api_key="ruler-key"
        )
        self.assertIsNotNone(res)
        mock_call.assert_called_once()
        self.assertEqual(mock_call.call_args.kwargs.get("api_key"), "ruler-key")

    @patch("advanced_benchmarks.call_endpoint")
    def test_run_longbench_test_forwards_api_key(self, mock_call):
        mock_call.return_value = {
            "response": "In 1452",
            "reasoning": "",
            "ttft": 0.1,
            "decode_time": 0.2,
            "prefill_speed": 100.0,
            "decode_speed": 50.0,
        }
        res = advanced_benchmarks.run_longbench_test(
            "http://127.0.0.1:8080", "test-model", tokens=100, api_key="longbench-key"
        )
        self.assertIsNotNone(res)
        mock_call.assert_called_once()
        self.assertEqual(mock_call.call_args.kwargs.get("api_key"), "longbench-key")

    @patch("advanced_benchmarks.call_endpoint")
    def test_run_swe_test_forwards_api_key(self, mock_call):
        mock_call.return_value = {
            "response": "```python\npass\n```",
            "reasoning": "",
            "ttft": 0.1,
            "prefill_speed": 100.0,
        }
        res = run_swe_test("http://127.0.0.1:8080", "test-model", api_key="swe-key")
        self.assertIsNotNone(res)
        mock_call.assert_called_once()
        self.assertEqual(mock_call.call_args.kwargs.get("api_key"), "swe-key")

    @patch("advanced_benchmarks._save_run_data")
    @patch("advanced_benchmarks.run_needle_test")
    def test_cli_api_key_argument(self, mock_needle, mock_save):
        mock_needle.return_value = {
            "benchmark": "Needle",
            "passed": True,
            "prompt_tokens": 100,
            "ttft": 0.1,
            "prefill_speed": 50.0,
            "decode_speed": 25.0,
        }
        results = main(
            [
                "--benchmark", "needle",
                "--api-key", "sk-cli-token",
            ]
        )
        self.assertEqual(len(results), 1)
        mock_needle.assert_called_once()
        self.assertEqual(mock_needle.call_args.kwargs.get("api_key"), "sk-cli-token")
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.kwargs.get("api_key"), "sk-cli-token")

    @patch("advanced_benchmarks.requests.post")
    def test_call_endpoint_env_fallback(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}',
            b'data: [DONE]'
        ]
        mock_post.return_value.__enter__.return_value = mock_resp

        # API_KEY priority
        with patch.dict(os.environ, {"API_KEY": "env-call-key", "OPENAI_API_KEY": "openai-call-key"}):
            advanced_benchmarks.call_endpoint("http://127.0.0.1:8080", "m1", "prompt")
            _, kwargs = mock_post.call_args
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer env-call-key")

        # OPENAI_API_KEY fallback
        mock_post.reset_mock()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-call-key"}, clear=True):
            advanced_benchmarks.call_endpoint("http://127.0.0.1:8080", "m1", "prompt")
            _, kwargs = mock_post.call_args
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer openai-call-key")

    @patch("advanced_benchmarks.requests.get")
    def test_get_model_settings_from_endpoint_env_fallback(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "m1"}]}
        mock_get.return_value.__enter__.return_value = mock_resp

        with patch.dict(os.environ, {"API_KEY": "env-model-key"}, clear=True):
            advanced_benchmarks.get_model_settings_from_endpoint("http://127.0.0.1:8080", "m1")
            _, kwargs = mock_get.call_args
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer env-model-key")

    def test_save_run_data_redacts_api_key_in_cli_arguments(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "run_test.json")
            cli_args = ["--endpoint", "http://127.0.0.1:8080", "--api-key", "super-secret-pass", "--mode", "all"]
            results = [{
                "benchmark": "Needle",
                "passed": True,
                "prefill_speed": 100.0,
                "decode_speed": 50.0,
                "ttft": 0.1,
            }]
            with patch("advanced_benchmarks.get_model_settings_from_endpoint", return_value={"model_name": "m1"}):
                advanced_benchmarks._save_run_data(results, "http://127.0.0.1:8080", "m1", cli_args, output_path=out_file)

            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                saved = json.load(f)

            saved_cli = saved["run_metadata"]["cli_arguments"]
            self.assertIn("--api-key", saved_cli)
            self.assertIn("********", saved_cli)
            self.assertNotIn("super-secret-pass", saved_cli)


if __name__ == "__main__":
    unittest.main()
