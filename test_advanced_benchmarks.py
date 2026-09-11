import unittest
from unittest.mock import patch
from advanced_benchmarks import is_safe_code, generate_filler_text

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

if __name__ == "__main__":
    unittest.main()
