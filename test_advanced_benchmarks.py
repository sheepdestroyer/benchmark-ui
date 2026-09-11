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
            ("import posix", "import posix", "Forbidden"),
            ("import pty", "import pty", "Forbidden"),
            ("import shutil", "import shutil", "Forbidden"),
            ("import ctypes", "import ctypes", "Forbidden"),
            ("import sys", "import sys", "Forbidden"),
            ("x = __import__('os')", "dunder import identifier", "Forbidden"),
            ("x = getattr(obj, 'attr')", "dangerous built-in call getattr", "Forbidden"),
            ("setattr(obj, 'attr', 1)", "dangerous built-in call setattr", "Forbidden"),
            ("delattr(obj, 'attr')", "dangerous built-in call delattr", "Forbidden"),
            ("exec('print(1)')", "dangerous built-in call exec", "Forbidden"),
            ("open('/etc/passwd')", "dangerous built-in call open", "Forbidden"),
            ("compile('1+1', '', 'eval')", "dangerous built-in call compile", "Forbidden"),
            ("x.__builtins__", "dunder attribute lookup __builtins__", "Forbidden"),
            ("x.__subclasses__()", "dunder attribute lookup __subclasses__", "Forbidden"),
            ("x.__globals__", "dunder attribute lookup __globals__", "Forbidden"),
            ("x.__code__", "dunder attribute lookup __code__", "Forbidden"),
            ("getattr(x, '__builtins__')", "constant string or built-in getattr", "Forbidden"),
        ]
        for code, label, expected_keyword in cases:
            with self.subTest(case=label):
                safe, msg = is_safe_code(code)
                self.assertFalse(safe)
                self.assertIsNotNone(msg)
                self.assertIn(expected_keyword, msg)

    def test_safe_math_and_data_code(self):
        safe_codes = [
            """
def add(a, b):
    return a + b
result = add(10, 20)
""",
            """
numbers = [1, 2, 3, 4, 5]
squared = [x ** 2 for x in numbers]
data = {'a': 1, 'b': 2}
val = data.get('a', 0)
""",
            """
class Calculator:
    def parse_and_eval(self, expr):
        tokens = expr.split()
        return float(tokens[0]) + float(tokens[1])
"""
        ]
        for code in safe_codes:
            safe, msg = is_safe_code(code)
            self.assertTrue(safe, f"Expected safe code to pass, but got error: {msg}")
            self.assertIsNone(msg)

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

if __name__ == "__main__":
    unittest.main()
