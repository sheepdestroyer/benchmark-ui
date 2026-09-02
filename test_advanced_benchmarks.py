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

if __name__ == "__main__":
    unittest.main()
