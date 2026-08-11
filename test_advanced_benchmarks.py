import unittest
from advanced_benchmarks import is_safe_code

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

    def test_forbidden_import(self):
        code = "import subprocess"
        safe, msg = is_safe_code(code)
        self.assertFalse(safe)
        self.assertIn("Forbidden import", msg)

    def test_forbidden_import_module_sub(self):
        code = "import subprocess.Popen"
        safe, msg = is_safe_code(code)
        self.assertFalse(safe)
        self.assertIn("Forbidden import", msg)

    def test_forbidden_import_from_module(self):
        code = "from subprocess import Popen"
        safe, msg = is_safe_code(code)
        self.assertFalse(safe)
        self.assertIn("Forbidden import module: subprocess", msg)

    def test_forbidden_import_from_name(self):
        code = "from os import system"
        safe, msg = is_safe_code(code)
        self.assertFalse(safe)
        self.assertIn("Forbidden import: os.system", msg)

    def test_forbidden_attribute(self):
        code = "import os\nos.system('ls')"
        safe, msg = is_safe_code(code)
        self.assertFalse(safe)
        self.assertIn("Forbidden attribute usage: os.system", msg)

    def test_forbidden_name(self):
        code = "eval('1 + 1')"
        safe, msg = is_safe_code(code)
        self.assertFalse(safe)
        self.assertIn("Forbidden identifier usage: eval", msg)

if __name__ == '__main__':
    unittest.main()
