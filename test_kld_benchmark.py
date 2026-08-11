import unittest
import math
from kld_benchmark import sanitize_nan_inf

class TestKldBenchmark(unittest.TestCase):
    def test_sanitize_nan_inf_floats(self):
        self.assertEqual(sanitize_nan_inf(1.5), 1.5)
        self.assertEqual(sanitize_nan_inf(0.0), 0.0)
        self.assertEqual(sanitize_nan_inf(-2.5), -2.5)

    def test_sanitize_nan_inf_special_floats(self):
        self.assertIsNone(sanitize_nan_inf(math.nan))
        self.assertIsNone(sanitize_nan_inf(math.inf))
        self.assertIsNone(sanitize_nan_inf(-math.inf))

    def test_sanitize_nan_inf_other_types(self):
        self.assertEqual(sanitize_nan_inf(42), 42)
        self.assertEqual(sanitize_nan_inf("string"), "string")
        self.assertIsNone(sanitize_nan_inf(None))
        self.assertEqual(sanitize_nan_inf(True), True)

    def test_sanitize_nan_inf_list(self):
        test_list = [1.0, math.nan, 2.0, math.inf, -math.inf, "test"]
        expected_list = [1.0, None, 2.0, None, None, "test"]
        self.assertEqual(sanitize_nan_inf(test_list), expected_list)

    def test_sanitize_nan_inf_dict(self):
        test_dict = {"a": 1.0, "b": math.nan, "c": math.inf, "d": "str"}
        expected_dict = {"a": 1.0, "b": None, "c": None, "d": "str"}
        self.assertEqual(sanitize_nan_inf(test_dict), expected_dict)

    def test_sanitize_nan_inf_nested(self):
        test_nested = {
            "list": [math.nan, {"nested_inf": math.inf, "normal": 42}],
            "dict": {"nested_nan": [1.0, math.nan], "text": "value"}
        }
        expected_nested = {
            "list": [None, {"nested_inf": None, "normal": 42}],
            "dict": {"nested_nan": [1.0, None], "text": "value"}
        }
        self.assertEqual(sanitize_nan_inf(test_nested), expected_nested)

if __name__ == '__main__':
    unittest.main()
