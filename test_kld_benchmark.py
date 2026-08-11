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
from kld_benchmark import parse_metrics

class TestKLDBenchmark(unittest.TestCase):
    def test_parse_metrics_all_present(self):
        output = """
        Final perplexity: 5.4321
        Mean KLD: 0.00123
        Same top p: 98.76
        """
        metrics = parse_metrics(output)
        self.assertAlmostEqual(metrics["ppl"], 5.4321)
        self.assertAlmostEqual(metrics["kld"], 0.00123)
        self.assertAlmostEqual(metrics["same_top"], 98.76)

    def test_parse_metrics_missing(self):
        output = "Some random text that does not contain metrics"
        expected = {
            "ppl": None,
            "kld": None,
            "same_top": None
        }
        self.assertEqual(parse_metrics(output), expected)

    def test_parse_metrics_partially_missing(self):
        metrics = parse_metrics("Mean KLD: 0.5")
        self.assertIsNone(metrics["ppl"])
        self.assertAlmostEqual(metrics["kld"], 0.5)
        self.assertIsNone(metrics["same_top"])

    def test_parse_metrics_ppl_formats(self):
        test_cases = [
            ("Final perplexity: 1.23", 1.23),
            ("Final estimate: PPL = 2.34", 2.34),
            ("Mean PPL(Q) : 3.45", 3.45),
        ]
        for out_str, expected_ppl in test_cases:
            with self.subTest(output=out_str):
                metrics = parse_metrics(out_str)
                self.assertAlmostEqual(metrics["ppl"], expected_ppl)

    def test_parse_metrics_scientific_notation(self):
        output = """
        Final perplexity: 1.2e3
        Mean KLD: 5.4e-2
        Same top p: +9.9E1
        """
        metrics = parse_metrics(output)
        self.assertAlmostEqual(metrics["ppl"], 1200.0)
        self.assertAlmostEqual(metrics["kld"], 0.054)
        self.assertAlmostEqual(metrics["same_top"], 99.0)

    def test_parse_metrics_special_values(self):
        output = """
        Final perplexity: inf
        Mean KLD: nan
        Same top p: -inf
        """
        metrics = parse_metrics(output)
        self.assertEqual(metrics["ppl"], float('inf'))
        self.assertTrue(math.isnan(metrics["kld"]))
        self.assertEqual(metrics["same_top"], float('-inf'))

if __name__ == '__main__':
    unittest.main()
