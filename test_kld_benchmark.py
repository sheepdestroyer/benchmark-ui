import unittest
import math
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
