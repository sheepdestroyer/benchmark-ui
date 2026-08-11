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
        expected = {
            "ppl": 5.4321,
            "kld": 0.00123,
            "same_top": 98.76
        }
        self.assertEqual(parse_metrics(output), expected)

    def test_parse_metrics_missing(self):
        output = "Some random text that does not contain metrics"
        expected = {
            "ppl": None,
            "kld": None,
            "same_top": None
        }
        self.assertEqual(parse_metrics(output), expected)

    def test_parse_metrics_ppl_formats(self):
        # Format 1
        out1 = "Final perplexity: 1.23"
        self.assertEqual(parse_metrics(out1)["ppl"], 1.23)

        # Format 2
        out2 = "Final estimate: PPL = 2.34"
        self.assertEqual(parse_metrics(out2)["ppl"], 2.34)

        # Format 3
        out3 = "Mean PPL(Q) : 3.45"
        self.assertEqual(parse_metrics(out3)["ppl"], 3.45)

    def test_parse_metrics_scientific_notation(self):
        output = """
        Final perplexity: 1.2e3
        Mean KLD: 5.4e-2
        Same top p: +9.9E1
        """
        metrics = parse_metrics(output)
        self.assertEqual(metrics["ppl"], 1200.0)
        self.assertEqual(metrics["kld"], 0.054)
        self.assertEqual(metrics["same_top"], 99.0)

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
