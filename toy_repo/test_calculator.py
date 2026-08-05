import unittest
from calculator import parse_and_eval

class TestCalculator(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(parse_and_eval("3 + 5"), 8)
        self.assertEqual(parse_and_eval("0 + 0"), 0)
        self.assertEqual(parse_and_eval("10 + 20 + 30"), 60)

    def test_subtraction(self):
        self.assertEqual(parse_and_eval("10 - 4"), 6)
        self.assertEqual(parse_and_eval("5 - 10"), -5)
        self.assertEqual(parse_and_eval("20 - 5 - 3"), 12)

    def test_multiplication(self):
        self.assertEqual(parse_and_eval("4 * 5"), 20)
        self.assertEqual(parse_and_eval("3 * 0"), 0)
        self.assertEqual(parse_and_eval("2 * 3 * 4"), 24)

    def test_division(self):
        self.assertEqual(parse_and_eval("20 / 4"), 5)
        self.assertEqual(parse_and_eval("10 / 4"), 2.5)

    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            parse_and_eval("10 / 0")

    def test_float_operations(self):
        self.assertAlmostEqual(parse_and_eval("3.5 + 2.1"), 5.6)
        self.assertAlmostEqual(parse_and_eval("7.5 / 2.5"), 3.0)
        self.assertAlmostEqual(parse_and_eval("1.2 * 3.4"), 4.08)

    def test_whitespace_and_empty(self):
        self.assertEqual(parse_and_eval(""), 0)
        self.assertEqual(parse_and_eval("   "), 0)
        self.assertEqual(parse_and_eval("  4  +  5  "), 9)

    def test_complex_expressions(self):
        self.assertEqual(parse_and_eval("3 + 4 * 2"), 11)
        self.assertEqual(parse_and_eval("10 - 2 * 3 + 4"), 8)
        self.assertEqual(parse_and_eval("100 / 5 + 3 * 4 - 2"), 30)

if __name__ == '__main__':
    unittest.main()
