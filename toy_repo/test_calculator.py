import unittest
from calculator import parse_and_eval

class TestCalculator(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(parse_and_eval("3 + 5"), 8.0)
        self.assertEqual(parse_and_eval("0 + 0"), 0.0)
        self.assertEqual(parse_and_eval("10 + 20 + 30"), 60.0)

    def test_subtraction(self):
        self.assertEqual(parse_and_eval("10 - 4"), 6.0)
        self.assertEqual(parse_and_eval("5 - 10"), -5.0)
        self.assertEqual(parse_and_eval("20 - 5 - 3"), 12.0)

    def test_multiplication(self):
        self.assertEqual(parse_and_eval("4 * 5"), 20.0)
        self.assertEqual(parse_and_eval("3 * 0"), 0.0)
        self.assertEqual(parse_and_eval("2 * 3 * 4"), 24.0)

    def test_division(self):
        self.assertEqual(parse_and_eval("20 / 4"), 5.0)
        self.assertEqual(parse_and_eval("10 / 4"), 2.5)

    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            parse_and_eval("10 / 0")

    def test_unsupported_operator(self):
        with self.assertRaises(ValueError):
            parse_and_eval("10 % 3")

    def test_float_operations(self):
        self.assertAlmostEqual(parse_and_eval("3.5 + 2.1"), 5.6)
        self.assertAlmostEqual(parse_and_eval("7.5 / 2.5"), 3.0)
        self.assertAlmostEqual(parse_and_eval("1.2 * 3.4"), 4.08)

    def test_whitespace_and_empty(self):
        self.assertEqual(parse_and_eval(""), 0.0)
        self.assertEqual(parse_and_eval("   "), 0.0)
        self.assertEqual(parse_and_eval("  4  +  5  "), 9.0)

    def test_complex_expressions(self):
        self.assertEqual(parse_and_eval("3 + 4 * 2"), 11.0)
        self.assertEqual(parse_and_eval("10 - 2 * 3 + 4"), 8.0)
        self.assertEqual(parse_and_eval("100 / 5 + 3 * 4 - 2"), 30.0)

    def test_single_number(self):
        self.assertEqual(parse_and_eval("42"), 42.0)
        self.assertEqual(parse_and_eval("0"), 0.0)
        self.assertEqual(parse_and_eval("3.14159"), 3.14159)

    def test_unsupported_operators(self):
        with self.assertRaises(ValueError):
            parse_and_eval("3 % 2")
        with self.assertRaises(ValueError):
            parse_and_eval("5 ^ 2")
        with self.assertRaises(ValueError):
            parse_and_eval("4 & 2")

    def test_negative_numbers(self):
        self.assertEqual(parse_and_eval("-5 + 3"), -2.0)
        self.assertEqual(parse_and_eval("-5 - 3"), -8.0)
        self.assertEqual(parse_and_eval("-10"), -10.0)
        self.assertEqual(parse_and_eval("10 + -5"), 5.0)

    def test_malformed_inputs(self):
        # Empty inputs return 0.0
        self.assertEqual(parse_and_eval(""), 0.0)
        self.assertEqual(parse_and_eval("   "), 0.0)

        # Invalid syntax raises ValueError
        with self.assertRaises(ValueError):
            parse_and_eval("3 +")
        with self.assertRaises(ValueError):
            parse_and_eval("3 + + 4")
        with self.assertRaises(ValueError):
            parse_and_eval("invalid syntax")

        # Wrong types raise TypeError
        with self.assertRaises(TypeError):
            parse_and_eval(42)
        with self.assertRaises(TypeError):
            parse_and_eval(None)

if __name__ == '__main__':
    unittest.main()
