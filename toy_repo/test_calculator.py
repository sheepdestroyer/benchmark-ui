import unittest
from calculator import parse_and_eval

class TestCalculator(unittest.TestCase):
    def test_simple_addition(self):
        self.assertEqual(parse_and_eval("3 + 5"), 8)

    def test_order_of_operations(self):
        # This test will fail due to the order of operations bug
        self.assertEqual(parse_and_eval("3 + 4 * 2"), 11)
        self.assertEqual(parse_and_eval("10 - 2 * 3 + 4"), 8)

if __name__ == '__main__':
    unittest.main()
