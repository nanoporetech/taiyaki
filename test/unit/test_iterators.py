import unittest
import numpy as np

from taiyaki import iterators


class IteratorsTest(unittest.TestCase):
    """Test methods in iterators module"""
    @classmethod
    def setUpClass(self):
        """No directory or file setup needed"""
        pass

    def f(self, L, x):
        """Apply centered_truncated_window iterator, convert output to list
        for use in testing"""
        return list(iterators.centered_truncated_window(L, x))

    def test_centered_truncated_window_even(self):
        """Test centered truncated window iterator on even length vector"""
        def f(L, x):
            """I think this definition must be superfluous? f is never used."""
            return list(iterators.centered_truncated_window(L, x))
        L = [1, 2, 3, 4]
        self.assertRaises(AssertionError, self.f, L, 0)
        self.assertEqual(self.f(L, 1), [(1,), (2,), (3,), (4,)])
        self.assertEqual(self.f(L, 2), [(1, 2), (2, 3), (3, 4), (4,)])
        self.assertEqual(self.f(L, 3), [(1, 2), (1, 2, 3), (2, 3, 4), (3, 4)])
        self.assertEqual(
            self.f(L, 4), [(1, 2, 3), (1, 2, 3, 4), (2, 3, 4), (3, 4)])

    def test_centered_truncated_window_odd(self):
        """Test centered truncated window iterator on odd length vector"""
        L = [1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(self.f(L, 6), [(1, 2, 3, 4), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6),
                                        (2, 3, 4, 5, 6, 7), (3, 4, 5, 6, 7), (4, 5, 6, 7), (5, 6, 7)])

if __name__ == '__main__':
    unittest.main()
