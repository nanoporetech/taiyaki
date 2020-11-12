"""Tests for cmdards module"""
import argparse
import sys
import unittest
from taiyaki import cmdargs


class CmdArgsTest(unittest.TestCase):
    """Test command args functionality"""
    @classmethod
    def setUpClass(self):
        self.EPS = sys.float_info.epsilon

    def test_positive_valid_float_values(self):
        """Test that valid floats get through positive arg check."""
        f = cmdargs.Positive(float)
        for x in [1e-30, self.EPS, 1e-5, 1.0, 1e5, 1e30]:
            self.assertAlmostEqual(x, f(x))

    def test_positive_invalid_float_values(self):
        """Test that invalid floats don't get through positive arg check."""
        f = cmdargs.Positive(float)
        for x in [-1.0, -self.EPS, -1e-5, 0.0]:
            with self.assertRaises(argparse.ArgumentTypeError):
                f(x)

    def test_positive_valid_int_values(self):
        """Test that valid ints get through positive arg check."""
        f = cmdargs.Positive(int)
        for x in [1, 10, 10000]:
            self.assertAlmostEqual(x, f(x))

    def test_positive_invalid_int_values(self):
        """Test that invalid ints don't get through positive arg check."""
        f = cmdargs.Positive(int)
        for x in [-1, 0]:
            with self.assertRaises(argparse.ArgumentTypeError):
                f(x)

    def test_nonnegative_valid_float_values(self):
        """Test that valid floats get through non-negative arg check."""
        f = cmdargs.NonNegative(float)
        for x in [1e-30, self.EPS, 1e-5, 0.0, 1.0, 1e5, 1e30]:
            self.assertAlmostEqual(x, f(x))

    def test_nonnegative_invalid_float_values(self):
        """Test that invalid floats don't get through non-negative arg
        check."""
        f = cmdargs.NonNegative(float)
        for x in [-1.0, -self.EPS, -1e-5]:
            with self.assertRaises(argparse.ArgumentTypeError):
                f(x)

    def test_nonegative_valid_int_values(self):
        """Test that valid ints get through non-negative arg check."""
        f = cmdargs.NonNegative(int)
        for x in [0, 1, 10, 10000]:
            self.assertAlmostEqual(x, f(x))

    def test_nonegative_invalid_int_values(self):
        """Test that invalid ints don't get through non-negative arg check."""
        f = cmdargs.NonNegative(int)
        for x in [-1, -10]:
            with self.assertRaises(argparse.ArgumentTypeError):
                f(x)

    def test_proportion_valid_float_values(self):
        """Test that valid floats get through proportion (0-1) arg check."""
        f = cmdargs.proportion
        for x in [1e-30, self.EPS, 1e-5, 0.0, 1.0, 1.0 - 1e-5, 1.0 - self.EPS,
                  1.0 - 1e-30]:
            self.assertAlmostEqual(x, f(x))

    def test_proportion_invalid_float_values(self):
        """Test that invalid floats don't get through proportion (0-1) arg
        check."""
        f = cmdargs.proportion
        for x in [-1e-30, -self.EPS, -1e-5, 1.0 + 1e-5, 1.0 + self.EPS]:
            with self.assertRaises(argparse.ArgumentTypeError):
                f(x)

    def test_bounded_valid_int_values(self):
        """Test that valid ints get through bounded arg check."""
        f = cmdargs.Bounded(int, 0, 10)
        for x in range(0, 11):
            self.assertEqual(x, f(x))

    def test_bounded_invalid_int_values(self):
        """Test that invalid ints don't get through bounded arg check."""
        f = cmdargs.Bounded(int, 0, 10)
        for x in [-2, -1, 11, 12]:
            with self.assertRaises(argparse.ArgumentTypeError):
                f(x)

    def test_device_action_conversions(self):
        """Test to ensure DeviceAction accurately processes device strings."""
        parser = argparse.ArgumentParser()
        parser.add_argument('device', action=cmdargs.DeviceAction)
        self.assertEqual(2, parser.parse_args(['2']).device)
        self.assertEqual(2, parser.parse_args(['cuda2']).device)
