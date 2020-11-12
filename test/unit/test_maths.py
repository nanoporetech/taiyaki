import unittest
import numpy as np
from taiyaki import maths


class MathsTest(unittest.TestCase):
    """Test maths functionality"""
    @classmethod
    def setUpClass(self):
        print('* Maths routines')
        np.random.seed(0xdeadbeef)

    def test_004_med_mad(self):
        """Test to see if med_mad works with axis not set (so flattening)."""
        x = np.array(
            [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.0, 0.5, 0.5, 1.0]])
        factor = 1
        loc, scale = maths.med_mad(x, factor=factor)
        self.assertTrue(np.allclose(loc, 0.5))
        self.assertTrue(np.allclose(scale, 0))

    def test_005_med_mad_over_axis0(self):
        """Test to see if med_mad works when axis=0."""
        x = np.array(
            [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.5, 1.0, 0.5, 1.0]])
        factor = 1
        loc, scale = maths.med_mad(x, factor=factor, axis=0)
        expected_loc = [0.5, 0.5, 0.5, 1.0]
        expected_scale = [0, 0, 0, 0]
        self.assertTrue(np.allclose(loc, expected_loc))
        self.assertTrue(np.allclose(scale, expected_scale))

    def test_006_med_mad_over_axis1(self):
        """Test to see if med_mad works when axis=1."""
        x = np.array(
            [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.0, 0.5, 0.5, 1.0]])
        factor = 1
        loc, scale = maths.med_mad(x, factor=factor, axis=1)
        expected_loc = [0.5, 0.75, 0.5]
        expected_scale = [0, 0.25, 0.25]
        self.assertTrue(np.allclose(loc, expected_loc))
        self.assertTrue(np.allclose(scale, expected_scale))

    def test_007_mad_keepdims(self):
        """Test to see if mad works when keepdims set (so not flattened)."""
        x = np.zeros((5, 6, 7))
        self.assertTrue(np.allclose(maths.mad(x, axis=0, keepdims=True),
                                    np.zeros((1, 6, 7))))
        self.assertTrue(np.allclose(maths.mad(x, axis=1, keepdims=True),
                                    np.zeros((5, 1, 7))))
        self.assertTrue(np.allclose(maths.mad(x, axis=2, keepdims=True),
                                    np.zeros((5, 6, 1))))


if __name__ == '__main__':
    unittest.main()
