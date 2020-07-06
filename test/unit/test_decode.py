import numpy as np
import unittest

from taiyaki import decode

import torch

try:
    import taiyaki.cupy_extensions.flipflop as cuff
    _cupy_is_available = torch.cuda.is_available()
except ImportError:
    _cupy_is_available = False


class TestFlipFlopDecode(unittest.TestCase):

    def setUp(self):
        self.scores = np.array([
            # BA step (can't start in flop!)
            [[0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],  # Aa step
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],  # aa stay
            [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],  # aB step
            [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],  # BB stay
            [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # BA step
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # AA stay
        ], dtype='f4')
        self.alphabet = 'AB'
        self.expected_path = np.array(
            [1, 0, 2, 2, 1, 1, 0, 0], dtype=int)[:, None]

    def assertArrayEqual(self, a, b):
        """ Test whether two :class:`ndarray` have the same shape and are equal
        in all elements

        Args:
            a (:class:`ndarray`): first array.
            b (:class:`ndarray`): second array.

        """
        self.assertEqual(a.shape, b.shape,
                         msg='Array shape mismatch: {} != {}\n'.format(a.shape, b.shape))
        self.assertTrue(np.allclose(a, b),
                        msg='Array element mismatch: {} != {}\n'.format(a, b))

    def test_cpu_decoding(self):
        """ Test CPU Viterbi decoding of flip-flop
        """
        _, _, path = decode.flipflop_viterbi(torch.tensor(self.scores))
        path = path.numpy()
        self.assertArrayEqual(path, self.expected_path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_gpu_decoding_no_cupy(self):
        """ Test GPU Viterbi decoding of flip-flop (if available)
        """
        _, _, path = decode.flipflop_viterbi(torch.tensor(self.scores, device=0),
                                             _never_use_cupy=True)
        path = path.cpu().numpy()
        self.assertArrayEqual(path, self.expected_path)

    @unittest.skipIf(not _cupy_is_available, "Cupy is not installed")
    def test_gpu_decoding_with_cupy(self):
        """ Test GPU Viterbi decoding of flip-flop using cupy routines
        (if available)
        """
        _, _, path = decode.flipflop_viterbi(
            torch.tensor(self.scores, device=0))
        path = path.cpu().numpy()
        self.assertArrayEqual(path, self.expected_path)

    def test_cpu_make_trans_no_grad(self):
        """ Test making transition scores when input does not require gradients
        """
        scores = torch.tensor(self.scores, requires_grad=False)
        trans = decode.flipflop_make_trans(scores)

    def test_cpu_make_trans_no_grad_non_leaf(self):
        """ Test making transition scores when input does not require gradients
        """
        scores = torch.tensor(self.scores, requires_grad=False)
        trans = decode.flipflop_make_trans(1.0 * scores)

    def test_cpu_make_trans_with_grad(self):
        """ Test making transition scores when input does require gradients
        """
        scores = torch.tensor(self.scores, requires_grad=True)
        trans = decode.flipflop_make_trans(scores)

    def test_cpu_make_trans_with_grad_non_leaf(self):
        """ Test making transition scores when input does require gradients
        """
        scores = torch.tensor(self.scores, requires_grad=True)
        trans = decode.flipflop_make_trans(1.0 * scores)

    def test_cpu_make_trans_with_grad_non_leaf_no_grad(self):
        """ Test making transition scores, complex case
        """
        scores = torch.tensor(self.scores, requires_grad=True)
        with torch.no_grad():
            trans = decode.flipflop_make_trans(1.0 * scores)

    @unittest.skipIf(not _cupy_is_available, "Cupy is not installed")
    def test_cupy_equals_torch_make_trans(self):
        """ Test that cupy and torch routines to calculate transition scores
        agree.
        """
        trans_torch = decode.flipflop_make_trans(torch.tensor(self.scores, device=0),
                                                 _never_use_cupy=True)
        trans_cupy = decode.flipflop_make_trans(
            torch.tensor(self.scores, device=0))
        self.assertArrayEqual(trans_torch.cpu().numpy(),
                              trans_cupy.cpu().numpy())
