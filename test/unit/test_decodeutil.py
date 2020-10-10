import unittest

import numpy as np
import torch

from taiyaki import decodeutil
from taiyaki.flipflopfings import nbase_flipflop
from taiyaki.layers import log_partition_flipflop
from taiyaki.maths import logsumexp


class TestDecodeutilFlipflop(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(0xdeadbeef)
        self.weights = np.random.randn(12, 40).astype('f4')
        self.expt_score = 27.16876983642578
        wt = torch.tensor(self.weights).unsqueeze(1)
        self.tensor_score = float(log_partition_flipflop(wt))

    def test_fwd_equals_global_norm(self):
        nbase = int(nbase_flipflop(self.weights.shape[1]))
        nstate = nbase + nbase
        init = np.zeros(nstate, dtype='f4')
        init[nbase:] = -50000
        fwd, f2 = decodeutil.forward(self.weights, init=init)
        fwd_score = float(logsumexp(fwd[-1], axis=0))
        print(f2, fwd_score, self.tensor_score)
        print(decodeutil.forward(self.weights, init=None)[1])
        self.assertAlmostEqual(fwd_score, self.tensor_score, places=5)

    def test_bwd_equals_global_norm(self):
        nbase = int(nbase_flipflop(self.weights.shape[1]))
        nstate = nbase + nbase
        bwd, _ = decodeutil.backward(self.weights)
        bwd_score = float(logsumexp(bwd[0, :nbase], axis=0))
        self.assertAlmostEqual(bwd_score, self.tensor_score, places=5)

    def test_fwd_score_equals_bwd_score(self):
        bwd, _ = decodeutil.backward(self.weights)
        bwd_score = float(logsumexp(bwd[0], axis=0))
        fwd, _ = decodeutil.forward(self.weights)
        fwd_score = float(logsumexp(fwd[-1], axis=0))
        print(fwd_score, bwd_score, logsumexp(
            self.weights.reshape(-1), axis=0))
        print(fwd, bwd)
        self.assertAlmostEqual(bwd_score, self.expt_score, places=5)
        self.assertAlmostEqual(fwd_score, self.expt_score, places=5)
        self.assertAlmostEqual(fwd_score, bwd_score, places=5)

    def test_score_consistent(self):
        nt, ns = self.weights.shape
        fwd, _ = decodeutil.forward(self.weights)
        bwd, _ = decodeutil.backward(self.weights)
        self.assertEqual(fwd.shape, bwd.shape)

        score = logsumexp(fwd + bwd, axis=1)
        print(score.mean())
        self.assertAlmostEqual(float(score.mean()), self.expt_score, places=5)
        score_range = float(score.max() - score.min())
        print(score_range)
        self.assertAlmostEqual(score_range, 0.0, places=5)
