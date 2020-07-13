import torch
import unittest
import numpy as np

from taiyaki.constants import SMALL_VAL
from taiyaki import ctc, flipflopfings, layers

# Test ctc loss and gradient calculations.
# To run as a  single test, in taiyaki dir and in venv do
# pytest test/unit/test_ctc_loss.py


def flipflop_transitioncode(fromstate, tostate, nbases):
    """Returns location in the flip-flop transition matrix of a transition
    from fromstate to tostate"""
    if tostate < nbases:  # if we go to a flip
        return tostate * 2 * nbases + fromstate
    else:
        return 2 * nbases * nbases + fromstate


class CtcGradientTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.nbases = 4
        self.batchsize = 1
        self.seqlen = 3
        self.nblocks = 4
        self.sharpen = 1.0
        self.nflipflop_transitions = flipflopfings.nstate_flipflop(
            self.nbases)  # 40 for ACGT
        self.dx_size = 0.001  # Size of small changes for gradient check
        self.grad_dp = 5  # Number of decimal places for gradient check

        # Sequence ACC. Flip-flop coded ACc or 015
        # 'sequences' rather than 'sequence' because
        # we could have more than one sequence packed together
        self.sequences = {'015': torch.tensor([0, 1, 5]),
                          '237': torch.tensor([2, 3, 7]),
                          '510': torch.tensor([5, 1, 0])}  # prob 0 according to outputs
        self.seqlens = torch.tensor([self.seqlen])
        # Network outputs are weights for flip-flop transitions
        # Define some example paths and assign weights to them.
        # These will be used to define the example output matrix
        #
        paths = {}
        weights = {}

        paths['015'] = [0,    0,    1,    5,    5]
        weights['015'] = [1.0,   1.0,  0.5,  1.0]

        paths['237'] = [2,    2,    3,    7,    7]
        weights['237'] = [1.0, 0.5, 1.0, 1.0]

        weights['510'] = [0.0]  # No weight for this sequence/path

        self.path_probabilities = {k: np.prod(v) for k, v in weights.items()}

        # Normalise path probabilities
        psum = sum(self.path_probabilities.values())
        self.path_probabilities = {k: v / psum
                                   for k, v in self.path_probabilities.items()}

        # Make output (transition weight) matrix with these path probs
        self.outputs = torch.zeros(self.nblocks, self.batchsize,
                                   self.nflipflop_transitions, dtype=torch.float)
        for k in paths.keys():
            for block in range(self.nblocks):
                transcode = flipflop_transitioncode(
                    paths[k][block], paths[k][block + 1], self.nbases)
                self.outputs[block, 0, transcode] = weights[k][block]

        # Log and normalise output (transition weight) matrix
        self.outputs = torch.log(self.outputs + SMALL_VAL)
        self.outputs = layers.global_norm_flipflop(self.outputs)

    def test_loss(self):
        """Test that loss = exp(-sequence probability)"""
        self.outputs.requires_grad = False

        # First check normalisation of output matrix
        logpart = float(layers.log_partition_flipflop(self.outputs))
        # Print output will appear only if test fails
        print("Check normalisation: exp(log_partition_flipflop) =", end="")
        print(" {:3.4f}, log={:3.4f}".format(np.exp(logpart), logpart))
        self.assertAlmostEqual(logpart, 0.0)

        # Now check probabilities for three sequences
        for sequence_name, sequence in self.sequences.items():
            sequence_prob = self.path_probabilities[sequence_name]
            print("Sequence {} P={:3.3f} ".format(sequence_name, sequence_prob),
                  end="")
            lossvector = ctc.crf_flipflop_loss(self.outputs, sequence,
                                               self.seqlens,
                                               self.sharpen)
            sequence_prob_from_ctc = float(
                torch.exp(-lossvector * self.nblocks))
            print("Pctc={:3.4f}, loss={:3.4f}".format(sequence_prob_from_ctc,
                                                      float(lossvector)))
            self.assertAlmostEqual(sequence_prob, sequence_prob_from_ctc)

    def test_grad(self):
        """Check that gradient accurately gives result of small change in
        output (transition weight) matrix"""
        self.outputs.requires_grad = True
        for ks, seq in self.sequences.items():
            print("Sequence", ks, end=": ")
            lossvector = ctc.crf_flipflop_loss(self.outputs, seq,
                                               self.seqlens,
                                               self.sharpen)
            print("P={:3.4f}, loss={:3.4f}".format(
                float(torch.exp(-lossvector * self.nblocks)),
                float(lossvector)))
            loss = torch.sum(lossvector)
            if self.outputs.grad is not None:
                self.outputs.grad.data.zero_()
            loss.backward()
            # Make random small change in outputs
            small_change = torch.randn_like(self.outputs) * self.dx_size
            outputs2 = self.outputs.detach() + small_change
            lossvector2 = ctc.crf_flipflop_loss(outputs2, seq,
                                                self.seqlens,
                                                self.sharpen)
            loss2 = torch.sum(lossvector2)
            loss_change = float(loss2 - loss)
            loss_change_from_grad = float(
                torch.sum(small_change * self.outputs.grad))
            print("    Change in loss = {:3.7f} ".format(loss_change), end="")
            print(", est from grad = {:3.7f}".format(loss_change_from_grad))
            self.assertAlmostEqual(loss_change / float(loss),
                                   loss_change_from_grad / float(loss),
                                   places=self.grad_dp)
