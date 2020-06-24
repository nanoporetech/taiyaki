""" Functions to help out with calculation and calibration of qscores for basecalls
"""
import numpy as np
import torch
from taiyaki import flipflopfings
from taiyaki.constants import SMALL_VAL


def qchar_from_qscore(score, zerochar=33):
    """Return ASCII character encoding q score from score

    :param score   : floating point number: error prob = 10^(-score/10)
                     (or list or np array of numbers)
    :param zerochar: ASCII code for character encoding probability 1 (score 0)

    :returns: char representing quality score, or str if input is vector

    :note:
    With default zerochar=33, we use ASCII 33(!) for 0, 34(") for 1 and so on,
    rounding score to nearest int."""
    asciicodes = (np.array(score) + zerochar + 0.5).astype(np.int8)
    return asciicodes.tostring().decode('ascii')


def qscore_from_errprob(errprob):
    """Return q score from probability of error

    :param errprob : probability of error, may be scalar or numpy array

    :returns: q = -10 log_10(errprob)
    """
    return -10.0 * np.log10(errprob)


def qchar_from_errprob(errprob, qscore_scale, qscore_offset):
    """Return character representing quality score from errorprob.

    :param errprob : probability of error, or np array of probs
    :param qscore_scale: qscore <-- qscore * qscore_scale + qscore_offset
                         before coding as char
    :param qscore_offset: see qscore_scale above

    :returns: char representing quality score, or str of chars
    """
    qscore = qscore_scale * qscore_from_errprob(errprob) + qscore_offset
    return qchar_from_qscore(qscore)


def transitions_into_base(b, nbases, device):
    """Return pytorch long tensor of all transition-matrix indices for
    all transitions into base (flip or flop).

    :param b: base (integer in range 0 to (nbases-1) )
    :param nbases: number of bases (4 for ACGT)
    :param device: what pytorch device to use

    :returns: 1D pytorch long tensor of indices (in range 0 to 39) for ACGT.

    :note: all transitions, including those where no base is emitted, are
           included.
    """
    # Transition A to b_flip
    colstart = nbases * 2 * b
    # All transitions into b_flip
    toflip = torch.arange(colstart, colstart + nbases * 2,
                          dtype=torch.long, device=device)
    # Transition b_flip to b_flop
    fliptoflop = 2 * nbases * nbases + b
    # Tensor containing b_flip to b_flop and b_flop to b_flop
    toflop = torch.tensor([fliptoflop, fliptoflop + nbases],
                          dtype=torch.long, device=device)
    return torch.cat((toflip, toflop))


def errprobs_from_trans(trans, path):
    """Calculate error probs from (batch of) posterior trans weights and path

    This is done by:

      sum(trans posteriors for all transitions into base at path[b] at block b)
    p=-------------------------------------------------------------------------
        sum(trans posteriors for all transitions into any base at block b)

    errorprob = 1-p

    :param trans: pytorch float tensor of
                  shape (nblocks x batchsize x nstates)
                  where nstates = 40 for 4-base models
                  containing posterior transition weights (not logs!)
    :param path : pytorch long tensor of shape (nblocks+1) x batchsize
                  containing flip-flop states (integers 0-7 for 4-base models).
                  The transition that goes with trans[n,bn,:] is the one from
                  path[n,bn] to path[n+1,bn]

    :returns: errprobs, pytorch float tensor of shape (nblocks+1) x batchsize
                  containing errorprob for
                  each element of the path, and -1.0 in row 0

    :note: errprobs[0:,:,:]=-1.0, but this doesn't matter since these
           probabilities are removed later on in the pipeline. The output
           matrix must be the same shape as the path in order to be
           fed into the stitching function.

    """
    nblocks, batchsize, flip_flop_transitions = trans.shape
    nbases = flipflopfings.nbase_flipflop(flip_flop_transitions)
    # baseprobs will contain total probability for emission of each base
    # at each block normalised by prob of emitting any base.
    baseprobs = torch.zeros((nblocks, batchsize, nbases), dtype=torch.float,
                            device=trans.device)
    # Calculate total probability of transition into base b at block n
    for destbase in range(nbases):
        t = transitions_into_base(destbase, nbases, device=trans.device)
        m = torch.zeros(flip_flop_transitions, dtype=torch.float,
                        device=trans.device)
        m[t] = 1.0
        baseprobs[:, :, destbase] = torch.matmul(trans, m)

    # Normalise
    baseprobs = baseprobs / (baseprobs.sum(dim=2, keepdim=True) + SMALL_VAL)

    # Calculate matrix p (see docstring)
    p = torch.empty_like(path, dtype=torch.float)
    # baseprobs is nblocks x batchsize x nbases, path is (nblocks+1) x
    # batchsize
    ix = path[1:].unsqueeze(2) % nbases
    p[1:] = torch.gather(baseprobs, 2, ix).squeeze(2)
    # errprob at block 0 set to -1
    p[0] = 2.0
    return 1.0 - p


def path_errprobs_to_qstring(errprobs, path, qscore_scale, qscore_offset):
    """Make qscore string from error probs, ignoring stays

    :param errprobs: 1D pytorch float tensor or numpy float vector containing
                    error probabilities for each element of a path
    :param path: 1D pytorch long tensor or numpy int array containing flip-flop
                    states for each block, same length as errprobs
    :param qscore_scale: qscore <-- qscore * qscore_scale + qscore_offset
                         before coding as chars
    :param qscore_offset: see qscore_scale above

    :returns: python str containing qscores encoded as ASCII characters

    :note: Elements of the path where no base is emitted are not included
           in the qstring, and the source base for the first transition is
           also not included.
           So the qstring is the same length as the basecall (provided we
           don't include the source base for the first transition in the
           basecall)
    """
    filtered_probs = errprobs[1:][path[1:] != path[:-1]]
    if type(filtered_probs) == torch.Tensor:
        filtered_probs = filtered_probs.detach().cpu().numpy()
    return qchar_from_errprob(filtered_probs, qscore_scale, qscore_offset)
