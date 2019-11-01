import numpy as np
import torch

from taiyaki import flipflopfings
from taiyaki.constants import LARGE_VAL
from taiyaki.layers import log_partition_flipflop


try:
    import taiyaki.cupy_extensions.flipflop as cuff
    _cupy_is_available = True
except ImportError:
    _cupy_is_available = False


def flipflop_viterbi(scores, _never_use_cupy=False):
    """ Find highest scoring flipflop paths for a batch of score matrices

    The input scores should be on a log scale, i.e. the score of a path
    is determined by summing the scores of the individual transitions.

    :param scores: batch of score matrices with dimensions [T, batch size, S]
         where T is the number of blocks (time axis) and S is the number of
         distict flipflop transitions. For 4 bases S = 40, and in general
         S = 2 * nbase * (nbase + 1)
    :param _never_use_cupy: this method delegates to cupy implementation if
         possible, unless _never_use_cupy=True, defaults to False
    """
    use_cupy = all([
        _cupy_is_available,
        scores.device.type == 'cuda',
        not _never_use_cupy,
    ])
    if use_cupy:
        return cuff.flipflop_viterbi(scores)
    else:
        return _flipflop_viterbi(scores)


def flipflop_make_trans(scores, _never_use_cupy=False):
    """ Calculates posterior probabilities (not logs!) from raw model output

    The input consists of globally normalised transition scores
    for a flipflop CRF. The output consists of posterior
    probabilities for each transition.

    It can be verified that this is equivalent to the derivative
    of the log-partition function with respect to the raw scores.

    :param scores: batch of score matrices with dimensions [T, batch size, S]
         where T is the number of blocks (time axis) and S is the number of
         distict flipflop transitions. For 4 bases S = 40, and in general
         S = 2 * nbase * (nbase + 1)
    :param _never_use_cupy: this method delegates to cupy implementation if
         possible, unless _never_use_cupy=True, defaults to False

    :return: pytorch float tensor of shape (T x batch size x S) containing
             posterior transition probabilities (not logs!)
    """
    use_cupy = all([
        _cupy_is_available,
        scores.device.type == 'cuda',
        not _never_use_cupy,
    ])
    if use_cupy:
        return cuff.flipflop_make_trans(scores)[0].softmax(2)
    else:
        scores_requires_grad = scores.requires_grad
        scores.requires_grad_()
        with torch.enable_grad():
            logZ = log_partition_flipflop(scores).sum()
        trans = torch.autograd.grad(logZ, scores, create_graph=True)[0]
        scores.requires_grad = scores_requires_grad
        return trans.detach()


@torch.no_grad()
def _flipflop_viterbi(scores):
    """ Find highest scoring flipflop paths for a batch of score matrices

    This is an idiomatic pytorch implementation.

    :param scores: batch of score matrices with dimensions [T, batch size, S] where
         T is the number of blocks (time axis) and S is the number of distict
         flipflop transitions. For 4 bases S = 40, and in general S = 2 * nbase * (nbase + 1)
    """
    T, N, S = scores.shape
    nbase = flipflopfings.nbase_flipflop(S)

    fwd = torch.zeros(T + 1, N, 2 * nbase, device=scores.device, dtype=scores.dtype)
    fwd[0, :, nbase:] = -LARGE_VAL
    traceback = torch.zeros(T, N, 2 * nbase, device=scores.device, dtype=torch.long)

    for t in range(T):
        to_flip = scores[t, :, :S - 2 * nbase].reshape((N, nbase, 2 * nbase))
        fwd[t + 1, :, :nbase], traceback[t, :, :nbase] = (fwd[t].unsqueeze(1) + to_flip).max(2)
        fwd[t + 1, :, nbase:], tb_flop = (fwd[t] + scores[t, :, -2 * nbase:]).reshape((N, 2, nbase)).max(1)
        traceback[t, :, nbase:] = nbase * tb_flop + torch.arange(nbase, device=traceback.device, dtype=traceback.dtype)

    path = torch.zeros(T + 1, N, device=scores.device, dtype=torch.long)

    path[T] = fwd[T].argmax(1)
    ix = torch.arange(N, device=traceback.device, dtype=torch.long)
    for t in range(T - 1, -1, -1):
        path[t] = traceback[t, ix, path[t + 1]]

    return fwd, traceback, path
