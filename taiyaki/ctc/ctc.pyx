cimport libctc
import cython
import numpy as np
cimport numpy as np

import torch


@cython.boundscheck(False)
@cython.wraparound(False)
def crf_flipflop_cost(np.ndarray[np.float32_t, ndim=3, mode="c"] logprob,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqs,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
                      sharpfact):
    """
    :param logprob: Tensor containing log probabilities
    :param seqs: Vector containing flip-flop coded sequences (see flipflopfings.flip_flop_code()), concatenated
    :param seqlen: Length of each sequence
    """
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]
    assert nstate == 40, "Number of states is {} not 40 as expected".format(nstate)

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs = np.zeros((nbatch,), dtype=np.float32)
    libctc.crf_flipflop_cost(&logprob[0, 0, 0], nstate, nblk, nbatch, &seqs[0],
                             &seqlen[0], sharpfact, &costs[0])
    assert np.all(costs <= 0.), "Error -- costs must be negative, got {}".format(costs)
    return -costs / nblk


@cython.boundscheck(False)
@cython.wraparound(False)
def crf_flipflop_grad(np.ndarray[np.float32_t, ndim=3, mode="c"] logprob,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqs,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
                      sharpfact):
    """
    :param logprob: Tensor containing log probabilities
    :param seqs: Vector containing flip-flop coded sequences (see flipflopfings.flip_flop_code()), concatenated
    :param seqlen: Length of each sequence
    """
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]
    assert nstate == 40, "Number of states is {} not 40 as expected".format(nstate)

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs = np.zeros((nbatch,), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] grads = np.zeros_like(logprob, dtype=np.float32)
    libctc.crf_flipflop_grad(&logprob[0, 0, 0], nstate, nblk, nbatch, &seqs[0],
                             &seqlen[0], sharpfact, &costs[0], &grads[0, 0, 0])
    return -costs / nblk, -grads / nblk


class FlipFlopCRF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logprob, seqs, seqlen, sharpfact):
        lp = logprob.detach().cpu().numpy().astype(np.float32)
        seqs = seqs.cpu().numpy().astype(np.int32)
        seqlen = seqlen.cpu().numpy().astype(np.int32)
        sharpfact = float(sharpfact)
        cost, grads = crf_flipflop_grad(lp, seqs, seqlen, sharpfact)
        ctx.save_for_backward(torch.tensor(grads, device=logprob.device))
        return torch.tensor(cost, device=logprob.device)

    @staticmethod
    def backward(ctx, output_grads):
        grads, = ctx.saved_tensors
        output_grads = output_grads.unsqueeze(1)
        return grads * output_grads, None, None, None


crf_flipflop_loss = FlipFlopCRF.apply
