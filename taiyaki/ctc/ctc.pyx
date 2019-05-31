cimport libctc
import cython
import numpy as np
cimport numpy as np

import torch


def nstate_to_nbase(size_t nstate):
    """Convert number of flip-flop states to number of bases and check for valid states number.
    """
    cdef np.float32_t nbase_f = np.sqrt(0.25 + (0.5 * nstate)) - 0.5
    assert np.mod(nbase_f, 1) == 0, (
        'Number of states not valid for flip-flop model. ' +
        'nstates: {}\tconverted nbases: {}').format(nstate, nbase_f)
    cdef size_t nbase = <size_t>nbase_f
    return nbase


##########################################
###### Standard flip-flop functions ######
##########################################


@cython.boundscheck(False)
@cython.wraparound(False)
def crf_flipflop_cost(np.ndarray[np.float32_t, ndim=3, mode="c"] logprob,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqs,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
                      sharpfact):
    """
    :param logprob: Tensor containing log probabilities
    :param seqs: Vector containing flip-flop coded sequences (see flipflopfings.flipflop_code()), concatenated
    :param seqlen: Length of each sequence
    """
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]
    # conversion checks that nstates converts to a valid number of bases
    nstate_to_nbase(nstate)

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs = np.zeros(
        (nbatch,), dtype=np.float32)
    libctc.crf_flipflop_cost(&logprob[0, 0, 0], nstate, nblk, nbatch, &seqs[0],
                             &seqlen[0], sharpfact, &costs[0])
    assert np.all(costs <= 0.), (
        "Error: costs must be negative, got {}").format(costs)
    return -costs / nblk


@cython.boundscheck(False)
@cython.wraparound(False)
def crf_flipflop_grad(np.ndarray[np.float32_t, ndim=3, mode="c"] logprob,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqs,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
                      sharpfact):
    """
    :param logprob: Tensor containing log probabilities
    :param seqs: Vector containing flip-flop coded sequences (see flipflopfings.flipflop_code()), concatenated
    :param seqlen: Length of each sequence
    """
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]
    # conversion checks that nstates converts to a valid number of bases
    nstate_to_nbase(nstate)

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs = np.zeros(
        (nbatch,), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] grads = np.zeros_like(
        logprob, dtype=np.float32)
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
        if logprob.requires_grad:
            cost, grads = crf_flipflop_grad(lp, seqs, seqlen, sharpfact)
            ctx.save_for_backward(torch.tensor(grads, device=logprob.device))
        else:
            cost = crf_flipflop_cost(lp, seqs, seqlen, sharpfact)
        return torch.tensor(cost, device=logprob.device)

    @staticmethod
    def backward(ctx, output_grads):
        grads, = ctx.saved_tensors
        output_grads = output_grads.unsqueeze(1)
        return grads * output_grads, None, None, None

crf_flipflop_loss = FlipFlopCRF.apply


##########################################################
###### Categorical modification flip-flop functions ######
##########################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def cat_mod_flipflop_cost(
        np.ndarray[np.float32_t, ndim=3, mode="c"] logprob,
        np.ndarray[np.int32_t, ndim=1, mode="c"] seqs,
        np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
        np.ndarray[np.int32_t, ndim=1, mode="c"] mod_cats,
        np.ndarray[np.int32_t, ndim=1, mode="c"] can_mods_offsets,
        np.ndarray[np.float32_t, ndim=1, mode="c"] mod_cat_weights,
        mod_weight, sharpfact):
    """
    :param logprob: Tensor containing log probabilities
    :param seqs: A vector containing sequences, concatenated
    :param mod_cats: A vector containing mod categories, concatenated
    :param seqlen: Length of each sequence
    """
    assert np.all(logprob[:,:,-can_mods_offsets[4]:] <= 0), (
        'Error: Some modified base log probs are positive.')
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]
    # conversion checks that nstates converts to a valid number of bases
    nstate_to_nbase(nstate - can_mods_offsets[4])

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs = np.zeros(
        (nbatch,), dtype=np.float32)
    libctc.cat_mod_flipflop_cost(
        &logprob[0, 0, 0], nstate, nblk, nbatch, &seqs[0], &seqlen[0],
        &mod_cats[0], &can_mods_offsets[0], &mod_cat_weights[0], mod_weight,
        sharpfact, &costs[0])
    assert np.all(costs <= 0.), ("Error: costs must be negative, " +
                                 "got {}").format(costs)
    return -costs / nblk


@cython.boundscheck(False)
@cython.wraparound(False)
def cat_mod_flipflop_grad(
        np.ndarray[np.float32_t, ndim=3, mode="c"] logprob,
        np.ndarray[np.int32_t, ndim=1, mode="c"] seqs,
        np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
        np.ndarray[np.int32_t, ndim=1, mode="c"] mod_cats,
        np.ndarray[np.int32_t, ndim=1, mode="c"] can_mods_offsets,
        np.ndarray[np.float32_t, ndim=1, mode="c"] mod_cat_weights,
        mod_weight, sharpfact):
    """
    :param logprob: Tensor containing log probabilities
    :param seqs: A vector containing sequences, concatenated
    :param mod_cats: A vector containing mod categories, concatenated
    :param seqlen: Length of each sequence
    """
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]
    # conversion checks that nstates converts to a valid number of bases
    nstate_to_nbase(nstate - can_mods_offsets[4])

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] grads = np.zeros_like(
        logprob, dtype=np.float32)
    libctc.cat_mod_flipflop_grad(
        &logprob[0, 0, 0], nstate, nblk, nbatch, &seqs[0], &seqlen[0],
        &mod_cats[0], &can_mods_offsets[0], &mod_cat_weights[0], mod_weight,
        sharpfact, &grads[0, 0, 0])
    return -grads / nblk


class CatModFlipFlop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logprob, seqs, seqlen, mod_cats, can_mods_offsets,
                mod_cat_weights, mod_weight, sharpfact):
        ctx.save_for_backward(
            logprob, seqs, seqlen, mod_cats, torch.tensor(can_mods_offsets),
            torch.tensor(mod_cat_weights), mod_weight, torch.tensor(sharpfact))
        lp = np.ascontiguousarray(
            logprob.detach().cpu().numpy().astype(np.float32))
        seqs = seqs.cpu().numpy().astype(np.int32)
        seqlen = seqlen.cpu().numpy().astype(np.int32)
        mod_cats = mod_cats.cpu().numpy().astype(np.int32)
        mod_weight, sharpfact = map(float, (mod_weight, sharpfact))
        cost = cat_mod_flipflop_cost(
            lp, seqs, seqlen, mod_cats, can_mods_offsets, mod_cat_weights,
            mod_weight, sharpfact)
        return torch.tensor(cost, device=logprob.device)

    @staticmethod
    def backward(ctx, output_grads):
        (logprob, seqs, seqlen, mod_cats, can_mods_offsets,
         mod_cat_weights, mod_weight, sharpfact) = ctx.saved_tensors
        lp = np.ascontiguousarray(
            logprob.detach().cpu().numpy().astype(np.float32))
        seqs = seqs.cpu().numpy().astype(np.int32)
        seqlen = seqlen.cpu().numpy().astype(np.int32)
        mod_cats = mod_cats.cpu().numpy().astype(np.int32)
        can_mods_offsets = can_mods_offsets.cpu().numpy().astype(np.int32)
        mod_cat_weights = mod_cat_weights.cpu().numpy().astype(np.float32)
        mod_weight, sharpfact = map(float, (mod_weight, sharpfact))
        grads = cat_mod_flipflop_grad(
            lp, seqs, seqlen, mod_cats, can_mods_offsets, mod_cat_weights,
            mod_weight, sharpfact)
        output_grads = output_grads.unsqueeze(1)
        return (torch.tensor(grads, device=logprob.device) * output_grads,
                None, None, None, None, None, None, None)

cat_mod_flipflop_loss = CatModFlipFlop.apply
