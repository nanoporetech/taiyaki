# cython: profile=False

cimport libctc
import cython
import numpy as np
cimport numpy as np

import torch

from taiyaki import flipflopfings


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
                      np.ndarray[np.uintp_t, ndim=1, mode="c"] moveidxs,
                      np.ndarray[np.uintp_t, ndim=1, mode="c"] stayidxs,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
                      pin=False):
    """

    Args:
        logprob (array [nblk x nbatch x nstate]):  Log scores.
        moveidxs (array [sum(seqlen) - nseq]):  Transition indicies for moves.
        stayidxs (array [sum(seqlen)]):  Transition indicies for stays.
        seqlen (array [nseq]): Length of sequences.
        pin (bool): Whether to pin memory of returned tensors

    Returns:
        array [nbatch]: Costs (scores) for nbatch elements.
    """
    assert np.all(np.isfinite(logprob)), "Input not finite"
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]
    # conversion checks that nstates converts to a valid number of bases
    nstate_to_nbase(nstate)

    costs = torch.zeros(nbatch, device='cpu', dtype=torch.float)
    if pin:
        costs = costs.pin_memory()

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs_np = costs.numpy()
    libctc.crf_flipflop_cost(&logprob[0, 0, 0], nstate, nblk, nbatch,
                             &moveidxs[0], &stayidxs[0], &seqlen[0],
                             &costs_np[0])
    assert np.all(np.isfinite(costs_np)), (
        "Error: all costs must be finite, got {}\n"
        "Try restarting from a checkpoint with a lower learning rate."
    ).format(costs_np)
    return -costs / nblk


@cython.boundscheck(False)
@cython.wraparound(False)
def crf_flipflop_grad(np.ndarray[np.float32_t, ndim=3, mode="c"] logprob,
                      np.ndarray[np.uintp_t, ndim=1, mode="c"] moveidxs,
                      np.ndarray[np.uintp_t, ndim=1, mode="c"] stayidxs,
                      np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
                      pin=False):
    """

    Args:
        logprob (array [nblk x nbatch x nstate]):  Log scores.
        moveidx (array [sum(seqlen) - nseq]):  Transition indicies for moves.
        stayidx (array [sum(seqlen)]):  Transition indicies for stays.
        seqlen (array [nseq]): Length of sequences.
        pin (bool): Whether to pin memory of returned tensors

    Returns:
        Tuple(array [nbatch], array[nblk x nbatch x nstate]):
    Costs (scores) for nbatch elements, and the gradient of that cost WRT
    elements of `logprob`.
    """
    assert np.all(np.isfinite(logprob)), "Input not finite"
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]
    # conversion checks that nstates converts to a valid number of bases
    nstate_to_nbase(nstate)

    costs = torch.zeros(nbatch, device='cpu', dtype=torch.float)
    grads = torch.zeros(nblk, nbatch, nstate, device='cpu', dtype=torch.float)
    if pin:
        costs = costs.pin_memory()
        grads = grads.pin_memory()

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs_np = costs.numpy()
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] grads_np = grads.numpy()
    libctc.crf_flipflop_grad(&logprob[0, 0, 0], nstate, nblk, nbatch,
                             &moveidxs[0], &stayidxs[0], &seqlen[0],
                             &costs_np[0], &grads_np[0, 0, 0])
    assert np.all(np.isfinite(costs_np)), (
        "Error: all costs must be finite, got {}.\n"
        "Try restarting from a checkpoint with a lower learning rate."
    ).format(costs_np)
    assert np.all(np.isfinite(grads_np)), ("Error: Gradients not finite.\n"
        "Try restarting from a checkpoint with a lower learning rate.")
    return -costs / nblk, -grads / nblk


class FlipFlopCRF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logprob, seqs, seqlen, sharpfact : float):
        lp = (sharpfact * logprob).detach().float().cpu().numpy()
        seqs = seqs.cpu().numpy().astype(np.int32)
        seqlen = seqlen.cpu().numpy().astype(np.int32)

        ntrans = lp.shape[2]
        nbase = flipflopfings.nbase_flipflop(ntrans)

        #  Calculate indices -- seqlen[0:-1] ensures input to split is array
        moveidxs = np.concatenate([
            flipflopfings.move_indices(seq, nbase)
            for seq in np.split(seqs, np.cumsum(seqlen[:-1]))]).astype(np.uintp)
        stayidxs = np.concatenate([
            flipflopfings.stay_indices(seq, nbase)
            for seq in np.split(seqs, np.cumsum(seqlen[:-1]))]).astype(np.uintp)
        assert np.all(np.logical_and(moveidxs >=0, moveidxs < ntrans))
        assert np.all(np.logical_and(stayidxs >=0, stayidxs < ntrans))

        if logprob.requires_grad:
            cost, grads = crf_flipflop_grad(lp, moveidxs, stayidxs, seqlen,
                                            pin=torch.cuda.is_available())
            grad_gpu = torch.empty_like(grads, device=logprob.device)
            ctx.save_for_backward(grad_gpu)
            grad_gpu.copy_(grads, non_blocking=True)
        else:
            cost = crf_flipflop_cost(lp, moveidxs, stayidxs, seqlen,
                                     pin=torch.cuda.is_available())
        return (cost / sharpfact).to(logprob.device, non_blocking=True)

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
        np.ndarray[np.uintp_t, ndim=1, mode="c"] moveidxs,
        np.ndarray[np.uintp_t, ndim=1, mode="c"] stayidxs,
        np.ndarray[np.uintp_t, ndim=1, mode="c"] modmoveidxs,
        np.ndarray[np.float32_t, ndim=1, mode="c"] modmovefacts,
        np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
        pin=False):
    """

    Args:
        logprob (array [nblk x nbatch x nstate]):  Log scores.
        moveidxs (array [sum(seqlen) - nseq]):  Transition indicies for moves.
        stayidxs (array [sum(seqlen)]):  Transition indicies for stays.
        modmoveidxs (array [sum(seqlen) - nseq]):  Transition indicies for
            modbase moves.
        modmovefacts (array [sum(seqlen) - nseq]):  Transition indicies for
            modbase moves.
        seqlen (array [nseq]): Length of sequences.
        pin (bool): Whether to pin memory of returned tensors

    Returns:
        array [nbatch]: Costs (scores) for nbatch elements.
    """
    assert np.all(np.isfinite(logprob)), "Input not finite"
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]

    costs = torch.zeros(nbatch, device='cpu', dtype=torch.float)
    if pin:
        costs = costs.pin_memory()

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs_np = costs.numpy()
    libctc.cat_mod_flipflop_cost(
        &logprob[0, 0, 0], nstate, nblk, nbatch, &moveidxs[0], &stayidxs[0],
        &modmoveidxs[0], &modmovefacts[0], &seqlen[0], &costs_np[0])
    assert np.all(np.isfinite(costs_np)), (
        "Error: all costs must be finite, got {}.\n"
        "Try restarting from a checkpoint with a lower learning rate."
    ).format(costs_np)
    return -costs / nblk


@cython.boundscheck(False)
@cython.wraparound(False)
def cat_mod_flipflop_grad(
        np.ndarray[np.float32_t, ndim=3, mode="c"] logprob,
        np.ndarray[np.uintp_t, ndim=1, mode="c"] moveidxs,
        np.ndarray[np.uintp_t, ndim=1, mode="c"] stayidxs,
        np.ndarray[np.uintp_t, ndim=1, mode="c"] modmoveidxs,
        np.ndarray[np.float32_t, ndim=1, mode="c"] modmovefacts,
        np.ndarray[np.int32_t, ndim=1, mode="c"] seqlen,
        pin=False):
    """

    Args:
        logprob (array [nblk x nbatch x nstate]):  Log scores.
        moveidxs (array [sum(seqlen) - nseq]):  Transition indicies for moves.
        stayidxs (array [sum(seqlen)]):  Transition indicies for stays.
        modmoveidxs (array [sum(seqlen) - nseq]):  Transition indicies for
            modbase moves.
        modmovefacts (array [sum(seqlen) - nseq]):  Transition indicies for
            modbase moves.
        seqlen (array [nseq]): Length of sequences.
        pin (bool): Whether to pin memory of returned tensors

    Returns:
        Tuple(array [nbatch], array[nblk x nbatch x nstate]):
    Costs (scores) for nbatch elements, and the gradient of that cost WRT
    elements of `logprob`.
    """
    assert np.all(np.isfinite(logprob)), "Input not finite"
    cdef size_t nblk, nbatch, nstate
    nblk, nbatch, nstate = logprob.shape[0], logprob.shape[1], logprob.shape[2]

    costs = torch.zeros(nbatch, device='cpu', dtype=torch.float)
    grads = torch.zeros(nblk, nbatch, nstate, device='cpu', dtype=torch.float)
    if pin:
        costs = costs.pin_memory()
        grads = grads.pin_memory()

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs_np = costs.numpy()
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] grads_np = grads.numpy()
    libctc.cat_mod_flipflop_grad(
        &logprob[0, 0, 0], nstate, nblk, nbatch, &moveidxs[0], &stayidxs[0],
        &modmoveidxs[0], &modmovefacts[0], &seqlen[0], &costs_np[0],
        &grads_np[0, 0, 0])
    assert np.all(np.isfinite(costs_np)), (
        "Error: all costs must be finite, got {}.\n"
        "Try restarting from a checkpoint with a lower learning rate."
    ).format(costs_np)
    assert np.all(np.isfinite(grads_np)), ("Error: Gradients not finite.\n"
        "Try restarting from a checkpoint with a lower learning rate.")
    return -costs / nblk, -grads / nblk


class CatModFlipFlop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logprob, seqs, seqlen, mod_cats, can_mods_offsets,
                mod_cat_weights, sharpfact : float):
        ntrans = logprob.shape[2]
        n_can_trans = ntrans - can_mods_offsets[-1]
        nbase = flipflopfings.nbase_flipflop(n_can_trans)
        trans_sharp = torch.ones(
            ntrans, device=logprob.device, dtype=logprob.dtype)
        trans_sharp[:n_can_trans] = sharpfact
        lp = np.ascontiguousarray(
            (logprob * trans_sharp).detach().float().cpu().numpy())
        seqs = seqs.cpu().numpy().astype(np.int32)
        seqlen = seqlen.cpu().numpy().astype(np.int32)
        mod_cats = mod_cats.cpu().numpy().astype(np.int32)

        #  Calculate indices -- seqlen[0:-1] ensures input to split is array
        seq_starts = np.cumsum(seqlen[:-1])
        batch_seqs = np.split(seqs, seq_starts)
        batch_mod_cats = np.split(mod_cats, seq_starts)
        moveidxs = np.concatenate([
            flipflopfings.move_indices(seq, nbase)
            for seq in batch_seqs]).astype(np.uintp)
        stayidxs = np.concatenate([
            flipflopfings.stay_indices(seq, nbase)
            for seq in batch_seqs]).astype(np.uintp)
        assert np.all(np.logical_and(moveidxs >=0, moveidxs < ntrans))
        assert np.all(np.logical_and(stayidxs >=0, stayidxs < ntrans))

        mod_offset = (nbase + 1) * nbase * 2
        mod_seq = np.concatenate([
            can_mods_offsets[np.mod(seq[1:], nbase)] + mod_cat[1:]
            for seq, mod_cat in zip(batch_seqs, batch_mod_cats)]).astype(int)
        modmoveidxs = (mod_offset + mod_seq).astype(np.uintp)
        modmovefacts = mod_cat_weights[mod_seq].astype(np.float32)

        if logprob.requires_grad:
            cost, grads = cat_mod_flipflop_grad(
                lp, moveidxs, stayidxs, modmoveidxs, modmovefacts, seqlen,
                pin=torch.cuda.is_available())
            ctx.save_for_backward(grads.to(logprob.device, non_blocking=True))
        else:
            cost = cat_mod_flipflop_cost(
                lp, moveidxs, stayidxs, modmoveidxs, modmovefacts, seqlen,
                pin=torch.cuda.is_available())
        return (cost / sharpfact).to(logprob.device, non_blocking=True)

    @staticmethod
    def backward(ctx, output_grads):
        grads, = ctx.saved_tensors
        output_grads = output_grads.unsqueeze(1)
        return (grads * output_grads,
                None, None, None, None, None, None, None)

cat_mod_flipflop_loss = CatModFlipFlop.apply
