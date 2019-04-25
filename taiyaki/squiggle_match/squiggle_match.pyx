cimport libsquiggle_match
import cython
from Bio import SeqIO
import numpy as np
cimport numpy as np
import os
import sys

from ont_fast5_api import fast5_interface
from taiyaki import config, helpers
from taiyaki.maths import mad
from taiyaki.constants import DEFAULT_ALPHABET, LARGE_LOG_VAL

import torch


_base_mapping = {k : i for i, k in enumerate(DEFAULT_ALPHABET)}
_cartesian_tetrahedron = np.array([[1.0, 0.0, -1.0 / np.sqrt(2.0)],
                                   [-1.0, 0.0, -1.0 / np.sqrt(2.0)],
                                   [0.0, 1.0, 1.0 / np.sqrt(2.0)],
                                   [0.0, -1.0, 1.0 / np.sqrt(2.0)]],
                                  dtype=config.taiyaki_dtype)


@cython.boundscheck(False)
@cython.wraparound(False)
def squiggle_match_cost(np.ndarray[np.float32_t, ndim=3, mode="c"] params,
                        np.ndarray[np.float32_t, ndim=1, mode="c"] signal,
                        np.ndarray[np.int32_t, ndim=1, mode="c"] siglen,
                        back_prob):
    """Forward scores of matching observed signals to predicted squiggles

    :param params: A [length, batch, 3] numpy array of predicted squiggle parameters.
        The 3 features are predicted level, spread and movement rate
    :param signal: A vector containing observed signals, concatenated
    :param seqlen: Length of each signal
    :param back_prob: Probably of entering the backsampling state
    """
    cdef size_t npos, nbatch
    npos, nbatch = params.shape[0], params.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs = np.zeros((nbatch,), dtype=np.float32)
    libsquiggle_match.squiggle_match_cost(&signal[0], &siglen[0], nbatch,
                                          &params[0, 0, 0], npos, back_prob,
                                          &costs[0])

    return -costs


@cython.boundscheck(False)
@cython.wraparound(False)
def squiggle_match_grad(np.ndarray[np.float32_t, ndim=3, mode="c"] params,
                        np.ndarray[np.float32_t, ndim=1, mode="c"] signal,
                        np.ndarray[np.int32_t, ndim=1, mode="c"] siglen,
                        back_prob):
    """Gradient of forward scores of matching observed signals to predicted squiggles

    :param params: A [length, batch, 3] numpy array of predicted squiggle parameters.
        The 3 features are predicted level, spread and movement rate
    :param signal: A vector containing observed signals, concatenated
    :param seqlen: Length of each signal
    :param back_prob: Probably of entering the backsampling state
    """
    cdef size_t nblk, nbatch, nstate
    npos, nbatch = params.shape[0], params.shape[1]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] grads = np.zeros_like(params, dtype=np.float32)
    libsquiggle_match.squiggle_match_grad(&signal[0], &siglen[0], nbatch,
                                          &params[0, 0, 0], npos, back_prob,
                                          &grads[0, 0, 0])

    return -grads


@cython.boundscheck(False)
@cython.wraparound(False)
def squiggle_match_path(np.ndarray[np.float32_t, ndim=3, mode="c"] params,
                        np.ndarray[np.float32_t, ndim=1, mode="c"] signal,
                        np.ndarray[np.int32_t, ndim=1, mode="c"] siglen,
                        back_prob, localpen, minscore):
    """Viterbi scores and paths of matching observed signals to predicted squiggles

    :param params: A [length, batch, 3] numpy array of predicted squiggle parameters.
        The 3 features are predicted level, spread and movement rate
    :param signal: A vector containing observed signals, concatenated
    :param seqlen: Length of each signal
    :param back_prob: Probably of entering the backsampling state
    """
    cdef size_t nblk, nbatch, nstate
    npos, nbatch = params.shape[0], params.shape[1]
    localpen = localpen if localpen is not None else LARGE_LOG_VAL
    minscore = minscore if minscore is not None else LARGE_LOG_VAL

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs = np.zeros((nbatch,), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] paths = np.zeros_like(signal, dtype=np.int32)
    libsquiggle_match.squiggle_match_viterbi_path(&signal[0], &siglen[0], nbatch,
                                                  &params[0, 0, 0], npos,
                                                  back_prob, localpen, minscore,
                                                  &paths[0], &costs[0])

    return -costs, paths


def load_references(filename):
    references = dict()
    for seq in SeqIO.parse(filename, 'fasta'):
        references[seq.id] = str(seq.seq)

    return references


def embed_sequence(seq, alphabet=DEFAULT_ALPHABET):
    """Embed sequence of bases (bytes) using points of a tetrahedron"""
    if alphabet == DEFAULT_ALPHABET:
        seq_index = np.array([_base_mapping[b] for b in seq])
    elif alphabet is None:
        seq_index = seq
    else:
        raise Exception('Alphabet not recognised in squiggle_match.pyx embed_sequence()')
    return _cartesian_tetrahedron[seq_index]


def init_worker(model, reference_file):
    torch.set_num_threads(1)

    global predict_squiggle
    predict_squiggle = model

    global references
    references = load_references(reference_file)


def worker(fast5_read_tuple, trim, back_prob, localpen, minscore):
    fast5_name, read_id = fast5_read_tuple
    if read_id in references:
       refseq = references[read_id]
    else:
        sys.stderr.write('Reference not found for {}\n'.format(read_id))
        return None

    try:
        with fast5_interface.get_fast5_file(fast5_name, 'r') as f5file:
            read = f5file.get_read(read_id)
            signal = read.get_raw_data()
    except:
        sys.stderr.write('Error reading {}\n'.format(read_id))
        return None

    signal = helpers.trim_array(signal, *trim)
    assert len(signal) > 0

    norm_sig = (signal - np.median(signal)) / mad(signal)
    norm_sig = np.ascontiguousarray(norm_sig, dtype=config.taiyaki_dtype)

    embedded_seq = np.expand_dims(embed_sequence(refseq), axis=1)
    with torch.no_grad():
        squiggle_params = predict_squiggle(torch.tensor(embedded_seq, dtype=torch.float32)).cpu().numpy()
    sig_len = np.array([len(norm_sig)], dtype=np.int32)

    squiggle_params = np.ascontiguousarray(squiggle_params, dtype=np.float32)
    cost, path = squiggle_match_path(squiggle_params, norm_sig, sig_len,
                                     back_prob, localpen, minscore)

    return (read_id, norm_sig, cost[0], path,
            np.squeeze(squiggle_params, axis=1), refseq)


class SquiggleMatch(torch.autograd.Function):
    """Pytorch autograd function wrapping squiggle_match_cost"""
    @staticmethod
    def forward(ctx, params, signal, siglen, back_prob):
        ctx.save_for_backward(params, signal, siglen, torch.tensor(back_prob))
        params = np.ascontiguousarray(params.detach().cpu().numpy().astype(np.float32))
        signal = np.ascontiguousarray(signal.detach().cpu().numpy().astype(np.float32))
        siglen = np.ascontiguousarray(siglen.detach().cpu().numpy().astype(np.int32))
        back_prob = float(back_prob)
        cost = squiggle_match_cost(params, signal, siglen, back_prob)
        return torch.tensor(cost)

    @staticmethod
    def backward(ctx, output_grads):
        params, signal, siglen, back_prob = ctx.saved_tensors
        device = params.device
        dtype = params.dtype
        params = np.ascontiguousarray(params.detach().cpu().numpy().astype(np.float32))
        signal = np.ascontiguousarray(signal.detach().cpu().numpy().astype(np.float32))
        siglen = np.ascontiguousarray(siglen.detach().cpu().numpy().astype(np.int32))
        back_prob = float(back_prob)
        grad = squiggle_match_grad(params, signal, siglen, back_prob)
        grad = torch.tensor(grad, dtype=dtype, device=device)
        output_grads = output_grads.unsqueeze(1).to(device)
        return grad * output_grads, None, None, None


squiggle_match_loss = SquiggleMatch.apply
