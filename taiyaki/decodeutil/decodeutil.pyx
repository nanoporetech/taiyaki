cimport libdecodeutil
import cython
import numpy as np
cimport numpy as np
from taiyaki.flipflopfings import nbase_flipflop

@cython.boundscheck(False)
@cython.wraparound(False)
def beamsearch(np.ndarray[np.float32_t, ndim=2, mode="c"] score,
               beam_cut=0.0, beam_width=5, guided=True):
    """  Conduct beam search for flip-flop model

    Beam search decoding for best sequence, optionally guided by backwards
      calls.

    Notes:
        Beams are cut in log-space, so `beam_cut` of 0.0 means no beams are cut.
          Value of beam_cut is approximately the Bayes factor between the
          proposed and best element of the beam.  Because best is updated
          continuously as base extensions are proposed, elements of the beam
          may be kept that would have been discarded had they been proposed
          later.

    Args:
        score (:class:`ndarray`): input scores (output of network) for decoding
        beam_cut (float): discard beam extensions whose score is `beam_cut` or
           more worse than the best found.
        beam_width (int): Maximum width (number of elements) in beam
        guided (bool): Whether to inform decoding using backwards scores

    Returns:
        Tuple[:class:`ndarray`, float]: Decoded sequence (integer encoded) and
          score for read
    """
    cdef size_t nbase, nt, nf, seqlen
    cdef float read_score
    nt, nf = score.shape[0], score.shape[1]
    nbase = nbase_flipflop(nf)

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] bwd
    if guided:
        bwd = backward(score)[0]
    else:
        bwd = np.zeros((nt + 1, nbase + nbase), dtype=np.float32)

    cdef np.ndarray[np.int8_t, ndim=1, mode="c"] res = np.zeros((nt,), dtype=np.int8)
    read_score = libdecodeutil.flipflop_beamsearch(&score[0,0], nbase, nt, &bwd[0,0],
                                                   beam_width, beam_cut, &res[0])
    seqlen = np.nonzero(res == -1)[0][0]

    return res[:seqlen], read_score


@cython.boundscheck(False)
@cython.wraparound(False)
def backward(np.ndarray[np.float32_t, ndim=2, mode="c"] score, init=None):
    """  Backwards calculation of flipflop scores

    Performs backward algorithm back the flipflop CRF, returning the backward
    scores for each time step and the log-partition.

    Args:
        score (:class:`ndarray`): input scores (output of network) for decoding
        init (None or :class:`ndarray`): Array[nbase+nbase] containing initial
           state or None.

    Returns:
        Tuple[:class:`ndarray`, float]: backward scores and log-partition
    """
    cdef size_t nbase, nt, nf, seqlen
    cdef float read_score
    nt, nf = score.shape[0], score.shape[1]
    nbase = nbase_flipflop(nf)

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] res = np.zeros((nt + 1, nbase + nbase), dtype='f4')
    if init is not None:
        res[nt] = init

    read_score = libdecodeutil.flipflop_backward(&score[0,0], nbase, nt, &res[0, 0])

    return res, read_score


@cython.boundscheck(False)
@cython.wraparound(False)
def forward(np.ndarray[np.float32_t, ndim=2, mode="c"] score, init=None):
    """  Forwards calculation of flipflop scores

    Performs forward algorithm for the flipflop CRF, returning the forward
    scores for each time step and the log-partition.

    Args:
        score (:class:`ndarray`): input scores (output of network) for decoding
        init (None or :class:`ndarray`): Array[nbase+nbase] containing initial
           state or None.

    Returns:
        Tuple[:class:`ndarray`, float]: forward scores and log-partition
    """
    cdef size_t nbase, nt, nf, seqlen
    cdef float read_score
    nt, nf = score.shape[0], score.shape[1]
    nbase = nbase_flipflop(nf)

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] res = np.zeros((nt + 1, nbase + nbase), dtype='f4')
    if init is not None:
        print('init is', init)
        res[0] = init
    read_score = libdecodeutil.flipflop_forward(&score[0,0], nbase, nt, &res[0, 0])

    return res, read_score
