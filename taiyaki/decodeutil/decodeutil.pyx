cimport libdecodeutil
import cython
import numpy as np
cimport numpy as np
from taiyaki.flipflopfings import nbase_flipflop

@cython.boundscheck(False)
@cython.wraparound(False)
def beamsearch(np.ndarray[np.float32_t, ndim=2, mode="c"] score,
               beam_cut=0.0, beam_width=5, guided=True):
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
def backward(np.ndarray[np.float32_t, ndim=2, mode="c"] score):
    cdef size_t nbase, nt, nf, seqlen
    cdef float read_score
    nt, nf = score.shape[0], score.shape[1]
    nbase = nbase_flipflop(nf)

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] res = np.zeros((nt + 1, nbase + nbase), dtype='f4')
    read_score = libdecodeutil.flipflop_backward(&score[0,0], nbase, nt, &res[0, 0])

    return res, read_score


@cython.boundscheck(False)
@cython.wraparound(False)
def forward(np.ndarray[np.float32_t, ndim=2, mode="c"] score):
    cdef size_t nbase, nt, nf, seqlen
    cdef float read_score
    nt, nf = score.shape[0], score.shape[1]
    nbase = nbase_flipflop(nf)

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] res = np.zeros((nt + 1, nbase + nbase), dtype='f4')
    read_score = libdecodeutil.flipflop_forward(&score[0,0], nbase, nt, &res[0, 0])

    return res, read_score
