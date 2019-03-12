from libc.stdint cimport int32_t

cdef extern from "c_crf_flipflop.h":
    void crf_flipflop_grad(const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
                           const int32_t * seqs, const int32_t * seqlen, float sharpfact,
                           float * score, float * grad);

    void crf_flipflop_cost(const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
                           const int32_t * seqs, const int32_t * seqlen, float sharpfact, float * score);
