from libc.stdint cimport int32_t

cdef extern from "c_crf_flipflop.h":
    void crf_flipflop_grad(
        const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
        const size_t * moveidxs, const size_t * stayidxs,
        const int32_t * seqlen, float sharpfact, float * score, float * grad);

    void crf_flipflop_cost(
        const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
        const size_t * moveidxs, const size_t * stayidxs,
        const int32_t * seqlen, float sharpfact, float * score);

cdef extern from "c_cat_mod_flipflop.h":
    void cat_mod_flipflop_grad(
        const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
        const size_t * moveidxs, const size_t * stayidxs,
        const size_t * modmoveidxs, const float * modmovefacts,
        const int32_t * seqlen, float sharpfact, float * score, float * grad);

    void cat_mod_flipflop_cost(
        const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
        const size_t * moveidxs, const size_t * stayidxs,
        const size_t * modmoveidxs, const float * modmovefacts,
        const int32_t * seqlen, float sharpfact, float * grad);
