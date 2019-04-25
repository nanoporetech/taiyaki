from libc.stdint cimport int32_t

cdef extern from "c_crf_flipflop.h":
    void crf_flipflop_grad(
        const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
        const int32_t * seqs, const int32_t * seqlen, float sharpfact,
        float * score, float * grad);

    void crf_flipflop_cost(
        const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
        const int32_t * seqs, const int32_t * seqlen, float sharpfact,
        float * score);


cdef extern from "c_runlength.h":
    void runlength_grad(const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
                        const int32_t * seqs, const int32_t * rles, const int32_t * seqlen,
                        float * score, float * grad);

    void runlength_cost(const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
                        const int32_t * seqs, const int32_t * rles, const int32_t * seqlen,
                        float * score);


cdef extern from "c_cat_mod_flipflop.h":
    void cat_mod_flipflop_grad(
        const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
        const int32_t * seqs, const int32_t * seqlen, const int32_t * mod_cats,
        const int32_t * can_mods_offsets, const float * mod_cat_weights,
        float mod_weight, float sharpfact, float * grad);

    void cat_mod_flipflop_cost(
        const float * logprob, size_t nstate, size_t nblk , size_t nbatch,
        const int32_t * seqs, const int32_t * seqlen, const int32_t * mod_cats,
        const int32_t * can_mods_offsets, const float * mod_cat_weights,
        float mod_weight, float sharpfact, float * grad);
