#include <stdint.h>

void crf_flipflop_grad(float const * logprob, size_t nstate, size_t nblk , size_t nbatch,
                       int32_t const * seqs, int32_t const * seqlen, float sharpfact, float * score,
		       float * grad);

void crf_flipflop_cost(float const * logprob, size_t nstate, size_t nblk , size_t nbatch,
                       int32_t const * seqs, int32_t const * seqlen, float sharpfact, float * score);
