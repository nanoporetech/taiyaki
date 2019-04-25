#include <stdint.h>

void runlength_grad(float const * logprob, size_t nstate, size_t nblk , size_t nbatch,
                    int32_t const * seqs, int32_t const * rles, int32_t const * seqlen, float * score, float * grad);

void runlength_cost(float const * logprob, size_t nstate, size_t nblk , size_t nbatch,
                    int32_t const * seqs, int32_t const * rles, int32_t const * seqlen, float * score);

