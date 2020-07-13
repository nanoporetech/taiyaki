#include <stdint.h>

void crf_flipflop_grad(float const *logprob, size_t ntrans, size_t nblk,
                       size_t nbatch, int32_t const *moveidxs,
                       int32_t const *stayidxs, int32_t const *seqlen,
                       float sharpfact, float *score, float *grad);

void crf_flipflop_cost(float const *logprob, size_t ntrans, size_t nblk,
                       size_t nbatch, int32_t const *moveidxs,
                       int32_t const *stayidxs, int32_t const *seqlen,
                       float sharpfact, float *score);
