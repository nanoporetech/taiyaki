#include <stdint.h>

void crf_flipflop_grad(float const *logprob, size_t ntrans, size_t nblk,
                       size_t nbatch, size_t const *moveidxs,
                       size_t const *stayidxs, int32_t const *seqlen,
                       float *score, float *grad);

void crf_flipflop_cost(float const *logprob, size_t ntrans, size_t nblk,
                       size_t nbatch, size_t const *moveidxs,
                       size_t const *stayidxs, int32_t const *seqlen,
                       float *score);
