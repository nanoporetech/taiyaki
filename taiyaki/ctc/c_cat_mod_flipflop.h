#include <stdint.h>

void cat_mod_flipflop_grad(float const *logprob, size_t ntrans, size_t nblk,
                           size_t nbatch, size_t const *moveidxs,
                           size_t const *stayidxs, size_t const *modmoveidxs,
                           float const *modmovefacts, int32_t const *seqlen,
                           float *score, float *grad);

void cat_mod_flipflop_cost(float const *logprob, size_t ntrans, size_t nblk,
                           size_t nbatch, size_t const *moveidxs,
                           size_t const *stayidxs, size_t const *modmoveidxs,
                           float const *modmovefacts, int32_t const *seqlen,
                           float *score);
