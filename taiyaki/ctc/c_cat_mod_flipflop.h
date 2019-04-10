#include <stdint.h>

void cat_mod_flipflop_grad(float const * logprob, size_t nstate,
                           size_t nblk, size_t nbatch, int32_t const * seqs,
                           int32_t const * seqlen, int32_t const * mod_cats,
                           int32_t const * can_mods_offsets,
                           float const * mod_cat_weights, float mod_weight,
                           float sharpfact, float * grad);

void cat_mod_flipflop_cost(float const * logprob, size_t nstate,
                           size_t nblk, size_t nbatch, int32_t const * seqs,
                           int32_t const * seqlen, int32_t const * mod_cats,
                           int32_t const * can_mods_offsets,
                           float const * mod_cat_weights, float mod_weight,
                           float sharpfact, float * score);
