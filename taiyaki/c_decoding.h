#include <stdint.h>
#include <stdlib.h>

void fast_viterbi_blocks(float const * weights, size_t nblock, size_t nbatch, size_t nparam, size_t nbase,
                         float stay_pen, float skip_pen, float local_pen, float * score, int32_t * seq);
