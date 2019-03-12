#include <stdint.h>
#include <stdlib.h>
void squiggle_match_cost(float const * signal, int32_t const * siglen, size_t nbatch,
                         float const * params, size_t npos, float prob_back, float * score);
void squiggle_match_grad(float const * signal, int32_t const * siglen, size_t nbatch,
                         float const * params, size_t npos, float prob_back, float * grad);
void squiggle_match_viterbi_path(float const * signal, int32_t const * siglen, size_t nbatch,
                                 float const * params, size_t npos, float prob_back, float localpen,
                                 float minscore, int32_t * path, float * score);
