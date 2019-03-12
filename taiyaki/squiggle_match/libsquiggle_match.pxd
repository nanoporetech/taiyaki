from libc.stdint cimport int32_t
cdef extern from "c_squiggle_match.h":
    void squiggle_match_cost(const float * signal, const int32_t * siglen, size_t nbatch,
                             const float * params, size_t npos, float prob_back, float * score)
    void squiggle_match_grad(const float * signal, const int32_t * siglen, size_t nbatch,
                             const float * params, size_t npos, float prob_back, float * grad)
    void squiggle_match_viterbi_path(const float * signal, const int32_t * siglen,
                                     size_t nbatch, const float * params, size_t npos,
                                     float prob_back, float localpen, float minscore,
                                     int32_t * path, float * score)
