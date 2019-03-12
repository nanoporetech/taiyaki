from libc.stdint cimport int32_t
cdef extern from "c_decoding.h":
    void fast_viterbi_blocks(const float * weights, size_t nblock, size_t nbatch, size_t nparam, size_t nbase,
                             float stay_pen, float skip_pen, float local_pen, float * score, int32_t * path)
