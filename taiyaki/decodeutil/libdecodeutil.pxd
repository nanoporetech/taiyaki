from libc.stdint cimport int8_t
cdef extern from "c_hashdecode.h":
    size_t flipflop_beamsearch(const float * score, size_t nbase, size_t n,
                               const float * bwd, int beam_size, float beamcut,
                               int8_t * seq)

cdef extern from "c_flipflopfwdbwd.h":
    float flipflop_forward(const float * score, size_t nbase, size_t nblock,
                           float * out)
    float flipflop_backward(const float * score, size_t nbase, size_t nblock,
                            float * out)

