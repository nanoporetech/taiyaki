#pragma once

#ifndef DECODEUTIL_H
#define DECODEUTIL_H

#include <stdlib.h>

//  Type to represent bases by.  Use single byte, since not many bases expected
typedef int8_t base_t;

float flipflop_beamsearch(const float * score, size_t nbase, size_t nblock, const float * bwd, int beam_width, float beamcut, base_t * seq);

#endif  /*  DECODEUTIL_H  */
