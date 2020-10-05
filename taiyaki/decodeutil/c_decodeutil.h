#pragma once

#ifndef DECODEUTIL_H
#define DECODEUTIL_H

#include <stdlib.h>

typedef int8_t base_t;

float flipflop_beamsearch(const float * score, size_t nbase, size_t nblock, int beam_width, base_t * seq);

#endif  /*  DECODEUTIL_H  */
