#pragma once

#ifndef FLIPFLOPBWD_H
#define FLIPFLOPBWD_H

#include <stdlib.h>

float flipflop_forward(const float * score, size_t nbase, size_t nblock, float * out);
float flipflop_backward(const float * score, size_t nbase, size_t nblock, float * out);

#endif  /*  FLIPFLOPBWD_H  */

