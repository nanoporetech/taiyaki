#pragma once
#ifndef VECT_MATHFUN_H
#define VECT_MATHFUN_H

#define ALIGNMENT 8

#include <math.h>

#include "avx_mathfun.h"

static inline float logsumexpf(float x, float y, float a) {
    /* expf(-17.0f) is below float precision, and so zero */
    float delta = a * fabsf(x - y);
    return fmaxf(x, y) + ((delta < 17.0f) ? (log1pf(expf(-delta)) / a) : 0.0f);
}

static inline float fmaxf_vec(const float *restrict x, size_t n) {
    float xmax = x[0];
    for (size_t i = 1; i < n; i++) {
        xmax = fmaxf(xmax, x[i]);
    }
    return xmax;
}

static inline float logsumexpf_vec(const float *restrict x, size_t n) {
    const float xmax = fmaxf_vec(x, n);
    float Z = expf(x[0] - xmax);
    for (size_t i = 1; i < n; i++) {
        Z += expf(x[i] - xmax);
    }
    return logf(Z);
}

static inline float fmaxf_avx(const float *restrict x, size_t n) {
    const size_t n4 = n >> 3;
    const __m256 *restrict xV = (const __m256 * restrict) x;

    __m256 xmaxV = _mm256_set1_ps(-HUGE_VALF);
    for (size_t i = 0; i < n4; i++) {
        xmaxV = _mm256_max_ps(xmaxV, xV[i]);
    }
    float res[8] __attribute__ ((aligned(32)));
    _mm256_store_ps(res, xmaxV);
    float xmax = fmax(fmaxf(fmaxf(res[0], res[1]), fmaxf(res[2], res[3])),
                      fmaxf(fmaxf(res[4], res[5]), fmaxf(res[6], res[7])));

    for (size_t i = (n4 << 3); i < n; i++) {
        xmax = fmaxf(xmax, x[i]);
    }
    return xmax;
}

static inline float logsumexpf_avx(const float *restrict x, size_t n) {
    const size_t n4 = n >> 3;
    const float xmax = fmaxf_avx(x, n);
    const __m256 xmaxV = _mm256_set1_ps(xmax);
    const __m256 *restrict xV = (const __m256 * restrict) x;

    __m256 Z = _mm256_setzero_ps();
    for (size_t i = 0; i < n4; i++) {
        Z += exp256_ps(xV[i] - xmaxV);
    }

    //  Sum and finalise vectorised bit
    float Zout;
    __m128 Zh = _mm256_castps256_ps128(Z) + _mm256_extractf128_ps(Z, 1);
    Zh = _mm_hadd_ps(Zh, Zh);
    Zh = _mm_hadd_ps(Zh, Zh);
    _mm_store_ss(&Zout, Zh);

    for (size_t i = (n4 << 3); i < n; i++) {
        //  Deal with remaining
        Zout += expf(x[i] - xmax);
    }

    return logf(Zout);
}

static inline void logaddexpf_avx(const float *restrict x, float *restrict y,
                                  float sharp, size_t n) {
    const size_t n4 = n >> 3;
    const __m256 *restrict xV = (const __m256 * restrict) x;
    __m256 *restrict yV = (__m256 * restrict) y;

    const __m256 sharpV = _mm256_set1_ps(sharp);
    const __m256 ones = _mm256_set1_ps(1.0f);
    for (size_t i = 0; i < n4; i++) {
        const __m256 delta = xV[i] - yV[i];
        const __m256 abs_delta = _mm256_max_ps(-delta, delta) * sharpV;
        yV[i] =
            _mm256_max_ps(xV[i],
                          yV[i]) + log256_ps(ones +
                                             exp256_ps(-abs_delta)) / sharpV;
    }

    for (size_t i = (n4 << 3); i < n; i++) {
        //  Deal with remaining
        y[i] =
            fmaxf(x[i],
                  y[i]) + logf(1.0f +
                               expf(-sharp * fabsf(x[i] - y[i]))) / sharp;
    }
}

static inline float softmax_inplace_avx(float *restrict x, float sharp,
                                        size_t n) {
    const size_t n4 = n >> 3;
    __m256 *restrict xV = (__m256 * restrict) x;

    const __m256 sharpV = _mm256_set1_ps(sharp);
    const float xmax = fmaxf_avx(x, n);
    const __m256 xmaxV = _mm256_set1_ps(xmax);
    __m256 Z = _mm256_setzero_ps();
    for (size_t i = 0; i < n4; i++) {
        xV[i] = exp256_ps(sharpV * (xV[i] - xmaxV));
        Z += xV[i];
    }

    //  Sum and finalise vectorised bit
    float Zout;
    __m128 Zh = _mm256_castps256_ps128(Z) + _mm256_extractf128_ps(Z, 1);
    Zh = _mm_hadd_ps(Zh, Zh);
    Zh = _mm_hadd_ps(Zh, Zh);
    _mm_store_ss(&Zout, Zh);

    for (size_t i = (n4 << 3); i < n; i++) {
        //  Deal with remaining
        x[i] = expf(sharp * (x[i] - xmax));
        Zout += x[i];
    }


    Z = _mm256_set1_ps(Zout);
    for (size_t i = 0; i < n4; i++) {
        xV[i] /= Z;
    }
    for (size_t i = (n4 << 3); i < n; i++) {
        x[i] /= Zout;
    }

    return Zout;
}


static inline size_t next_aligned(size_t i, size_t a) {
    const size_t mask = ~(a - 1);
    return (i + a - 1) & mask;
}

#endif                          /* VECT_MATHFUN_H */
