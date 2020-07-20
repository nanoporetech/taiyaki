#pragma once
#ifndef VECT_MATHFUN_H
#define VECT_MATHFUN_H

#include <math.h>

#include "sse_mathfun.h"
#include "avx_mathfun.h"

#define ALIGNMENT 8


static inline float logsumexpf(float x, float y, float a) {
    /* expf(-17.0f) is below float precision, and so zero */
    float delta = a * fabsf(x - y);
    return fmaxf(x, y) + ((delta < 17.0f) ? (log1pf(expf(-delta)) / a) : 0.0f);
}

static inline size_t nstate_to_nbase(size_t ntrans) {
    double nbase_d = sqrt(0.25 + (0.5 * ntrans)) - 0.5;
    assert(fmod(nbase_d, 1.0) == 0.0);
    return (size_t) round(nbase_d);
}

static inline float fmaxf_vec(const float *restrict x, size_t n) {
    float xmax = x[0];
    for (size_t i = 1; i < n; i++) {
        xmax = fmaxf(xmax, x[i]);
    }
    return xmax;
}

static inline float fmaxf_sse(const float *restrict x, size_t n) {
    const size_t n4 = n >> 2;
    const __m128 *restrict xV = (const __m128 * restrict) x;

    __m128 xmaxV = _mm_set1_ps(-HUGE_VALF);
    for (size_t i = 0; i < n4; i++) {
        xmaxV = _mm_max_ps(xmaxV, xV[i]);
    }
    float res[4] __attribute__ ((aligned(16)));
    _mm_store_ps(res, xmaxV);
    float xmax = fmaxf(fmaxf(res[0], res[1]), fmaxf(res[2], res[3]));

    for (size_t i = (n4 << 2); i < n; i++) {
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

static inline float logsumexpf_sse(const float *restrict x, size_t n) {
    const size_t n4 = n >> 2;
    const float xmax = fmaxf_sse(x, n);
    const __m128 xmaxV = _mm_set1_ps(xmax);
    const __m128 *restrict xV = (const __m128 * restrict) x;

    __m128 Z = _mm_setzero_ps();
    for (size_t i = 0; i < n4; i++) {
        Z += exp_ps(xV[i] - xmaxV);
    }

    //  Sum and finalise vectorised bit
    float Zout;
    Z = _mm_hadd_ps(Z, Z);
    Z = _mm_hadd_ps(Z, Z);
    _mm_store_ss(&Zout, Z);

    for (size_t i = (n4 << 2); i < n; i++) {
        //  Deal with remaining
        Zout += expf(x[i] - xmax);
    }

    return logf(Zout);
}

static inline void logaddexpf_sse(const float *restrict x, float *restrict y,
                                  float sharp, size_t n) {
    const size_t n4 = n >> 2;
    const __m128 *restrict xV = (const __m128 * restrict) x;
    __m128 *restrict yV = (__m128 * restrict) y;

    const __m128 sharpV = _mm_set1_ps(sharp);
    for (size_t i = 0; i < n4; i++) {
        const __m128 delta = xV[i] - yV[i];
        const __m128 abs_delta = _mm_max_ps(-delta, delta) * sharpV;
        const __m128 ones = _mm_set1_ps(1.0f);
        yV[i] =
            _mm_max_ps(xV[i],
                       yV[i]) + log_ps(ones + exp_ps(-abs_delta)) / sharp;
    }

    for (size_t i = (n4 << 2); i < n; i++) {
        //  Deal with remaining
        y[i] =
            fmaxf(x[i],
                  y[i]) + logf(1.0f +
                               expf(-sharp * fabsf(x[i] - y[i]))) / sharp;
    }
}

static inline float softmax_inplace_sse(float *restrict x, float sharp, size_t n) {
    const size_t n4 = n >> 2;
    __m128 *restrict xV = (__m128 * restrict) x;

    const __m128 sharpV = _mm_set1_ps(sharp);
    const float xmax = fmaxf_sse(x, n);
    const __m128 xmaxV = _mm_set1_ps(xmax);
    __m128 Z = _mm_setzero_ps();
    for (size_t i = 0; i < n4; i++) {
        xV[i] = exp_ps(sharpV * (xV[i] - xmaxV));
        Z += xV[i];
    }
    //  Sum and finalise vectorised bit
    float Zout;
    Z = _mm_hadd_ps(Z, Z);
    Z = _mm_hadd_ps(Z, Z);
    _mm_store_ss(&Zout, Z);

    for (size_t i = (n4 << 2); i < n; i++) {
        //  Deal with remaining
        x[i] = expf(sharp * (x[i] - xmax));
        Zout += x[i];
    }

    Z = _mm_set1_ps(Zout);
    for (size_t i = 0; i < n4; i++) {
        xV[i] /= Z;
    }
    for (size_t i = (n4 << 2); i < n; i++) {
        x[i] /= Zout;
    }

    return Zout;
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

static inline float softmax_inplace_avx(float *restrict x, float sharp, size_t n) {
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
#endif            /* VECT_MATHFUN_H */