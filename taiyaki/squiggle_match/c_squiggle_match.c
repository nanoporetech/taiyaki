#define _BSD_SOURCE 1
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static float LARGE_VAL = 1e30f;
static size_t nparam = 3;

static inline float loglaplace(float x, float loc, float sc, float logsc) {
    return -fabsf(x - loc) / sc - logsc - M_LN2;
}

static inline float laplace(float x, float loc, float sc, float logsc) {
    return expf(loglaplace(x, loc, sc, logsc));
}

static inline float dloglaplace_loc(float x, float loc, float sc, float logsc) {
    return ((x > loc) - (x < loc)) / sc;
}

static inline float dloglaplace_scale(float x, float loc, float sc, float logsc) {
    return (fabsf(x - loc) / sc - 1.0) / sc;
}

static inline float dloglaplace_logscale(float x, float loc, float sc,
                                         float logsc) {
    return fabsf(x - loc) / sc - 1.0;
}

static inline float dlaplace_loc(float x, float loc, float sc, float logsc) {
    return laplace(x, loc, sc, logsc) * dloglaplace_loc(x, loc, sc, logsc);
}

static inline float dlaplace_scale(float x, float loc, float sc, float logsc) {
    return laplace(x, loc, sc, logsc) * dloglaplace_scale(x, loc, sc, logsc);
}

static inline float dlaplace_logscale(float x, float loc, float sc, float logsc) {
    return laplace(x, loc, sc, logsc) * dloglaplace_logscale(x, loc, sc, logsc);
}

static inline float plogisticf(float x) {
    return 0.5f * (1.0f + tanhf(x / 2.0f));
}

static inline float logplogisticf(float x) {
    return -log1pf(expf(-x));
}

static inline float qlogisticf(float p) {
    return 2.0f * atanhf(2.0f * p - 1.0f);
}

static inline float dlogisticf(float x) {
    const float p = plogisticf(x);
    return p * (1.0f - p);
}



static inline float logsumexp(float x, float y) {
    return fmaxf(x, y) + log1pf(expf(-fabsf(x - y)));
}


static inline float max_array(const float *x, size_t n) {
    float max = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    return max;
}

static inline float sum_array(const float *x, size_t n) {
    float sum = x[0];
    for (size_t i = 1; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

static inline float logsum_array(const float *x, size_t n) {
    float sum = x[0];
    for (size_t i = 1; i < n; i++) {
        sum = logsumexp(sum, x[i]);
    }
    return sum;
}

static inline void softmax_inplace(float *x, size_t n) {
    const float xmax = max_array(x, n);

    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - xmax);
    }

    const float sum = sum_array(x, n);
    for (size_t i = 0; i < n; i++) {
        x[i] /= sum;
    }
}



float squiggle_match_forward(float const *signal, size_t nsample,
                             float const *params, size_t ldp,
                             float const *scale, size_t npos, float prob_back,
                             float *fwd) {
    assert(nsample > 0);
    assert(npos > 0);
    assert(NULL != signal);
    assert(NULL != params);
    assert(NULL != scale);
    assert(NULL != fwd);
    const size_t nstate = 2 * npos;

    const float move_back_pen = logf(prob_back);
    const float stay_in_back_pen = logf(0.5f);
    const float move_from_back_pen = logf(0.5f);

    float *move_pen = calloc(npos, sizeof(float));
    float *stay_pen = calloc(npos, sizeof(float));
    for (size_t pos = 0; pos < npos; pos++) {
        const float mp = (1.0f - prob_back) * plogisticf(params[pos * ldp + 2]);
        move_pen[pos] = logf(mp);
        stay_pen[pos] = log1pf(-mp - prob_back);
    }

    // Point prior  -- must start at beginning of sequence
    for (size_t pos = 0; pos < nstate; pos++) {
        fwd[pos] = -LARGE_VAL;
    }
    fwd[0] = 0.0;

    for (size_t sample = 0; sample < nsample; sample++) {
        const size_t fwd_prev_off = sample * nstate;
        const size_t fwd_curr_off = sample * nstate + nstate;

        for (size_t pos = 0; pos < npos; pos++) {
            //  Stay in same position
            fwd[fwd_curr_off + pos] = fwd[fwd_prev_off + pos] + stay_pen[pos];
        }
        for (size_t pos = 0; pos < npos; pos++) {
            // Stay in backwards state
            fwd[fwd_curr_off + npos + pos] =
                fwd[fwd_prev_off + npos + pos] + stay_in_back_pen;
        }
        for (size_t pos = 1; pos < npos; pos++) {
            //  Move to next position
            fwd[fwd_curr_off + pos] = logsumexp(fwd[fwd_curr_off + pos],
                                                fwd[fwd_prev_off + pos - 1] +
                                                move_pen[pos]);
        }
        for (size_t pos = 1; pos < npos; pos++) {
            // Move backwards
            fwd[fwd_curr_off + npos + pos - 1] =
                logsumexp(fwd[fwd_curr_off + npos + pos - 1],
                          fwd[fwd_prev_off + pos] + move_back_pen);
        }
        for (size_t pos = 1; pos < npos; pos++) {
            // Move from back state
            fwd[fwd_curr_off + pos] = logsumexp(fwd[fwd_curr_off + pos],
                                                fwd[fwd_prev_off + npos + pos -
                                                    1] + move_from_back_pen);
        }

        for (size_t pos = 0; pos < npos; pos++) {
            // Add on emission
            const float location = params[pos * ldp + 0];
            const float logscale = params[pos * ldp + 1];
            const float logscore =
                loglaplace(signal[sample], location, scale[pos], logscale);
            fwd[fwd_curr_off + pos] += logscore;
            fwd[fwd_curr_off + npos + pos] += logscore;
        }
    }

    free(move_pen);
    free(stay_pen);

    // Must finish in final position
    return fwd[nsample * nstate + npos - 1];
}


float squiggle_match_backward(float const *signal, size_t nsample,
                              float const *params, size_t ldp,
                              float const *scale, size_t npos, float prob_back,
                              float *bwd) {
    assert(nsample > 0);
    assert(npos > 0);
    assert(NULL != signal);
    assert(NULL != params);
    assert(NULL != scale);
    assert(NULL != bwd);
    const size_t nstate = 2 * npos;

    const float move_back_pen = logf(prob_back);
    const float stay_in_back_pen = logf(0.5f);
    const float move_from_back_pen = logf(0.5f);

    float *move_pen = calloc(npos, sizeof(float));
    float *stay_pen = calloc(npos, sizeof(float));
    for (size_t pos = 0; pos < npos; pos++) {
        const float mp = (1.0f - prob_back) * plogisticf(params[pos * ldp + 2]);
        move_pen[pos] = logf(mp);
        stay_pen[pos] = log1pf(-mp - prob_back);
    }


    float *tmp = calloc(nstate, sizeof(float));

    // Point prior  -- must start at end of sequence
    for (size_t pos = 0; pos < nstate; pos++) {
        bwd[nstate * nsample + pos] = -LARGE_VAL;
    }
    bwd[nstate * nsample + npos - 1] = 0.0;

    for (size_t sample = nsample; sample > 0; sample--) {
        const size_t bwd_prev_off = sample * nstate;
        const size_t bwd_curr_off = sample * nstate - nstate;
        for (size_t pos = 0; pos < npos; pos++) {
            const float location = params[pos * ldp + 0];
            const float logscale = params[pos * ldp + 1];
            const float logscore =
                loglaplace(signal[sample - 1], location, scale[pos], logscale);
            tmp[pos] = bwd[bwd_prev_off + pos] + logscore;
            tmp[npos + pos] = bwd[bwd_prev_off + npos + pos] + logscore;
        }
        for (size_t pos = 0; pos < npos; pos++) {
            // Stay in position
            bwd[bwd_curr_off + pos] = tmp[pos] + stay_pen[pos];
        }
        for (size_t pos = 1; pos < npos; pos++) {
            //  Move to next position
            bwd[bwd_curr_off + pos - 1] = logsumexp(bwd[bwd_curr_off + pos - 1],
                                                    tmp[pos] + move_pen[pos]);
        }
        for (size_t pos = 0; pos < npos; pos++) {
            // Stay in back state
            bwd[bwd_curr_off + npos + pos] = tmp[npos + pos] + stay_in_back_pen;
        }
        for (size_t pos = 1; pos < npos; pos++) {
            // Move out of back state
            bwd[bwd_curr_off + npos + pos - 1] =
                logsumexp(bwd[bwd_curr_off + npos + pos - 1],
                          tmp[pos] + move_from_back_pen);
        }
        for (size_t pos = 1; pos < npos; pos++) {
            // Move into back state
            bwd[bwd_curr_off + pos] = logsumexp(bwd[bwd_curr_off + pos],
                                                tmp[npos + pos - 1] +
                                                move_back_pen);
        }
    }

    free(move_pen);
    free(stay_pen);

    free(tmp);

    // Must start in first position
    return bwd[0];
}


float squiggle_match_viterbi(float const *signal, size_t nsample,
                             float const *params, size_t ldp,
                             float const *scale, size_t npos, float prob_back,
                             float localpen, float minscore, int32_t * path,
                             float *fwd) {
    assert(nsample > 0);
    assert(npos > 0);
    assert(NULL != signal);
    assert(NULL != params);
    assert(NULL != scale);
    assert(NULL != path);
    assert(NULL != fwd);
    const size_t nfstate = npos + 2;
    const size_t nstate = npos + nfstate;

    const float move_back_pen = logf(prob_back);
    const float stay_in_back_pen = logf(0.5f);
    const float move_from_back_pen = logf(0.5f);

    float *move_pen = calloc(nfstate, sizeof(float));
    float *stay_pen = calloc(nfstate, sizeof(float));
    {
        float mean_move_pen = 0.0f;
        float mean_stay_pen = 0.0f;
        for (size_t pos = 0; pos < npos; pos++) {
            const float mp =
                (1.0f - prob_back) * plogisticf(params[pos * ldp + 2]);
            move_pen[pos + 1] = logf(mp);
            stay_pen[pos + 1] = log1pf(-mp - prob_back);
            mean_move_pen += move_pen[pos + 1];
            mean_stay_pen += stay_pen[pos + 1];
        }
        mean_move_pen /= npos;
        mean_stay_pen /= npos;

        move_pen[0] = mean_move_pen;
        move_pen[nfstate - 1] = mean_move_pen;
        stay_pen[0] = mean_stay_pen;
        stay_pen[nfstate - 1] = mean_stay_pen;
    }

    for (size_t st = 0; st < nstate; st++) {
        // States are start .. positions .. end
        fwd[st] = -LARGE_VAL;
    }
    // Must begin in start state
    fwd[0] = 0.0;

    int32_t *traceback = calloc(nsample * nstate, sizeof(int32_t));

    for (size_t sample = 0; sample < nsample; sample++) {
        const size_t fwd_prev_off = (sample % 2) * nstate;
        const size_t fwd_curr_off = ((sample + 1) % 2) * nstate;
        const size_t tr_off = sample * nstate;

        for (size_t st = 0; st < nfstate; st++) {
            //  Stay in start, end or normal position
            fwd[fwd_curr_off + st] = fwd[fwd_prev_off + st] + stay_pen[st];
            traceback[tr_off + st] = st;
        }
        for (size_t st = 0; st < npos; st++) {
            //  Stay in back position
            const size_t idx = nfstate + st;
            fwd[fwd_curr_off + idx] =
                fwd[fwd_prev_off + idx] + stay_in_back_pen;
            traceback[tr_off + idx] = idx;
        }
        for (size_t st = 1; st < nfstate; st++) {
            //  Move to next position
            const float step_score =
                fwd[fwd_prev_off + st - 1] + move_pen[st - 1];
            if (step_score > fwd[fwd_curr_off + st]) {
                fwd[fwd_curr_off + st] = step_score;
                traceback[tr_off + st] = st - 1;
            }
        }
        for (size_t destpos = 1; destpos < npos; destpos++) {
            const size_t destst = destpos + 1;
            //  Move from start into sequence
            const float score =
                fwd[fwd_prev_off] + move_pen[0] - localpen * destpos;
            if (score > fwd[fwd_curr_off + destst]) {
                fwd[fwd_curr_off + destst] = score;
                traceback[tr_off + destst] = 0;
            }
        }
        for (size_t origpos = 0; origpos < (npos - 1); origpos++) {
            const size_t destst = nfstate - 1;
            const size_t origst = origpos + 1;
            const size_t deltapos = npos - 1 - origpos;
            //  Move from sequence into end
            const float score =
                fwd[fwd_prev_off + origst] + move_pen[origst] -
                localpen * deltapos;
            if (score > fwd[fwd_curr_off + destst]) {
                fwd[fwd_curr_off + destst] = score;
                traceback[tr_off + destst] = origst;
            }
        }
        for (size_t st = 1; st < npos; st++) {
            // Move to back
            const float back_score = fwd[fwd_prev_off + st + 1] + move_back_pen;
            if (back_score > fwd[fwd_curr_off + nfstate + st - 1]) {
                fwd[fwd_curr_off + nfstate + st - 1] = back_score;
                traceback[tr_off + nfstate + st - 1] = st + 1;
            }
        }
        for (size_t st = 1; st < npos; st++) {
            // Move from back
            const float back_score =
                fwd[fwd_prev_off + nfstate + st - 1] + move_from_back_pen;
            if (back_score > fwd[fwd_curr_off + st + 1]) {
                fwd[fwd_curr_off + st + 1] = back_score;
                traceback[tr_off + st + 1] = nfstate + st - 1;
            }
        }


        for (size_t pos = 0; pos < npos; pos++) {
            //  Add on score for samples
            const float location = params[pos * ldp + 0];
            const float logscale = params[pos * ldp + 1];
            const float logscore =
                fmaxf(-minscore,
                      loglaplace(signal[sample], location, scale[pos],
                                 logscale));
            //  State to add to is offset by one because of start state
            fwd[fwd_curr_off + pos + 1] += logscore;
            fwd[fwd_curr_off + nfstate + pos] += logscore;
        }

        // Score for start and end states
        fwd[fwd_curr_off + 0] -= localpen;
        fwd[fwd_curr_off + nfstate - 1] -= localpen;

    }

    //  Score of best path and final states.  Could be either last position or end state
    const size_t fwd_offset = (nsample % 2) * nstate;
    const float score =
        fmaxf(fwd[fwd_offset + nfstate - 2], fwd[fwd_offset + nfstate - 1]);
    if (fwd[fwd_offset + nfstate - 2] > fwd[fwd_offset + nfstate - 1]) {
        path[nsample - 1] = nfstate - 2;
    } else {
        path[nsample - 1] = nfstate - 1;
    }

    for (size_t sample = 1; sample < nsample; sample++) {
        const size_t rs = nsample - sample;
        const size_t tr_off = rs * nstate;
        path[rs - 1] = traceback[tr_off + path[rs]];
    }
    free(traceback);

    // Correct path so start and end states are encoded as -1, other states as positions
    {
        size_t sample_min = 0;
        size_t sample_max = nsample;
        for (; sample_min < nsample; sample_min++) {
            if (0 != path[sample_min]) {
                break;
            }
            path[sample_min] = -1;
        }
        for (; sample_max > 0; sample_max--) {
            if (nfstate - 1 != path[sample_max - 1]) {
                break;
            }
            path[sample_max - 1] = -1;
        }
        for (size_t sample = sample_min; sample < sample_max; sample++) {
            assert(path[sample] > 0);
            if (path[sample] >= nfstate) {
                path[sample] -= nfstate;
            } else {
                path[sample] -= 1;
            }
        }
    }

    free(move_pen);
    free(stay_pen);

    return score;
}


void squiggle_match_cost(float const *signal, int32_t const *siglen,
                         size_t nbatch, float const *params, size_t npos,
                         float prob_back, float *score) {
    size_t sigidx[nbatch];
    sigidx[0] = 0;
    for (size_t idx = 1; idx < nbatch; idx++) {
        sigidx[idx] = sigidx[idx - 1] + siglen[idx - 1];
    }

#pragma omp parallel for
    for (size_t batch = 0; batch < nbatch; batch++) {
        const size_t nsample = siglen[batch];
        const size_t signal_offset = sigidx[batch];
        const size_t param_offset = batch * nparam;
        const size_t ldp = nbatch * nparam;

        float *fwd = calloc(2 * npos * (nsample + 1), sizeof(float));
        float *scale = calloc(npos, sizeof(float));
        for (size_t pos = 0; pos < npos; pos++) {
            scale[pos] = expf(params[param_offset + pos * ldp + 1]);
        }
        score[batch] =
            squiggle_match_forward(signal + signal_offset, nsample,
                                   params + param_offset, ldp, scale, npos,
                                   prob_back, fwd);
        free(scale);
        free(fwd);
    }
}


void squiggle_match_scores_fwd(float const *signal, int32_t const *siglen,
                               size_t nbatch, float const *params, size_t npos,
                               float prob_back, float *score) {
    squiggle_match_cost(signal, siglen, nbatch, params, npos, prob_back, score);
}


void squiggle_match_scores_bwd(float const *signal, int32_t const *siglen,
                               size_t nbatch, float const *params, size_t npos,
                               float prob_back, float *score) {
    size_t sigidx[nbatch];
    sigidx[0] = 0;
    for (size_t idx = 1; idx < nbatch; idx++) {
        sigidx[idx] = sigidx[idx - 1] + siglen[idx - 1];
    }

#pragma omp parallel for
    for (size_t batch = 0; batch < nbatch; batch++) {
        const size_t nsample = siglen[batch];
        const size_t signal_offset = sigidx[batch];
        const size_t param_offset = batch * nparam;
        const size_t ldp = nbatch * nparam;
        float *bwd = calloc(2 * npos * (nsample + 1), sizeof(float));
        float *scale = calloc(npos, sizeof(float));
        for (size_t pos = 0; pos < npos; pos++) {
            scale[pos] = expf(params[param_offset + pos * ldp + 1]);
        }
        score[batch] =
            squiggle_match_backward(signal + signal_offset, nsample,
                                    params + param_offset, ldp, scale, npos,
                                    prob_back, bwd);
        free(scale);
        free(bwd);
    }
}


void squiggle_match_viterbi_path(float const *signal, int32_t const *siglen,
                                 size_t nbatch, float const *params,
                                 size_t npos, float prob_back, float localpen,
                                 float minscore, int32_t * path, float *score) {
    size_t sigidx[nbatch];
    sigidx[0] = 0;
    for (size_t idx = 1; idx < nbatch; idx++) {
        sigidx[idx] = sigidx[idx - 1] + siglen[idx - 1];
    }

#pragma omp parallel for
    for (size_t batch = 0; batch < nbatch; batch++) {
        const size_t nsample = siglen[batch];
        const size_t signal_offset = sigidx[batch];
        const size_t param_offset = batch * nparam;
        const size_t ldp = nbatch * nparam;
        const size_t nstate = 2 * npos + 2;
        float *fwd = calloc(2 * nstate, sizeof(float));
        float *scale = calloc(npos, sizeof(float));
        for (size_t pos = 0; pos < npos; pos++) {
            scale[pos] = expf(params[param_offset + pos * ldp + 1]);
        }
        score[batch] =
            squiggle_match_viterbi(signal + signal_offset, nsample,
                                   params + param_offset, ldp, scale, npos,
                                   prob_back, localpen, minscore,
                                   path + signal_offset, fwd);
        free(scale);
        free(fwd);
    }
}



float squiggle_match_posterior(float const *signal, size_t nsample,
                               float const *params, size_t ldp,
                               float const *scale, size_t npos, float prob_back,
                               float *post) {
    const size_t nstate = 2 * npos;
    float *fwd = post;
    float *bwd = calloc(nstate * (nsample + 1), sizeof(float));
    float score =
        squiggle_match_forward(signal, nsample, params, ldp, scale, npos,
                               prob_back, fwd);
    squiggle_match_backward(signal, nsample, params, ldp, scale, npos,
                            prob_back, bwd);

    for (size_t sample = 1; sample <= nsample; sample++) {
        const size_t offset = sample * nstate;

        //  Normalised to form posteriors
        {
            for (size_t pos = 0; pos < nstate; pos++) {
                fwd[offset + pos] += bwd[offset + pos];
            }

            softmax_inplace(fwd + offset, nstate);
        }
    }
    free(bwd);

    return score;
}



void squiggle_match_grad(float const *signal, int32_t const *siglen,
                         size_t nbatch, float const *params, size_t npos,
                         float prob_back, float *grad) {
    size_t sigidx[nbatch];
    sigidx[0] = 0;
    for (size_t idx = 1; idx < nbatch; idx++) {
        sigidx[idx] = sigidx[idx - 1] + siglen[idx - 1];
    }

#pragma omp parallel for
    for (size_t batch = 0; batch < nbatch; batch++) {
        const size_t nsample = siglen[batch];
        const size_t signal_offset = sigidx[batch];
        const size_t param_offset = batch * nparam;
        const size_t ldp = nbatch * nparam;
        const size_t nstate = 2 * npos;
        float *fwd = calloc(nstate * (nsample + 1), sizeof(float));
        float *bwd = calloc(nstate * (nsample + 1), sizeof(float));
        float *scale = calloc(npos, sizeof(float));
        for (size_t pos = 0; pos < npos; pos++) {
            scale[pos] = expf(params[param_offset + pos * ldp + 1]);
        }
        squiggle_match_forward(signal + signal_offset, nsample,
                               params + param_offset, ldp, scale, npos,
                               prob_back, fwd);
        squiggle_match_backward(signal + signal_offset, nsample,
                                params + param_offset, ldp, scale, npos,
                                prob_back, bwd);


        for (size_t pos = 0; pos < npos; pos++) {
            grad[param_offset + pos * ldp + 0] = 0.0f;
            grad[param_offset + pos * ldp + 1] = 0.0f;
            grad[param_offset + pos * ldp + 2] = 0.0f;
        }

        for (size_t sample = 1; sample <= nsample; sample++) {
            const size_t offset = sample * nstate;
            const float sig = signal[signal_offset + sample - 1];

            //  Normalised to form posteriors
            float fact = fwd[offset] + bwd[offset];
            {
                for (size_t st = 1; st < nstate; st++) {
                    fact = logsumexp(fact, fwd[offset + st] + bwd[offset + st]);
                }
            }

            for (size_t pos = 0; pos < npos; pos++) {
                const float loc = params[param_offset + pos * ldp + 0];
                const float logsc = params[param_offset + pos * ldp + 1];
                const float prob_pos =
                    expf(fwd[offset + pos] + bwd[offset + pos] - fact);
                const float prob_posnpos =
                    expf(fwd[offset + npos + pos] + bwd[offset + npos + pos] -
                         fact);
                grad[param_offset + pos * ldp + 0] +=
                    (prob_pos + prob_posnpos) * dloglaplace_loc(sig, loc,
                                                                scale[pos],
                                                                logsc);
                grad[param_offset + pos * ldp + 1] +=
                    (prob_pos + prob_posnpos) * dloglaplace_logscale(sig, loc,
                                                                     scale[pos],
                                                                     logsc);
            }

            for (size_t pos = 0; pos < npos; pos++) {
                const float loc = params[param_offset + pos * ldp + 0];
                const float logsc = params[param_offset + pos * ldp + 1];
                const float logem = loglaplace(sig, loc, scale[pos], logsc);
                const float pprob_pos =
                    expf(fwd[offset - nstate + pos] + bwd[offset + pos] +
                         logem - fact);
                const float move_pen =
                    plogisticf(params[param_offset + pos * ldp + 2]);
                const float dlogisticf_move_pen =
                    (1.0f - prob_back) * move_pen * (1.0f - move_pen);
                grad[param_offset + pos * ldp + 2] -=
                    pprob_pos * dlogisticf_move_pen;
            }


            for (size_t pos = 1; pos < npos; pos++) {
                const float loc = params[param_offset + pos * ldp + 0];
                const float logsc = params[param_offset + pos * ldp + 1];
                const float logem = loglaplace(sig, loc, scale[pos], logsc);
                const float pprob_pos =
                    expf(fwd[offset - nstate + pos - 1] + bwd[offset + pos] +
                         logem - fact);
                const float move_pen =
                    plogisticf(params[param_offset + pos * ldp + 2]);
                const float dlogisticf_move_pen =
                    (1.0f - prob_back) * move_pen * (1.0f - move_pen);
                grad[param_offset + pos * ldp + 2] +=
                    pprob_pos * dlogisticf_move_pen;
            }
        }


        free(scale);
        free(bwd);
        free(fwd);
    }
}

#ifdef SQUIGGLE_TEST
const float test_signal[] = {
    1.0120153f, 1.0553021f, 10.0172595f, 10.0962240f, 10.0271495f,
    1.0117957f, 4.6153470f, 5.4212851f, 3.0914187f, 1.2078583f,
    1.5120153f, 1.4553021f, 3.6172595f, 3.8962240f, 3.9271495f,
    0.5117957f, 4.6153470f, 5.4212851f, 2.5914187f, 3.2078583f
};
const int32_t test_siglen[2] = { 10, 10 };

float test_param[30] = {
    // t = 0, b = 0
    1.0f, 0.0f, -1.0f,
    // t = 0, b = 1
    1.0f, 0.0f, -1.0f,
    // t = 1, b = 0
    10.0f, 0.0f, -2.0f,
    // t = 1, b = 1
    3.0f, 0.0f, -2.0f,
    // t = 2, b = 0
    1.0f, 0.0f, -1.5f,
    // t = 2, b = 1
    1.0f, 0.0f, -1.5f,
    // t = 3, b = 0
    5.0f, 0.0f, -0.5f,
    // t = 3, b = 1
    5.0f, 0.0f, -0.5f,
    // t = 4, b = 0
    3.0f, 0.0f, -1.0f,
    // t = 4, b = 1
    3.0f, 0.0f, -1.0f
};

#include <stdio.h>

int main(void) {
    const size_t npos = 5;
    const size_t nbatch = 2;
    float score[2] = { 0.0f };
    int32_t path[20] = { 0 };
    const float DELTA = 1e-3f;
    const float prob_back = 0.3f;
    const float localpen = 2000.0f;
    const float minscore = 12.0;
    const size_t msize = npos * nbatch * nparam;


    squiggle_match_scores_fwd(test_signal, test_siglen, nbatch, test_param,
                              npos, prob_back, score);
    printf("Forwards scores: %f %f\n", score[0], score[1]);

    squiggle_match_scores_bwd(test_signal, test_siglen, nbatch, test_param,
                              npos, prob_back, score);
    printf("Backwards scores: %f %f\n", score[0], score[1]);

    squiggle_match_viterbi_path(test_signal, test_siglen, nbatch, test_param,
                                npos, prob_back, localpen, minscore, path,
                                score);
    printf("Viterbi scores: %f %f\n", score[0], score[1]);
    size_t offset = 0;
    for (size_t batch = 0; batch < nbatch; batch++) {
        const size_t nsample = test_siglen[batch];
        for (size_t sample = 0; sample < nsample; sample++) {
            printf(" %d", path[offset + sample]);
        }
        fputc('\n', stdout);
        offset += nsample;
    }


    float *grad = calloc(msize, sizeof(float));
    squiggle_match_grad(test_signal, test_siglen, nbatch, test_param, npos,
                        prob_back, grad);
    float maxdelta = 0.0;
    for (size_t pos = 0; pos < npos; pos++) {
        const size_t offset = pos * nbatch * nparam;
        for (size_t st = 0; st < nparam; st++) {
            maxdelta =
                fmaxf(maxdelta,
                      fabsf(grad[offset + st] - grad[offset + nparam + st]));
        }
    }
    printf("Max grad delta = %f\n", maxdelta);

    printf("Derviatives:\n");
    float fscore[2] = { 0.0f };
    for (size_t pos = 0; pos < npos; pos++) {
        printf("  Pos %zu\n", pos);
        const size_t offset = pos * nbatch * nparam;
        for (size_t st = 0; st < nparam; st++) {
            // Positive difference
            const float orig = test_param[offset + st];
            test_param[offset + st] = orig + DELTA;
            squiggle_match_scores_fwd(test_signal, test_siglen, nbatch,
                                      test_param, npos, prob_back, score);
            fscore[0] = score[0];
            fscore[1] = score[1];
            // Negative difference
            test_param[offset + st] = orig - DELTA;
            squiggle_match_scores_fwd(test_signal, test_siglen, nbatch,
                                      test_param, npos, prob_back, score);
            fscore[0] = (fscore[0] - score[0]) / (2.0f * DELTA);
            fscore[1] = (fscore[1] - score[1]) / (2.0f * DELTA);
            // Report and reset
            test_param[offset + st] = orig;
            squiggle_match_scores_fwd(test_signal, test_siglen, nbatch,
                                      test_param, npos, prob_back, score);
            printf("    %f d=%f [%f %f] (%f %f)\n", grad[offset + st],
                   fabsf(grad[offset + st] - fscore[0]), fscore[0], fscore[1],
                   score[0], score[1]);

        }
    }
    free(grad);


    for (size_t pos = 0; pos < npos; pos++) {
        const size_t offset = pos * nbatch * nparam;
        for (size_t sample = 0; sample < test_siglen[0]; sample++) {
            const float loc = test_param[offset + 0];
            const float logsc = test_param[offset + 1];

            const float df =
                dloglaplace_logscale(test_signal[sample], loc, expf(logsc),
                                     logsc);
            const float dplus =
                loglaplace(test_signal[sample], loc, expf(logsc + DELTA),
                           logsc + DELTA);
            const float dminus =
                loglaplace(test_signal[sample], loc, expf(logsc - DELTA),
                           logsc - DELTA);
            const float approxdf = (dplus - dminus) / (2.0f * DELTA);
            printf("dlog/dloc = %f\t%f\t%f\n", df, approxdf,
                   fabsf(df - approxdf));
        }
    }
}
#endif
