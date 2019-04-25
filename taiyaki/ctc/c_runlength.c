#include <assert.h>
#include <err.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "c_runlength.h"

#define _PARAM_STAY_OFF 12
#define _PARAM_SHAPE_OFF 0
#define _PARAM_SCALE_OFF 4
#define _PARAM_CAT_OFF 8
#define _NBASE 4
#define LARGE_VAL 1e30f

static inline float powm1(float x, float y){
    return expm1(y*logf(x));
}


static inline float weibull_cdf(float x, float sh, float sc){
    return -expm1f(-powf(x / sc, sh));
}


static inline float weibull_comp_cdf(float x, float sh, float sc){
    return expf(-powf(x / sc, sh));
}

static inline float entropy(float x, float logx){
    return(x>0.0f) ? (x * logx) : 0.0f;
}


/**   Calculate logarithm of probability mass for the Discrete Weibull distribution
 *
 *    Optionally calculate derviatives WRT shape and scale.
 *
 *    Calculates logPMF = log [ cF(x) - cF(x+1) ] where cF(x) is the complementary CRF
 *    for the Weibull distribution.
 *
 *    log cF(x ; sh, sc) = -(x / sc)^sh
 *
 *    Optionally calculates d logPMF / d shape and  d logPMF / d scale.
 *
 *    @param x Value at which to calculate mass.
 *    @param sh Shape of the discrete Weibull distribution.
 *    @param sc Scale of the discrete Weibull distribution.
 *    @param dsh [out] Pointer of where to store derivative WRT shape; don't calculate if NULL.
 *    @param dsc [out] Pointer of where to store derivative WRT scale; don't calculate if NULL.
 *
 *    @returns logarithm of probability mass function.
 **/
static inline float discrete_weibull_logpmf(float x, float sh, float sc, float *dsh, float *dsc){
    const float MIN_PROB = 1e-8; 
    const float log_cprob1 = -powf(x / sc, sh);
    const float log_cprob2 = -powf((x + 1.0f) / sc, sh);
    const float delta_log_cprob = -log_cprob2 * powm1(x / (1.0f + x), sh);
    const float cprob1 = expf(log_cprob1);
    const float cprob2 = expf(log_cprob2);
    //const float entropy1 = entropy(cprob1, log_cprob1);
    //const float entropy2 = entropy(cprob2, log_cprob2);
    //const float cprob = cprob1 - cprob2;
    const float tmp =  -expm1f(delta_log_cprob);
    const float cprob = MIN_PROB + cprob1 * tmp;
    const float log_cprob = logf(cprob);

    if(NULL != dsh){
        // Derivative WRT shape
        // Using the identity dcF / dsh = d log cF / dsh * cF
        //const float dF1_dsh = (x == 0.0f) ? 0.0f : (logf(x / sc) * entropy1);
        //const float dF2_dsh = logf((x + 1.0f) / sc) * entropy2;
        //*dsh = (dF1_dsh - dF2_dsh) / cprob;
        const float f1 = (x == 0.0f) ? 0.0f : (logf(x / sc) * log_cprob1);
        const float f2 = logf((x+1.0f) / sc) * log_cprob2;
        *dsh = f2 + (f1 - f2) / tmp;
	if(!isfinite(*dsh)){
		errx(EXIT_FAILURE, "NAN created %s:%d -- x %f p %f sh %f dsh %f sc %f dsc %f\n", __FILE__, __LINE__, x, cprob, sh, *dsh, sc, *dsc);
	}
    }
    if(NULL != dsc){
        // Derivative WRT to scale
        // Using the identity dcF / dsc = d log cF / dsc * cF
        const float fact = - sh / sc;
        //const float dF1_dsc = fact * entropy1;
        //const float dF2_dsc = fact * entropy2;
        //*dsc = (dF1_dsc - dF2_dsc) / cprob;
        *dsc =  fact * (log_cprob2 - delta_log_cprob / tmp);
	if(!isfinite(*dsc)){
		errx(EXIT_FAILURE, "NAN created %s:%d  -- x %f p %f sh %f dsh %f sc %f dsc %f\n", __FILE__, __LINE__, x, cprob, sh, *dsh, sc, *dsc);
	}
    }

    if(!isfinite(log_cprob)){
         errx(EXIT_FAILURE, "NAN created %s:%d -- x %f p %f sh %f sc %f\n", __FILE__, __LINE__, x, log_cprob, sh, sc);
    }

    return log_cprob;
}


static inline float logsumexpf(float x, float y){
    return fmaxf(x, y) + log1pf(expf(-fabsf(x-y)));
}


float runlength_forward(float const * param, size_t nblk, size_t ldp, int32_t const * seq,
                        int32_t const * rle, size_t nseqpos, float * fwd){
    assert(nseqpos > 0);
    assert(NULL != param);
    assert(NULL != seq);
    assert(NULL != rle);
    assert(NULL != fwd);
    const size_t npos = nseqpos + 1;

    //  Point prior  -- must start in stay at beginning of sequence
    for(size_t pos=0 ; pos < npos ; pos++){
        fwd[pos] = -LARGE_VAL;
    }
    fwd[0] = 0.0;

    for(size_t blk=0 ; blk < nblk ; blk++){
        const size_t fwd_prev_off = blk * npos;
        const size_t fwd_curr_off = (blk + 1) * npos;
        const size_t post_curr_off = blk * ldp;

        //  Stay in initial state.  Arbitrarily assume 'A'
        fwd[fwd_curr_off] = fwd[fwd_prev_off] + param[post_curr_off + _PARAM_STAY_OFF + 0];
        for(size_t pos=1 ; pos < npos ; pos++){
            //  Stay in state.  nseqpos possible stay positions
            fwd[fwd_curr_off + pos] = fwd[fwd_prev_off + pos]
                                    + param[post_curr_off + _PARAM_STAY_OFF + seq[pos - 1]];
        }

        for(size_t pos=0 ; pos < nseqpos ; pos++){
            //  Step to new state
            const size_t destst = pos + 1;
            const size_t base1 = seq[pos];
            const size_t len1 = rle[pos];
            const size_t poff = post_curr_off + base1;
            const float step_score = fwd[fwd_prev_off + pos]
                                   + param[poff + _PARAM_CAT_OFF]
                                   + discrete_weibull_logpmf(len1 - 1,
                                                             param[poff + _PARAM_SHAPE_OFF],
                                                             param[poff + _PARAM_SCALE_OFF],
                                                             NULL, NULL);
            fwd[fwd_curr_off + destst] = logsumexpf(fwd[fwd_curr_off + destst], step_score);
        }
    }

    // Final score is sum of final state + its stay
    float score = fwd[nblk * npos + npos - 1];
    return score;
}


float runlength_backward(float const * param, size_t nblk, size_t ldp, int32_t const * seq,
                         int32_t const * rle, size_t nseqpos, float * bwd){
    assert(nseqpos > 0);
    assert(NULL != param);
    assert(NULL != seq);
    assert(NULL != rle);
    assert(NULL != bwd);
    const size_t npos = nseqpos + 1;


    //  Point prior -- must have ended in either final stay or state
    for(size_t pos=0 ; pos < npos ; pos++){
        bwd[nblk * npos + pos] = -LARGE_VAL;
    }
    // Final stay
    bwd[nblk * npos + npos - 1] = 0.0;

    for(size_t blk=nblk ; blk > 0 ; blk--){
        const size_t bwd_prev_off = blk * npos;
        const size_t bwd_curr_off = (blk - 1) * npos;
        const size_t post_curr_off = (blk - 1) * ldp;

        bwd[bwd_curr_off] = bwd[bwd_prev_off] + param[post_curr_off + _PARAM_STAY_OFF];
        for(size_t pos=1 ; pos < npos ; pos++){
            //  Remain in stay state.  nseqpos possible stay positions
            bwd[bwd_curr_off + pos] = bwd[bwd_prev_off + pos]
                                    + param[post_curr_off + _PARAM_STAY_OFF + seq[pos-1]];
        }

        for(size_t pos=0 ; pos < nseqpos ; pos++){
            //  Step -- must have come from non-stay
            const size_t origst = pos + 1;
            const size_t base1 = seq[pos];
            const size_t len1 = rle[pos];
            const size_t poff = post_curr_off + base1;
            const float step_score = bwd[bwd_prev_off + origst]
                                   + param[poff + _PARAM_CAT_OFF]
                                   + discrete_weibull_logpmf(len1 - 1,
                                                             param[poff + _PARAM_SHAPE_OFF],
                                                             param[poff + _PARAM_SCALE_OFF],
                                                             NULL, NULL);
            bwd[bwd_curr_off + pos] = logsumexpf(bwd[bwd_curr_off + pos], step_score);
        }
    }

    return bwd[0];
}


void runlength_cost(float const * param, size_t nstate, size_t nblk , size_t nbatch,
              int32_t const * seqs, int32_t const * rles, int32_t const * seqlen, float * score){
    size_t ldp = nbatch * nstate;
    size_t seqidx[nbatch];
    seqidx[0] = 0;
    for(size_t idx=1 ; idx < nbatch ; idx++){
        seqidx[idx] = seqidx[idx - 1] + seqlen[idx - 1];
    }

#pragma omp parallel for
    for(size_t batch=0 ; batch < nbatch ; batch++){
       if(0 == seqlen[batch]){
           score[batch] = 0.0;
           continue;
        }
        const size_t offset = batch * nstate;
        float * fwd = calloc((1 + nblk) * (1 + seqlen[batch]), sizeof(float));
        score[batch] = runlength_forward(param + offset, nblk, ldp, seqs + seqidx[batch],
                                         rles + seqidx[batch], seqlen[batch], fwd);
        free(fwd);
    }
}


void runlength_scores_fwd(float const * param, size_t nstate, size_t nblk , size_t nbatch,
                          int32_t const * seqs, int32_t const * rles, int32_t const * seqlen,
                          float * score){
    runlength_cost(param, nstate, nblk, nbatch, seqs, rles, seqlen, score);
}


void runlength_scores_bwd(float const * param, size_t nstate, size_t nblk , size_t nbatch,
                          int32_t const * seqs, int32_t const * rles, int32_t const * seqlen,
                          float * score){
    size_t ldp = nbatch * nstate;
    size_t seqidx[nbatch];
    seqidx[0] = 0;
    for(size_t idx=1 ; idx < nbatch ; idx++){
        seqidx[idx] = seqidx[idx - 1] + seqlen[idx - 1];
    }

#pragma omp parallel for
    for(size_t batch=0 ; batch < nbatch ; batch++){
       if(0 == seqlen[batch]){
           score[batch] = 0.0;
           continue;
        }
        const size_t offset = batch * nstate;
        float * bwd = calloc((1 + nblk) * (1 + seqlen[batch]), sizeof(float));
        score[batch] = runlength_backward(param + offset, nblk, ldp, seqs + seqidx[batch],
                                          rles + seqidx[batch], seqlen[batch], bwd);
        free(bwd);
    }
}


void runlength_grad(float const * param, size_t nstate, size_t nblk , size_t nbatch,
                    int32_t const * seqs, int32_t const * rles, int32_t const * seqlen,
                    float * score, float * grad){
    const size_t ldp = nbatch * nstate;

    size_t seqidx[nbatch];
    seqidx[0] = 0;
    for(size_t idx=1 ; idx < nbatch ; idx++){
        seqidx[idx] = seqidx[idx - 1] + seqlen[idx - 1];
    }

#pragma omp parallel for
    for(size_t batch=0 ; batch < nbatch ; batch++){
       if(0 == seqlen[batch]){
           continue;
        }
        const size_t batch_offset = batch * nstate;
        const int32_t nseqpos = seqlen[batch];
        const int32_t npos = 1 + nseqpos;
        int32_t const * seq = seqs + seqidx[batch];
        int32_t const * rle = rles + seqidx[batch];
        float * fwd = calloc((nblk + 1) * npos, sizeof(float));
        float * bwd = calloc((nblk + 1) * npos, sizeof(float));
        score[batch] = runlength_forward(param + batch_offset, nblk, ldp, seq, rle, nseqpos, fwd);
        runlength_backward(param + batch_offset, nblk, ldp, seq, rle, nseqpos, bwd);

        // Normalised transition matrix
        for(size_t blk=0 ; blk < nblk ; blk++){
            const size_t foffset = blk * npos;
            const size_t boffset = blk * npos + npos;
            const size_t goffset = batch_offset + blk * nbatch * nstate;
            // Make sure gradient calc is zero'd
            memset(grad + goffset, 0, nstate * sizeof(float));

            //  Calculate log-score should be constant.
            //  Recalculate close to position to reduce numerical error
            float fact = fwd[foffset] + bwd[foffset];
            for(size_t pos=1; pos < npos ; pos++){
                fact = logsumexpf(fact, fwd[foffset + pos] + bwd[foffset + pos]);
            }

            grad[goffset + _PARAM_STAY_OFF] += expf((fwd[foffset] + bwd[boffset]
                                             + param[goffset + _PARAM_STAY_OFF] - fact));
            for(size_t pos=1 ; pos < npos ; pos++){
                // Remain in stay state
                grad[goffset + _PARAM_STAY_OFF + seq[pos-1]] += expf(( fwd[foffset + pos] + bwd[boffset + pos]
                                                              + param[goffset + _PARAM_STAY_OFF + seq[pos-1]] - fact));
            }
            for(size_t pos=0 ; pos < nseqpos ; pos++){
                // Steps
                const size_t base1 = seq[pos];
                const float param_sh = param[goffset + base1 + _PARAM_SHAPE_OFF];
                const float param_sc = param[goffset + base1 + _PARAM_SCALE_OFF];

                float dsh=NAN, dsc=NAN;
                const float logpmf = discrete_weibull_logpmf(rle[pos] - 1, param_sh, param_sc, &dsh, &dsc);
		if(! (isfinite(logpmf) && isfinite(dsh) && isfinite(dsc))){
			warnx("NAN created %s:%d -- pos %zu x %d p %f sh %f dsh %f sc %f dsc %f\n", __FILE__, __LINE__, pos, rle[pos]-1, logpmf, param_sh, dsh, param_sc, dsc);
		}

                const float logdscore0 = fwd[foffset + pos]
                                       + bwd[boffset + pos + 1]
                                       + param[goffset + base1 + _PARAM_CAT_OFF]
                                       + logpmf;
                const float dscore0 = expf(logdscore0 - fact);
		if(!isfinite(dscore0)){
			warnx( "NAN pos %zu logdscore0 %f fact %f\n", pos, logdscore0, fact);
		}


                grad[goffset + base1 + _PARAM_CAT_OFF] += dscore0;
                grad[goffset + base1 + _PARAM_SHAPE_OFF] += dscore0 * dsh;
                grad[goffset + base1 + _PARAM_SCALE_OFF] += dscore0 * dsc;
            }
        }

        free(bwd);
        free(fwd);
    }
}




#ifdef RUNLENGTH_TEST

const int32_t test_seq1[8] = {0, 1, 3, 2,
                               0, 1, 3, 2};
const int32_t test_rle1[8] = {1, 1, 1, 2,
                              1, 1, 1, 2};

const int32_t test_seqlen1[2] = {4, 4};

float test_param1[320] = {
    // t = 0, blk = 0  -- Emit 0
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.77797989, 0.04829781, 0.06430573, 0.05602876,
    0.019834005, 0.010122076, 0.008180559, 0.015251180,
    // t = 0, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.77797989, 0.04829781, 0.06430573, 0.05602876,
    0.019834005, 0.010122076, 0.008180559, 0.015251180,
    // t = 1, blk = 0 -- Stay 0
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.03862139, 0.09763279, 0.04683259, 0.06607721,
    0.62430426, 0.04096683, 0.06079292, 0.02477199,
    // t = 1, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.03862139, 0.09763279, 0.04683259, 0.06607721,
    0.62430426, 0.04096683, 0.06079292, 0.02477199,
    // t = 2, blk = 0  -- Emit 1 or stay 0
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.04860824, 0.43823059, 0.11629226, 0.04543547,
    0.4002689578, 0.0097434277, 0.0201715293, 0.0212495252,
    // t = 2, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.04860824, 0.43823059, 0.11629226, 0.04543547,
    0.4002689578, 0.0097434277, 0.0201715293, 0.0212495252,
    // t = 3, blk = 0  -- Emit 1 or stay (0, 1)
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.03667420, 0.45438334, 0.08784141, 0.07171496,
    0.151437577, 0.005969148, 0.275324347, 0.016655017,
    // t = 3, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.03667420, 0.45438334, 0.08784141, 0.07171496,
    0.151437577, 0.005969148, 0.275324347, 0.016655017,
    // t = 4, blk = 0  -- Emit 3 or stay 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.02934134, 0.08286614, 0.06488921, 0.45799906,
    0.003911363, 0.424935206, 0.021216121, 0.014841561,
    // t = 4, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.02934134, 0.08286614, 0.06488921, 0.45799906,
    0.003911363, 0.424935206, 0.021216121, 0.014841561,
    // t = 5, blk = 0  -- Stay 3
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.08495230, 0.04668598, 0.02651227, 0.08040296,
    0.04208008, 0.06269254, 0.00657268, 0.65010119,
    // t = 5, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.08495230, 0.04668598, 0.02651227, 0.08040296,
    0.04208008, 0.06269254, 0.00657268, 0.65010119,
    // t = 6, blk = 0 -- Emit 2
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.05060846, 0.05394262, 0.74161697, 0.07701774,
    0.016003002, 0.007056044, 0.031690709, 0.022064455,
    // t = 6, blk = 1
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.05060846, 0.05394262, 0.74161697, 0.07701774,
    0.016003002, 0.007056044, 0.031690709, 0.022064455,
    // t = 7, blk = 0 --  Stay 2
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.08495230, 0.04668598, 0.02651227, 0.08040296,
    0.00351678, 0.03321462, 0.62785557, 0.09685952,
    // t = 7, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.08495230, 0.04668598, 0.02651227, 0.08040296,
    0.00351678, 0.03321462, 0.62785557, 0.09685952,
    // t = 8, blk = 0  -- Emit 2 or stay 2
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.08495230, 0.04668598, 0.42651227, 0.08040296,
    0.001212893, 0.023044454, 0.418378963, 0.018810179,
    // t = 8, blk = 1
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.08495230, 0.04668598, 0.42651227, 0.08040296,
    0.001212893, 0.023044454, 0.418378963, 0.018810179,
    // t = 9, blk = 0  -- Emit 2 or stay 2
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.08495230, 0.04668598, 0.52651227, 0.08040296,
    0.037403703, 0.010355229, 0.307103949, 0.006583609,
    // t = 9, blk = 1
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.08495230, 0.04668598, 0.52651227, 0.08040296,
    0.037403703, 0.010355229, 0.307103949, 0.006583609
};



#include <stdio.h>

int main(int argc, char * argv[]){

    const size_t nblk = 10;
    const size_t nstate = 16;
    const size_t nbatch = 2;
    float score[2] = {0.0f};
    const float DELTA = 1e-4f;
    const size_t msize = nblk * nstate * nbatch;

    for(size_t i=0 ; i < nblk * nbatch ; i++){
        const size_t offset = i * nstate;
        for(size_t j=_PARAM_CAT_OFF ; j < nstate ; j++){
            test_param1[offset + j] = logf(test_param1[offset + j]);
        }
    }
    for(size_t i=0 ; i < nblk * nbatch ; i++){
        for(size_t j=0 ; j < 4 ; j++){
            const size_t offset = i * nstate + j;
            printf("%f %f %f %f : %f %f %f %f\n",
                   test_param1[offset + _PARAM_CAT_OFF],
                   test_param1[offset + _PARAM_STAY_OFF],
                   test_param1[offset + _PARAM_SHAPE_OFF],
                   test_param1[offset + _PARAM_SCALE_OFF],
                   expf(discrete_weibull_logpmf(0,
                                           test_param1[offset + _PARAM_SHAPE_OFF],
                                           test_param1[offset + _PARAM_SCALE_OFF],
                                           NULL, NULL)),
                   expf(discrete_weibull_logpmf(1,
                                           test_param1[offset + _PARAM_SHAPE_OFF],
                                           test_param1[offset + _PARAM_SCALE_OFF],
                                           NULL, NULL)),
                   expf(discrete_weibull_logpmf(2,
                                           test_param1[offset + _PARAM_SHAPE_OFF],
                                           test_param1[offset + _PARAM_SCALE_OFF],
                                           NULL, NULL)),
                   expf(discrete_weibull_logpmf(3,
                                           test_param1[offset + _PARAM_SHAPE_OFF],
                                           test_param1[offset + _PARAM_SCALE_OFF],
                                           NULL, NULL)));
	    float dsh = NAN, dsc = NAN;
	    const float logpmf0 = discrete_weibull_logpmf(0,
			                                  test_param1[offset + _PARAM_SHAPE_OFF],
							  test_param1[offset + _PARAM_SCALE_OFF],
							  &dsh, &dsc);
	    const float orig_sh = test_param1[offset + _PARAM_SHAPE_OFF];
	    test_param1[offset + _PARAM_SHAPE_OFF] += DELTA;
	    const float logpmf0_psh = discrete_weibull_logpmf(0,
			                                  test_param1[offset + _PARAM_SHAPE_OFF],
							  test_param1[offset + _PARAM_SCALE_OFF],
							  NULL, NULL);
	    test_param1[offset + _PARAM_SHAPE_OFF] = orig_sh - DELTA;
	    const float logpmf0_msh = discrete_weibull_logpmf(0,
			                                  test_param1[offset + _PARAM_SHAPE_OFF],
							  test_param1[offset + _PARAM_SCALE_OFF],
							  NULL, NULL);
	    test_param1[offset + _PARAM_SHAPE_OFF] = orig_sh;

	    const float orig_sc = test_param1[offset + _PARAM_SCALE_OFF];
	    test_param1[offset + _PARAM_SCALE_OFF] += DELTA;
	    const float logpmf0_psc = discrete_weibull_logpmf(0,
			                                  test_param1[offset + _PARAM_SHAPE_OFF],
							  test_param1[offset + _PARAM_SCALE_OFF],
							  NULL, NULL);
	    test_param1[offset + _PARAM_SCALE_OFF] = orig_sc - DELTA;
	    const float logpmf0_msc = discrete_weibull_logpmf(0,
			                                  test_param1[offset + _PARAM_SHAPE_OFF],
							  test_param1[offset + _PARAM_SCALE_OFF],
							  NULL, NULL);
	    printf("dsh %f  app. %f\n", dsh, 0.5 * (logpmf0_psh - logpmf0_msh) / DELTA);
	    printf("dsc %f  app. %f\n", dsc, 0.5 * (logpmf0_psc - logpmf0_msc) / DELTA);
        }
    }




    //
    //    F / B calculations
    //
    runlength_scores_fwd(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
    printf("Forwards scores: %f %f\n", score[0], score[1]);

    runlength_scores_bwd(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
    printf("Backwards scores: %f %f\n", score[0], score[1]);

    //runlength_viterbi_cost(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
    //printf("Viterbi scores: %f %f\n", score[0], score[1]);

    float * grad = calloc(msize, sizeof(float));
    runlength_grad(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, grad);
    float maxdelta = 0.0;
    for(size_t blk=0 ; blk < nblk ; blk++){
        const size_t offset = blk * nbatch * nstate;
        for(size_t st=0 ; st < nstate ; st++){
            maxdelta = fmaxf(maxdelta, fabsf(grad[offset + st] - grad[offset + nstate + st]));
        }
    }
    printf("Max grad delta = %f\n", maxdelta);

    printf("Derviatives:\n");
    float fscore[2] = {0.0f};
    for(size_t blk=0 ; blk < nblk ; blk++){
        printf("  Block %zu\n", blk);
        const size_t offset = blk * nbatch * nstate;
        for(size_t st=0 ; st < nstate ; st++){
            // Positive difference
            const float orig = test_param1[offset + st];
            test_param1[offset + st] = orig + DELTA;
            runlength_scores_fwd(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
            fscore[0] = score[0];
            fscore[1] = score[1];
            // Negative difference
            test_param1[offset + st] = orig - DELTA;
            runlength_scores_fwd(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
            fscore[0] = (fscore[0] - score[0]) / (2.0f * DELTA);
            fscore[1] = (fscore[1] - score[1]) / (2.0f * DELTA);
            // Report and reset
            test_param1[offset + st] = orig;
            printf("    %zu :  %f %f d=%f r=%f [%f %f]\n", st, fscore[0], grad[offset + st], fabsf(grad[offset + st] - fscore[0]), grad[offset + st] / fscore[0], fscore[0], fscore[1]);
        }
    }

    //
    //   Viterbi calculations
    //
    /*
    memset(grad, 0, msize * sizeof(float));
    runlength_viterbi_grad(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, grad);
    maxdelta = 0.0;
    for(size_t blk=0 ; blk < nblk ; blk++){
        const size_t offset = blk * nbatch * nstate;
        for(size_t st=0 ; st < nstate ; st++){
            maxdelta = fmaxf(maxdelta, fabsf(grad[offset + st] - grad[offset + nstate + st]));
        }
    }
    printf("Max grad delta = %f\n", maxdelta);

    printf("Derviatives:\n");
    for(size_t blk=0 ; blk < nblk ; blk++){
        printf("  Block %zu\n", blk);
        const size_t offset = blk * nbatch * nstate;
        for(size_t st=0 ; st < nstate ; st++){
            // Positive difference
            const float orig = test_param1[offset + st];
            test_param1[offset + st] = orig + DELTA;
            runlength_viterbi_cost(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
            fscore[0] = score[0];
            fscore[1] = score[1];
            // Negative difference
            test_param1[offset + st] = orig - DELTA;
            runlength_viterbi_cost(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
            fscore[0] = (fscore[0] - score[0]) / (2.0f * DELTA);
            fscore[1] = (fscore[1] - score[1]) / (2.0f * DELTA);
            // Report and reset
            test_param1[offset + st] = orig;
            printf("    %f d=%f r=%f [%f %f]\n", grad[offset + st], fabsf(grad[offset + st] - fscore[0]), grad[offset + st] / fscore[0], fscore[0], fscore[1]);
        }
    }
    */
}
#endif /* RUNLENGTH_TEST */
