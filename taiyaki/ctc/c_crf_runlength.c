#include <assert.h>
#include <err.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "c_crf_runlength.h"

#define _PARAM_SHAPE_OFF 0
#define _PARAM_SCALE_OFF 4
#define _PARAM_TRANS_OFF 8
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

static inline size_t rle_index(size_t base_from, bool stay_from, size_t base_to, bool stay_to){
	assert(stay_to ^ (base_from != base_to));
	return _PARAM_TRANS_OFF + base_to * 2 * _NBASE + base_from + (stay_from ? _NBASE : 0);
}


float crf_runlength_forward(float const * param, size_t nblk, size_t ldp, int32_t const * seq,
                            int32_t const * rle, size_t nseqpos, float * fwd){
    assert(nseqpos > 0);
    assert(NULL != param);
    assert(NULL != seq);
    assert(NULL != rle);
    assert(NULL != fwd);
    const size_t npos = nseqpos + nseqpos;

    //  Point prior  -- must start in first position
    for(size_t pos=0 ; pos < npos ; pos++){
        fwd[pos] = -LARGE_VAL;
    }
    fwd[0] = 0.0;

    for(size_t blk=0 ; blk < nblk ; blk++){
        const size_t fwd_prev_off = blk * npos;
        const size_t fwd_curr_off = (blk + 1) * npos;
        const size_t post_curr_off = blk * ldp;

        //  Stay in first position
        for(size_t pos=0 ; pos < nseqpos ; pos++){
	    const size_t base = seq[pos];
            //  Stay in state -- just moved
            const float stay_from_move = fwd[fwd_prev_off + pos]
                                       + param[post_curr_off + rle_index(base, false, base, true)];
	    //  Stay in state -- already staying
            const float stay_from_stay = fwd[fwd_prev_off + nseqpos + pos]
                                       + param[post_curr_off + rle_index(base, true, base, true)];
	    fwd[fwd_curr_off + nseqpos + pos] = logsumexpf(stay_from_stay, stay_from_move);
        }

	fwd[fwd_curr_off] = -LARGE_VAL;
        for(size_t pos=1 ; pos < nseqpos ; pos++){
            //  Move to new position
            const size_t base_from = seq[pos - 1];
            const size_t base_to = seq[pos];

            const size_t len1 = rle[pos];
            const float move_from_move = fwd[fwd_prev_off + pos - 1]
                                       + param[post_curr_off + rle_index(base_from, false, base_to, false)];
            const float move_from_stay = fwd[fwd_prev_off + nseqpos + pos - 1]
                                       + param[post_curr_off + rle_index(base_from, true, base_to, false)];
	    const float run_score = discrete_weibull_logpmf(len1 - 1,
                                                            param[post_curr_off + _PARAM_SHAPE_OFF + base_to],
                                                            param[post_curr_off + _PARAM_SCALE_OFF + base_to],
                                                            NULL, NULL);
            fwd[fwd_curr_off + pos] = run_score + logsumexpf(move_from_move, move_from_stay);
        }
    }

    // Final score is sum of final state + its stay
    float score = logsumexpf(fwd[nblk * npos + nseqpos - 1],
		             fwd[nblk * npos + nseqpos + nseqpos - 1]);
    return score;
}


float crf_runlength_backward(float const * param, size_t nblk, size_t ldp, int32_t const * seq,
                         int32_t const * rle, size_t nseqpos, float * bwd){
    assert(nseqpos > 0);
    assert(NULL != param);
    assert(NULL != seq);
    assert(NULL != rle);
    assert(NULL != bwd);
    const size_t npos = nseqpos + nseqpos;


    //  Point prior -- must have ended in either final stay or state
    for(size_t pos=0 ; pos < npos ; pos++){
        bwd[nblk * npos + pos] = -LARGE_VAL;
    }
    // Final state and stay
    bwd[nblk * npos + nseqpos - 1] = 0.0;
    bwd[nblk * npos + npos - 1] = 0.0;

    for(size_t blk=nblk ; blk > 0 ; blk--){
        const size_t bwd_prev_off = blk * npos;
        const size_t bwd_curr_off = (blk - 1) * npos;
        const size_t post_curr_off = (blk - 1) * ldp;

	// Remained in stay at beginning of sequence
        bwd[bwd_curr_off + nseqpos] = bwd[bwd_prev_off + nseqpos] + param[post_curr_off + rle_index(0, true, 0, true)];
        for(size_t pos=0 ; pos < nseqpos ; pos++){
	    const size_t base = seq[pos];
            //  Remain in stay state -- just moved
            bwd[bwd_curr_off + pos] = bwd[bwd_prev_off + nseqpos + pos]
                                    + param[post_curr_off + rle_index(base, false, base, true)];
            //  Remain in stay state -- from stay
            bwd[bwd_curr_off + nseqpos + pos] = bwd[bwd_prev_off + nseqpos + pos]
                                              + param[post_curr_off + rle_index(base, true, base, true)];

        }

        for(size_t pos=1 ; pos < nseqpos ; pos++){
            //  Move  -- must go to non-stay
            const size_t base_from = seq[pos - 1];
            const size_t base_to = seq[pos];
            const size_t len1 = rle[pos];
            const float move_from_move = bwd[bwd_prev_off + pos]
                                       + param[post_curr_off + rle_index(base_from, false, base_to, false)];
            const float move_from_stay = bwd[bwd_prev_off + pos]
                                       + param[post_curr_off + rle_index(base_from, true, base_to, false)];
            const float run_score = discrete_weibull_logpmf(len1 - 1,
                                                            param[post_curr_off + _PARAM_SHAPE_OFF + base_to],
                                                            param[post_curr_off + _PARAM_SCALE_OFF + base_to],
                                                            NULL, NULL);
            bwd[bwd_curr_off + pos - 1] = logsumexpf(bwd[bwd_curr_off + pos - 1],
			                             move_from_move + run_score);
            bwd[bwd_curr_off + nseqpos + pos - 1] = logsumexpf(bwd[bwd_curr_off + nseqpos + pos - 1],
			                                       move_from_stay + run_score);
        }
    }

    //  Must start in first stay
    return bwd[0];
}


void crf_runlength_cost(float const * param, size_t nstate, size_t nblk , size_t nbatch,
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
        float * fwd = calloc((1 + nblk) * (2 * seqlen[batch]), sizeof(float));
        score[batch] = crf_runlength_forward(param + offset, nblk, ldp, seqs + seqidx[batch],
                                         rles + seqidx[batch], seqlen[batch], fwd);
        free(fwd);
    }
}


void crf_runlength_scores_fwd(float const * param, size_t nstate, size_t nblk , size_t nbatch,
                          int32_t const * seqs, int32_t const * rles, int32_t const * seqlen,
                          float * score){
    crf_runlength_cost(param, nstate, nblk, nbatch, seqs, rles, seqlen, score);
}


void crf_runlength_scores_bwd(float const * param, size_t nstate, size_t nblk , size_t nbatch,
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
        float * bwd = calloc((1 + nblk) * (2 * seqlen[batch]), sizeof(float));
        score[batch] = crf_runlength_backward(param + offset, nblk, ldp, seqs + seqidx[batch],
                                               rles + seqidx[batch], seqlen[batch], bwd);
        free(bwd);
    }
}


void crf_runlength_grad(float const * param, size_t nstate, size_t nblk , size_t nbatch,
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
        const int32_t npos = nseqpos + nseqpos;
        int32_t const * seq = seqs + seqidx[batch];
        int32_t const * rle = rles + seqidx[batch];
        float * fwd = calloc((nblk + 1) * npos, sizeof(float));
        float * bwd = calloc((nblk + 1) * npos, sizeof(float));
        score[batch] = crf_runlength_forward(param + batch_offset, nblk, ldp, seq, rle, nseqpos, fwd);
        crf_runlength_backward(param + batch_offset, nblk, ldp, seq, rle, nseqpos, bwd);

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

            for(size_t pos=0 ; pos < nseqpos ; pos++){
		const size_t base = seq[pos];
                // Remain in stay state from stay
		const size_t idx_stay = rle_index(base, true, base, true);
                grad[goffset + idx_stay] += expf( fwd[foffset + nseqpos + pos]
				                + bwd[boffset + nseqpos + pos]
                                                + param[goffset + idx_stay]
					       	- fact);
                // Remain in stay state from move
		const size_t idx_move = rle_index(base, false, base, true);
                grad[goffset + idx_move] += expf( fwd[foffset + pos]
				                + bwd[boffset + nseqpos + pos]
                                                + param[goffset + idx_move]
					       	- fact);
            }

            for(size_t pos=1 ; pos < nseqpos ; pos++){
                // Steps
                const size_t base_from = seq[pos - 1];
                const size_t base_to = seq[pos];
                const size_t idx_move = rle_index(base_from, false, base_to, false);
                const size_t idx_stay = rle_index(base_from, true, base_to, false);
                const float param_sh = param[goffset + base_to + _PARAM_SHAPE_OFF];
                const float param_sc = param[goffset + base_to + _PARAM_SCALE_OFF];

                float dsh=NAN, dsc=NAN;
                const float logpmf = discrete_weibull_logpmf(rle[pos] - 1, param_sh, param_sc, &dsh, &dsc);
		if(! (isfinite(logpmf) && isfinite(dsh) && isfinite(dsc))){
			warnx("NAN created %s:%d -- pos %zu x %d p %f sh %f dsh %f sc %f dsc %f\n", __FILE__, __LINE__, pos, rle[pos]-1, logpmf, param_sh, dsh, param_sc, dsc);
		}

                const float logdscore_stay = fwd[foffset + nseqpos + pos - 1]
                                           + bwd[boffset + pos]
                                           + param[goffset + idx_stay]
                                           + logpmf;
                const float dscore_stay = expf(logdscore_stay - fact);
		if(!isfinite(dscore_stay)){
			warnx( "NAN pos %zu logdscore_stay %f fact %f\n", pos, logdscore_stay, fact);
		}

                const float logdscore_move = fwd[foffset + pos - 1]
                                           + bwd[boffset + pos]
                                           + param[goffset + idx_move]
                                           + logpmf;
                const float dscore_move = expf(logdscore_move - fact);
		if(!isfinite(dscore_move)){
			warnx( "NAN pos %zu logdscore_move %f fact %f\n", pos, logdscore_move, fact);
		}


                grad[goffset + idx_stay] += dscore_stay;
                grad[goffset + base_to + _PARAM_SHAPE_OFF] += dscore_stay * dsh;
                grad[goffset + base_to + _PARAM_SCALE_OFF] += dscore_stay * dsc;

                grad[goffset + idx_move] += dscore_move;
                grad[goffset + base_to + _PARAM_SHAPE_OFF] += dscore_move * dsh;
                grad[goffset + base_to + _PARAM_SCALE_OFF] += dscore_move * dsc;
            }
        }

        free(bwd);
        free(fwd);
    }
}




#ifdef RUNLENGTH_TEST

const int32_t test_seq1[10] = {1, 0, 1, 3, 2,
                               1, 0, 1, 3, 2};
const int32_t test_rle1[10] = {1, 1, 1, 1, 2,
                              1, 1, 1, 1, 2};

const int32_t test_seqlen1[2] = {5, 5};

float test_param1[800] = {
    // t = 0, blk = 0  -- Emit 0 from 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0024588857, 0.0039163350, 0.0057933686, 0.0048475368,
    0.0006674938, 0.7055353623, 0.0062614771, 0.0037594293,
    0.0113568228, 0.0098524701, 0.0263307091, 0.0084061846,
    0.0043773681, 0.0465025831, 0.0082970864, 0.0033361490,
    0.0072847195, 0.0108070260, 0.0038091278, 0.0099386691,
    0.0129666947, 0.0100539178, 0.0015341372, 0.0193237322,
    0.0060240244, 0.0191505084, 0.0086507200, 0.0022742978,
    0.0008024779, 0.0030336297, 0.0311153858, 0.0015316701,
    // t = 0, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0024588857, 0.0039163350, 0.0057933686, 0.0048475368,
    0.0006674938, 0.7055353623, 0.0062614771, 0.0037594293,
    0.0113568228, 0.0098524701, 0.0263307091, 0.0084061846,
    0.0043773681, 0.0465025831, 0.0082970864, 0.0033361490,
    0.0072847195, 0.0108070260, 0.0038091278, 0.0099386691,
    0.0129666947, 0.0100539178, 0.0015341372, 0.0193237322,
    0.0060240244, 0.0191505084, 0.0086507200, 0.0022742978,
    0.0008024779, 0.0030336297, 0.0311153858, 0.0015316701,
    // t = 1, blk = 0 -- Stay 0 from move
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.7074173385, 0.0232972817, 0.0037748260, 0.0016217470,
    0.0133754462, 0.0176856453, 0.0076132325, 0.0081847766,
    0.0044044748, 0.0055975362, 0.0267850269, 0.0020387869,
    0.0047802115, 0.0187768312, 0.0225954327, 0.0107832135,
    0.0194320778, 0.0104842712, 0.0131892496, 0.0164179994,
    0.0092047769, 0.0097449631, 0.0057158597, 0.0017472798,
    0.0049074040, 0.0019444302, 0.0008472454, 0.0004441373,
    0.0165442142, 0.0003053081, 0.0015504522, 0.0087885237,
    // t = 1, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.7074173385, 0.0232972817, 0.0037748260, 0.0016217470,
    0.0133754462, 0.0176856453, 0.0076132325, 0.0081847766,
    0.0044044748, 0.0055975362, 0.0267850269, 0.0020387869,
    0.0047802115, 0.0187768312, 0.0225954327, 0.0107832135,
    0.0194320778, 0.0104842712, 0.0131892496, 0.0164179994,
    0.0092047769, 0.0097449631, 0.0057158597, 0.0017472798,
    0.0049074040, 0.0019444302, 0.0008472454, 0.0004441373,
    0.0165442142, 0.0003053081, 0.0015504522, 0.0087885237,
    // t = 2, blk = 0  -- Emit 1 from stay 0 or stay 0 (from stay)
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0007496803, 0.0050309000, 0.0067914625, 0.0077990403,
    0.3517043579, 0.0024265405, 0.0003516180, 0.0009002021,
    0.0225241891, 0.0122288018, 0.0027623691, 0.0235128312,
    0.3683247098, 0.157970713, 0.0105214792, 0.0138495624,
    0.0128591924, 0.0286768654, 0.0232649871, 0.0012800672,
    0.0014079208, 0.0112251786, 0.0134895968, 0.0019244475,
    0.0043541516, 0.0148757305, 0.0016573046, 0.0095518399,
    0.0110481214, 0.0076073197, 0.0065423353, 0.0049601258,
    // t = 2, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0007496803, 0.0050309000, 0.0067914625, 0.0077990403,
    0.3517043579, 0.0024265405, 0.0003516180, 0.0009002021,
    0.0225241891, 0.0122288018, 0.0027623691, 0.0235128312,
    0.3683247098, 0.157970713, 0.0105214792, 0.0138495624,
    0.0128591924, 0.0286768654, 0.0232649871, 0.0012800672,
    0.0014079208, 0.0112251786, 0.0134895968, 0.0019244475,
    0.0043541516, 0.0148757305, 0.0016573046, 0.0095518399,
    0.0110481214, 0.0076073197, 0.0065423353, 0.0049601258,
    // t = 3, blk = 0  -- Emit 1 from stay 0 or stay 0 from stay 0 or stay 1 from move
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0089169425, 0.0204893546, 0.0078609179, 0.0121122319,
    0.2299182085, 0.0009818002, 0.0123050441, 0.0025584841,
    0.0026559546, 0.2055943539, 0.0104440380, 0.0159956414,
    0.3004618489, 0.0087352857, 0.0313558198, 0.0035277693,
    0.0092550327, 0.0011409924, 0.0004733293, 0.0047309563,
    0.0090561258, 0.0178979729, 0.0003237941, 0.0155244183,
    0.0079154399, 0.0062602855, 0.0212582665, 0.0103166489,
    0.0084103024, 0.0006881162, 0.0069306017, 0.0059040218,
    // t = 3, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0089169425, 0.0204893546, 0.0078609179, 0.0121122319,
    0.2299182085, 0.0009818002, 0.0123050441, 0.0025584841,
    0.0026559546, 0.2055943539, 0.0104440380, 0.0159956414,
    0.3004618489, 0.0087352857, 0.0313558198, 0.0035277693,
    0.0092550327, 0.0011409924, 0.0004733293, 0.0047309563,
    0.0090561258, 0.0178979729, 0.0003237941, 0.0155244183,
    0.0079154399, 0.0062602855, 0.0212582665, 0.0103166489,
    0.0084103024, 0.0006881162, 0.0069306017, 0.0059040218,
    // t = 4, blk = 0  -- Emit 3 from stay 1 or stay 1 from stay 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0045144728, 0.0152364397, 0.0018720109, 0.0420903087,
    0.0239211541, 0.0299524632, 0.0003726000, 0.0039259700,
    0.0139920032, 0.0024608693, 0.0013762509, 0.0010279794,
    0.0154602536, 0.3564521207, 0.0002227533, 0.0036813149,
    0.0105946021, 0.0127061626, 0.0047861877, 0.0008319803,
    0.0160639509, 0.0048363418, 0.0011550219, 0.0074113201,
    0.0055299491, 0.0159763357, 0.0233073960, 0.0033654280,
    0.0063297962, 0.3546224956,  0.0011800171, 0.0147440505,
    // t = 4, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0045144728, 0.0152364397, 0.0018720109, 0.0420903087,
    0.0239211541, 0.0299524632, 0.0003726000, 0.0039259700,
    0.0139920032, 0.0024608693, 0.0013762509, 0.0010279794,
    0.0154602536, 0.3564521207, 0.0002227533, 0.0036813149,
    0.0105946021, 0.0127061626, 0.0047861877, 0.0008319803,
    0.0160639509, 0.0048363418, 0.0011550219, 0.0074113201,
    0.0055299491, 0.0159763357, 0.0233073960, 0.0033654280,
    0.0063297962, 0.3546224956,  0.0011800171, 0.0147440505,
    // t = 5, blk = 0  -- Stay 3  from move 3
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0014882883, 0.0131499234, 0.0040514728, 0.0317370398,
    0.0047969750, 0.0012508880, 0.0014686562, 0.0009161597,
    0.0011055122, 0.0160382777, 0.0023294087, 0.0196615460,
    0.0265793182, 0.0068026587, 0.0025203085, 0.0084010835,
    0.0107499707, 0.0018732077, 0.0004093150, 0.0274004225,
    0.0025055700, 0.0169708857, 0.0110173089, 0.0073413633,
    0.0055491697, 0.0036744513, 0.0138278391, 0.7045283690,
    0.0220236745, 0.0060608459, 0.0215773739, 0.0021927159,
    // t = 5, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0014882883, 0.0131499234, 0.0040514728, 0.0317370398,
    0.0047969750, 0.0012508880, 0.0014686562, 0.0009161597,
    0.0011055122, 0.0160382777, 0.0023294087, 0.0196615460,
    0.0265793182, 0.0068026587, 0.0025203085, 0.0084010835,
    0.0107499707, 0.0018732077, 0.0004093150, 0.0274004225,
    0.0025055700, 0.0169708857, 0.0110173089, 0.0073413633,
    0.0055491697, 0.0036744513, 0.0138278391, 0.7045283690,
    0.0220236745, 0.0060608459, 0.0215773739, 0.0021927159,
    // t = 6, blk = 0 -- Emit 2 from stay 3
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.0035368766, 0.0176763945, 0.0144094604, 0.0005121515,
    0.0113218259, 0.0100978844, 0.0094784046, 0.0039237728,
    0.0010149917, 0.0102944451, 0.0075408981, 0.0037130285,
    0.0007712012, 0.0009160400, 0.0035260655, 0.0006303581,
    0.0151389411, 0.0135004758, 0.0027071724, 0.0034833912,
    0.0026259961, 0.0004788973, 0.0076983941, 0.7171575062,
    0.0108426881, 0.0095991159, 0.0237890337, 0.0018305138,
    0.0433776194, 0.0147823422, 0.0182747877, 0.0153493259,
    // t = 6, blk = 1
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.0035368766, 0.0176763945, 0.0144094604, 0.0005121515,
    0.0113218259, 0.0100978844, 0.0094784046, 0.0039237728,
    0.0010149917, 0.0102944451, 0.0075408981, 0.0037130285,
    0.0007712012, 0.0009160400, 0.0035260655, 0.0006303581,
    0.0151389411, 0.0135004758, 0.0027071724, 0.0034833912,
    0.0026259961, 0.0004788973, 0.0076983941, 0.7171575062,
    0.0108426881, 0.0095991159, 0.0237890337, 0.0018305138,
    0.0433776194, 0.0147823422, 0.0182747877, 0.0153493259,
    // t = 7, blk = 0 --  Stay 2 from move 2
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0072703959, 0.0038688174, 0.0145134286, 0.0010705109,
    0.0058539880, 0.0002523869, 0.0173387193, 0.0002549596,
    0.0352496596, 0.0149159136, 0.0213974385, 0.0040372424,
    0.0184696336, 0.0490622354, 0.0015680775, 0.0049759929,
    0.0115750953, 0.0116075920, 0.7038388732, 0.0081012472,
    0.0036997281, 0.0084331103, 0.0006375292, 0.0047651898,
    0.0102613474, 0.0053520854, 0.0008666171, 0.0067453974,
    0.0049010803, 0.0145512570, 0.0041740272, 0.0003904234,
    // t = 7, blk = 1
    0.1333333, 0.1333333, 0.1333333, 0.1333333,
    0.6, 0.6, 0.6, 0.6,
    0.0072703959, 0.0038688174, 0.0145134286, 0.0010705109,
    0.0058539880, 0.0002523869, 0.0173387193, 0.0002549596,
    0.0352496596, 0.0149159136, 0.0213974385, 0.0040372424,
    0.0184696336, 0.0490622354, 0.0015680775, 0.0049759929,
    0.0115750953, 0.0116075920, 0.7038388732, 0.0081012472,
    0.0036997281, 0.0084331103, 0.0006375292, 0.0047651898,
    0.0102613474, 0.0053520854, 0.0008666171, 0.0067453974,
    0.0049010803, 0.0145512570, 0.0041740272, 0.0003904234,
    // t = 8, blk = 0  -- Emit 2 from stay 3 or stay 2 from stay 2
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.0065312704, 0.0048444858, 0.0059903353, 0.0027631761,
    0.0226886582, 0.0043530531, 0.0092550274, 0.0136627715,
    0.0067889200, 0.0117303198, 0.0178888890, 0.0052730074,
    0.0104339262, 0.0024049406, 0.0347368204, 0.0115717174,
    0.0015636077, 0.0107197186, 0.0107525709, 0.0383855063,
    0.0009785354, 0.0023067402, 0.3577762979, 0.3554982556,
    0.0162829278, 0.0092637600, 0.0004637169, 0.0038278852,
    0.0033467964, 0.0140091961, 0.0019783804, 0.0019287864,
    // t = 8, blk = 1
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.0065312704, 0.0048444858, 0.0059903353, 0.0027631761,
    0.0226886582, 0.0043530531, 0.0092550274, 0.0136627715,
    0.0067889200, 0.0117303198, 0.0178888890, 0.0052730074,
    0.0104339262, 0.0024049406, 0.0347368204, 0.0115717174,
    0.0015636077, 0.0107197186, 0.0107525709, 0.0383855063,
    0.0009785354, 0.0023067402, 0.3577762979, 0.3554982556,
    0.0162829278, 0.0092637600, 0.0004637169, 0.0038278852,
    0.0033467964, 0.0140091961, 0.0019783804, 0.0019287864,
    // t = 9, blk = 0  -- Emit 2 from stay 3 or stay 2 from stay 2
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.0354652428, 0.0029854747, 0.0128524772, 0.0421994210,
    0.0037176721, 0.0084114227, 0.0033211400, 0.0052347903,
    0.0077124671, 0.0026205456, 0.0019610206, 0.0016939819,
    0.0240758092, 0.0001718899, 0.0083702136, 0.0156192301,
    0.0024609326, 0.0035612521, 0.0026555157, 0.0168106178,
    0.0002516839, 0.0002317396, 0.3504806908, 0.3691176080,
    0.0003221497, 0.0225912061, 0.0084688295, 0.0018269403,
    0.0345275618, 0.0059005448, 0.0013243012, 0.0030556270,
    // t = 9, blk = 1
    2.8, 2.8, 2.8, 2.8,
    1.2, 1.2, 1.2, 1.2,
    0.0354652428, 0.0029854747, 0.0128524772, 0.0421994210,
    0.0037176721, 0.0084114227, 0.0033211400, 0.0052347903,
    0.0077124671, 0.0026205456, 0.0019610206, 0.0016939819,
    0.0240758092, 0.0001718899, 0.0083702136, 0.0156192301,
    0.0024609326, 0.0035612521, 0.0026555157, 0.0168106178,
    0.0002516839, 0.0002317396, 0.3504806908, 0.3691176080,
    0.0003221497, 0.0225912061, 0.0084688295, 0.0018269403,
    0.0345275618, 0.0059005448, 0.0013243012, 0.0030556270
};



#include <stdio.h>

int main(int argc, char * argv[]){

    const size_t nblk = 10;
    const size_t nstate = 40;
    const size_t nbatch = 2;
    float score[2] = {0.0f};
    float score2[2] = {0.0f};
    const float DELTA = 1e-3f;
    const size_t msize = nblk * nstate * nbatch;

    for(size_t i=0 ; i < nblk * nbatch ; i++){
        const size_t offset = i * nstate;
        for(size_t j=_PARAM_TRANS_OFF ; j < nstate ; j++){
            test_param1[offset + j] = logf(test_param1[offset + j]);
        }
    }
    for(size_t i=0 ; i < nblk * nbatch ; i++){
        for(size_t j=0 ; j < 4 ; j++){
            const size_t offset = i * nstate + j;
            printf("%f %f %f : %f %f %f %f\n",
                   test_param1[offset + _PARAM_TRANS_OFF],
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
    crf_runlength_scores_fwd(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
    printf("Forwards scores: %f %f\n", score[0], score[1]);

    crf_runlength_scores_bwd(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
    printf("Backwards scores: %f %f\n", score[0], score[1]);

    //crf_runlength_viterbi_cost(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
    //printf("Viterbi scores: %f %f\n", score[0], score[1]);

    float * grad = calloc(msize, sizeof(float));
    crf_runlength_grad(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score2, grad);
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
            crf_runlength_scores_fwd(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
            fscore[0] = score[0];
            fscore[1] = score[1];
            // Negative difference
            test_param1[offset + st] = orig - DELTA;
            crf_runlength_scores_fwd(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
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
    crf_runlength_viterbi_grad(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, grad);
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
            crf_runlength_viterbi_cost(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
            fscore[0] = score[0];
            fscore[1] = score[1];
            // Negative difference
            test_param1[offset + st] = orig - DELTA;
            crf_runlength_viterbi_cost(test_param1, nstate, nblk, nbatch, test_seq1, test_rle1, test_seqlen1, score);
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
