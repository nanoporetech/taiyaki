#include <assert.h>
#include <math.h>
#include "c_decoding.h"

#define BIG_FLOAT 1.e30f


/**
 *
 *   @param x[nr*nc] Array containing matrix stored column-major
 *   @param nr Number of rows
 *   @param nc Number of columns
 *   @param idx[nc] Array[out] to write indices
 *
 *   @returns Indices found stored in array `idx`
 **/
void colmaxf(float * x, size_t nr, size_t nc, int * idx){
    assert(nr > 0);
    assert(nc > 0);
    assert(NULL != x);
    assert(NULL != idx);

    for(int r=0 ; r < nr ; r++){
        // Initialise
        idx[r] = 0;
    }

    for(int c=1 ; c < nc ; c++){
        const size_t offset2 = c * nr;
        for(int r=0 ; r<nr ; r++){
            if(x[offset2 + r] > x[idx[r] * nr + r]){
                idx[r] = c;
            }
        }
    }
}


/**  Find location of maximum element of array
 *
 *   @param x Array
 *   @param n Length of array
 *
 *   @returns Index of maximum element or -1 on error
 **/
int argmaxf(const float *x, size_t n) {
    assert(n > 0);
    if (NULL == x) {
        return -1;
    }
    int imax = 0;
    float vmax = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > vmax) {
            vmax = x[i];
            imax = i;
        }
    }
    return imax;
}


/**  Backtrace to determine best Viterbi path
 *
 *   @param curr_score[nstate]         Array containing forward scores of last position
 *   @param nstate                     Number of states including start and end states
 *   @param nblock                     Number of block
 *   @param traceback[nstate*nblock]   Array containing traceback matrix stored
 *                                     column-major format, or NULL
 *   @param path[nblock]               Array[out] to write path, or NULL
 *
 *   @returns Score of best path.  Best scoring path is writen to the array path if both
 *            traceback and path are non-NULL
 **/
float viterbi_local_backtrace(float const *curr_score, size_t nstate, size_t nblock, int const * traceback, int32_t * path){
    assert(NULL != curr_score);

    const int32_t START_STATE = nstate - 2;
    const int32_t END_STATE = nstate - 1;

    int32_t last_state = argmaxf(curr_score, nstate);
    float logscore = curr_score[last_state];

    if(NULL != path && NULL != traceback){
        // Decode
        for(size_t i=0 ; i<=nblock ; i++){
            // Initialise entries to stay
            path[i] = -1;
        }

        for(int i=0 ; i < nblock ; i++){
            const int ri = nblock - i - 1;
            const int32_t state = traceback[ri * nstate + last_state];
            if(state >= 0){
                path[ri + 1] = last_state;
                last_state = state;
            }
        }
        path[0] = last_state;

        //  Transcode start to stay
        for(int i=0 ; i < nblock ; i++){
            if(path[i] == START_STATE){
                path[i] = -1;
            } else {
                break;
            }
        }
        //  Transcode end to stay
        for(int i=nblock ; i >= 0 ; i--){
            if(path[i] == END_STATE){
                path[i] = -1;
            } else {
                break;
            }
        }
    }

    return logscore;
}


/**  Forwards sweep of Viterbi algorithm,
 *
 *   @param logpost       Array containing weights for each block.  Strided matrix with
 *                        column-major storage.
 *   @param nblock        Number of blocks in each chunk
 *   @param nparam        Number of parameters (weights) output per block
 *   @param nbase         Number of bases
 *   @param stride        Stride of matrix logpost
 *   @param stay_pen      Penalty to suppress stays (positive == more suppression)
 *   @param skip_pen      Penalty to suppress skips (positive == more suppression)
 *   @param local_pen     Local matching penalty (positive == less clipping)
 *   @param path[nblock]  Array[out] to write path, or NULL
 *
 *   @returns Score of best path.  Best scoring path is writen to the array path if both
 *            traceback and path are non-NULL
 **/
float fast_viterbi(float const * logpost, size_t nblock, size_t nparam, size_t nbase, size_t stride,
                   float stay_pen, float skip_pen, float local_pen,
                   int32_t *path){
    float logscore = NAN;
    assert(NULL != logpost);

    const int nstep = nbase;
    const int nskip = nbase * nbase;

    const size_t nhst = nparam - 1;
    assert(nhst % nstep == 0);
    assert(nhst % nskip == 0);
    const int step_rem = nhst / nstep;
    const int skip_rem = nhst / nskip;

    const size_t nstate = nhst + 2;  // States including start and end for local matching
    const size_t START_STATE = nhst;
    const size_t END_STATE = nhst + 1;
    const size_t STAY = 0;

    float * cscore = calloc(nstate, sizeof(float));
    float * pscore = calloc(nstate, sizeof(float));
    int * step_idx = calloc(step_rem, sizeof(int));
    int * skip_idx = calloc(skip_rem, sizeof(int));
    int * traceback = calloc(nstate * nblock, sizeof(int));

    if(NULL != cscore && NULL != pscore && NULL != step_idx && NULL != skip_idx && NULL != traceback){
        // Initialise -- must begin in start state
        for(size_t i=0 ; i < nstate ; i++){
            cscore[i] = -BIG_FLOAT;
        }
        cscore[START_STATE] = 0.0f;

        //  Forwards Viterbi
        for(int i=0 ; i < nblock ; i++){
            const size_t lpoffset = i * stride;
            const size_t toffset = i * nstate;
            {  // Swap vectors
                float * tmp = pscore;
                pscore = cscore;
                cscore = tmp;
            }

            //  Step indices
            colmaxf(pscore, step_rem, nstep, step_idx);
            //  Skip indices
            colmaxf(pscore, skip_rem, nskip, skip_idx);

            // Update score for step and skip
            for(int hst=0 ; hst < nhst ; hst++){
                int step_prefix = hst / nstep;
                int skip_prefix = hst / nskip;
                int step_hst = step_prefix + step_idx[step_prefix] * step_rem;
                int skip_hst = skip_prefix + skip_idx[skip_prefix] * skip_rem;

                float step_score = pscore[step_hst];
                float skip_score = pscore[skip_hst] - skip_pen;
                if(step_score > skip_score){
                    // Arbitrary assumption here!  Should be >= ?
                    cscore[hst] = step_score;
                    traceback[toffset + hst] = step_hst;
                } else {
                    cscore[hst] = skip_score;
                    traceback[toffset + hst] = skip_hst;
                }
                cscore[hst] += logpost[lpoffset + hst + 1];
            }

            // Stay
            for(int hst=0 ; hst < nhst ; hst++){
                const float score = pscore[hst] + logpost[lpoffset + STAY] - stay_pen;
                if(score > cscore[hst]){
                    // Arbitrary assumption here!  Should be >= ?
                    cscore[hst] = score;
                    traceback[toffset + hst] = -1;
                }
            }

            // Remain in start state -- local penalty or stay
            cscore[START_STATE] = pscore[START_STATE] + fmaxf(-local_pen, logpost[lpoffset + STAY] - stay_pen);
            traceback[toffset + START_STATE] = START_STATE;
            // Exit start state
            for(int hst=0 ; hst < nhst ; hst++){
                const float score = pscore[START_STATE] + logpost[lpoffset + hst + 1];
                if(score > cscore[hst]){
                    cscore[hst] = score;
                    traceback[toffset + hst] = START_STATE;
                }
            }

            // Remain in end state -- local penalty or stay
            cscore[END_STATE] = pscore[END_STATE] + fmaxf(-local_pen, logpost[lpoffset + STAY] - stay_pen);
            traceback[toffset + END_STATE] = END_STATE;
            // Enter end state
            for(int hst=0 ; hst < nhst ; hst++){
                const float score = pscore[hst] - local_pen;
                if(score > cscore[END_STATE]){
                    cscore[END_STATE] = score;
                    traceback[toffset + END_STATE] = hst;
                }
            }
        }

        logscore = viterbi_local_backtrace(cscore, nstate, nblock, traceback, path);
    }

    free(traceback);
    free(skip_idx);
    free(step_idx);
    free(pscore);
    free(cscore);

    return logscore;
}


/**
 *
 *    @param weights[nparam*nbatch*nblock]  Array containing weight tensor
 *    @param nblock              Number of blocks in each chunk
 *    @param nbatch              Batch size (number of chunks)
 *    @param nparam              Number of parameters (weights) output per block
 *    @param nbase               Number of bases
 *    @param stay_pen            Penalty to suppress stays (positive == more suppression)
 *    @param skip_pen            Penalty to suppress skips (positive == more suppression)
 *    @param local_pen           Local matching penalty (positive == less clipping)
 *    @param score[nbatch]       Array[out] to contain score of best path for each chunk
 *    @param path[nblock*nbatch] Array[out] to contain best path for each chunk, stored as
 *                               column in a matrix (column-major format)
 *
 *    @returns Scores of best paths are written to score array, path
 **/
void fast_viterbi_blocks(float const * weights, size_t nblock, size_t nbatch, size_t nparam, size_t nbase,
                           float stay_pen, float skip_pen, float local_pen, float * score, int32_t * path){
    assert(NULL != weights);  // weights [nblock x nbatch x nparam]
    assert(NULL != score);    // score [nbatch]
    assert(NULL != path);     // path [nbatch x (nblock + 1)]

    #pragma omp parallel for
    for(size_t batch=0 ; batch < nbatch ; batch++){
        const size_t path_offset = batch * nblock;
        const size_t w_offset = batch * nparam;
        const size_t w_stride = nbatch * nparam;

        score[batch] = fast_viterbi(weights + w_offset, nblock, nparam, nbase, w_stride,
                                    stay_pen, skip_pen, local_pen,
                                    path + path_offset);
    }
}

