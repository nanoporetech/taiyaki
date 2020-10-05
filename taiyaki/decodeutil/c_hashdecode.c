#define _BSD_SOURCE
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "fasthash.h"
#include "qsort.h"
#include "yastring.h"

#define MAX(X, Y) (((X) >= (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) <= (Y)) ? (X) : (Y))
#define MOVE_IDX(FROM, TO, N) ((FROM) + 2 * (N) * (((TO) < (N)) ? (TO) : (N)))
#define STAY_IDX(A, N) MOVE_IDX(A, A, N)

typedef int8_t base_t;

float logsumexpf(float x, float y){
    float absdif = fabsf(x - y);
    return fmaxf(x, y) + ((absdif < 17.0f) ? log1pf(expf(-absdif)) : 0.0f);
}


size_t nstate_from_nbase(size_t nbase){
    return nbase + nbase;
}


size_t ntrans_from_nbase(size_t nbase){
    return (nbase + nbase) * (nbase + 1);
}


typedef struct _beamelt {
    yastring seq;
    uint64_t hash;
    float score;
} beamelt;


typedef struct _beamrec {
    uint64_t hash;
    base_t base;
    float score;
    size_t origbeam;
} beamrec;


void beamrec_fprint(FILE * fh, const beamrec elt){
    fprintf(fh, "%lx from beam %zu + %d has score %f\n",
            elt.hash, elt.origbeam, elt.base, elt.score);
}


static const char ALPHA[9] = "ACGTacgt";
void beamelt_fprint(FILE * fh, const beamelt elt){
    fprintf(fh, "%lx has score %f\n", elt.hash, elt.score);
    for(size_t i=0 ; i < elt.seq.len ; i++){
        fputc(ALPHA[elt.seq.str[i]], fh);
    }
    fputc('\n', fh);
}


void beamrec_sorthash(beamrec * A, size_t n){
    beamrec tmp;
#define LESS(i, j) A[i].hash > A[j].hash
#define SWAP(i,j) tmp = A[i], A[i] = A[j], A[j] = tmp
	QSORT(n, LESS, SWAP);
}


void beamrec_sortscore(beamrec * A, size_t n){
    beamrec tmp;
#define LESS(i, j) A[i].score > A[j].score
#define SWAP(i,j) tmp = A[i], A[i] = A[j], A[j] = tmp
	QSORT(n, LESS, SWAP);
}


beamrec * beamrec_bsearchhash(beamrec elt, beamrec * A, size_t n){
	if(NULL == A || 0 == n){ return NULL;}
    int64_t lp = 0;
    int64_t up = n - 1;
    while(lp <= up){
        int64_t mp = (lp + up) / 2;
        //printf("Comparing %lx to (%lx, %lx, %lx) -- %zu %zu %zu\n", elt.hash, A[lp].hash, A[mp].hash, A[up].hash, lp, mp, up);
        if(elt.hash == A[mp].hash){
			return A + mp;
        }
        if(elt.hash < A[mp].hash){
            lp = mp + 1;
        } else {
            up = mp - 1;
        }
    }

    assert(elt.hash != A[0].hash);
    assert(elt.hash != A[n-1].hash);

    return NULL;
}



int beamrec_cmphash(const void * rec1, const void *rec2){
    assert(NULL != rec1);
    assert(NULL != rec2);
    const beamrec _rec1 = *(beamrec *) rec1;
    const beamrec _rec2 = *(beamrec *) rec2;

    const float hash1 = _rec1.hash;
    const float hash2 = _rec2.hash;
    return (hash1 < hash2) - (hash1 > hash2);
}


int beamrec_cmpscore(const void * rec1, const void *rec2){
    assert(NULL != rec1);
    assert(NULL != rec2);
    const beamrec _rec1 = *(beamrec *) rec1;
    const beamrec _rec2 = *(beamrec *) rec2;

    const float score1 = _rec1.score;
    const float score2 = _rec2.score;
    return (score1 < score2) - (score1 > score2);
}


beamelt beamelt_init(char state, uint64_t seed){
    return (beamelt){.seq=yastring_append(yastring_new(), state),
                     .hash=chainfasthash64(seed, state),
                     .score=0.0f};
}


void beamelt_free(beamelt elt){
    yastring_free(elt.seq);
}

beamelt beamelt_copy(const beamelt elt){
    return (beamelt){.seq = yastring_copy(elt.seq),
                     .hash = elt.hash,
                     .score = elt.score};
}


int beamelt_cmpscore(void * elt1, void * elt2){
    const beamelt _elt1 = *(beamelt *) elt1;
    const beamelt _elt2 = *(beamelt *) elt2;

    const float score1 = _elt1.score;
    const float score2 = _elt2.score;
    return (score1 < score2) - (score1 > score2);
}


int beamelt_cmpstate(void * elt1, void * elt2){
    const beamelt _elt1 = *(beamelt *) elt1;
    const beamelt _elt2 = *(beamelt *) elt2;

    const float hash1 = _elt1.hash;
    const float hash2 = _elt2.hash;
    return (hash1 < hash2) - (hash1 > hash2);
}


bool isordered(void * x, size_t nelt, size_t size, int (*cmp)(const void *, const void*)){
    size_t xc = (size_t) x;
    for(size_t i=size ; i < nelt * size ; i+=size){
        int res = cmp((void *)(xc + i - size), (void *)(xc + i));
        if(res > 0){ return false;}
    }
    return true;
}


/**  Beam-search to determine best flip-flop sequence
 *
 *   Args:
 *       score:           Array [nblock x ntrans] of scores
 *       nbase:           Number of bases for flip-flop
 *       nblock:          Number blocks for which scores are provided
 *       bwd:             Array [nblock x nstate] of backwards scores
 *       max_beam_width:  Width of beam (number of values retained between steps).
 *       seq [out]:       Buffer to write-out sequence found.  Should contain
 *   sufficient space for output (maximum nblock).
 *
 *   Returns: score of best sequence found.  Best sequence is written to `seq`
 **/
float flipflop_beamsearch(const float * score, size_t nbase, size_t nblock,
        const float * bwd, int max_beam_width, float beamcut, base_t * seq){
    assert(NULL != score);
    assert(NULL != seq);
    const float logbeamcut = logf(beamcut);
    const uint64_t seed = 0x880355f21e6d1965ULL;

    const size_t nstate = nstate_from_nbase(nbase);
    const size_t ntrans = ntrans_from_nbase(nbase);

    //    Initialise beam search
    //  Each element of beam may extended by one of `nbase` bases, or stay
    const size_t real_max_beam_width = MAX(nbase, max_beam_width);
    const size_t max_beamrec = (nbase + 1) * real_max_beam_width;
    beamrec * beamext = calloc(max_beamrec, sizeof(beamrec));

    beamelt * currbeam = calloc(real_max_beam_width, sizeof(beamelt));
    beamelt * prevbeam = calloc(real_max_beam_width, sizeof(beamelt));
    for(size_t i=0 ; i< nbase ; i++){
        currbeam[i] = beamelt_init(i, seed);
    }
    size_t beam_width = nbase;

    for(size_t blk=0 ; blk < nblock ; blk++){
        size_t nelt = 0;
        const float * currscore = score + blk * ntrans;
        const float * bwdscore = bwd + blk * nstate + nstate;
        {
            beamelt * tmp = prevbeam;
            prevbeam = currbeam;
            currbeam = tmp;
        }

        //  Good lower bound on max score
        float max_score = NAN;
        {
            size_t prevbase = yastring_lastchar(prevbeam[0].seq);
            // Transition to flop state
            max_score = currscore[nbase * nstate + prevbase]
                      + bwdscore[(prevbase < nbase) ? (prevbase + nbase) : prevbase];
            for(size_t i=0 ; i < nbase ; i++){
                // Transitions to flip state
                max_score = fmaxf(max_score, currscore[i * nstate + prevbase] + bwdscore[i]);
            }

            max_score += prevbeam[0].score;
        }


        for(size_t i=0 ; i < beam_width ; i++){
            //     Deal with all extensions of beam.
            //  Since elements of beam are unique, so are the extensions
            const beamelt pelt = prevbeam[i];
            const base_t prevbase = yastring_lastchar(pelt.seq);
            for(size_t base=0 ; base < nbase ; base++){
                uint64_t newbase = (base != prevbase) ? base : (prevbase + nbase);
                float newscore = pelt.score + currscore[MOVE_IDX(prevbase, newbase, nbase)] + bwdscore[newbase];
                if(newscore < max_score + logbeamcut){
                    // Cut beam
                    continue;
                }
                if(newscore > max_score){
                    max_score = newscore;
                }
                uint64_t newhash = chainfasthash64(pelt.hash, newbase);
                beamext[nelt] = (beamrec){newhash, newbase, newscore, i};
                nelt += 1;
            }
        }

        for(size_t i=0 ; i < beam_width ; i++){
            const beamelt pelt = prevbeam[i];
            const base_t base = yastring_lastchar(pelt.seq);
            const float newscore = pelt.score + currscore[STAY_IDX(base, nbase)] + bwdscore[base];
            if(newscore < max_score + logbeamcut){
                // Cut beam
                continue;
            }
            if(newscore > max_score){
                max_score = newscore;
            }
            beamrec newrec = {pelt.hash, -1, newscore, i};
            beamext[nelt] = newrec;
            nelt += 1;
        }

        //  Sort by hash then merge records with same sequence
        beamrec_sorthash(beamext, nelt);
        size_t nelt_uniq = 1;
        for(size_t i=1, j=0 ; i < nelt ; i++){
            if(beamext[i].hash == beamext[j].hash){
                beamext[j].score = logsumexpf(beamext[i].score, beamext[j].score);
                beamext[i].score = -HUGE_VAL;  // End up last, sorted by score
            } else {
                j = i;
                nelt_uniq += 1;
            }
        }

        //  Order by score, to find best elements
        beamrec_sortscore(beamext, nelt);
        assert(isordered(beamext, nelt, sizeof(beamrec), beamrec_cmpscore));
        size_t new_beam_width = MIN(max_beam_width, nelt_uniq);
        for(size_t i=0 ; i < new_beam_width ; i++){
            // Copy best elements
            currbeam[i] = beamelt_copy(prevbeam[beamext[i].origbeam]);
            if(beamext[i].base != -1){
                // Not a stay -- hash and state sequence have changed
                currbeam[i].hash = beamext[i].hash;
                currbeam[i].seq = yastring_append(currbeam[i].seq,
                                                  beamext[i].base);
            }
            // Copy score, removing backwards contribution
            currbeam[i].score = beamext[i].score - bwdscore[yastring_lastchar(currbeam[i].seq)];
        }
        for(size_t i=0 ; i < beam_width ; i++){
            //  Free previous beam
            beamelt_free(prevbeam[i]);
        }
        beam_width = new_beam_width;
    }

    //  Copy best result to output
    const float final_score = currbeam[0].score;
    const size_t seqlen = currbeam[0].seq.len;
    for(size_t i=0 ; i < seqlen ; i++){
        seq[i] = currbeam[0].seq.str[i];
    }
    for(size_t i=seqlen ; i < nblock ; i++){
        seq[i] = -1;
    }

    //  Clear-up
    for(size_t i=0 ; i < beam_width ; i++){
        beamelt_free(currbeam[i]);
    }
    free(currbeam);
    free(prevbeam);  // elements already free'd in loop above
    free(beamext);

    return final_score;
}



#ifdef DECODEUTIL_TEST
#include <stdio.h>

float randomf(void){
    float res = (float)((double)random() / RAND_MAX);
    assert(res >= 0.0f);
    assert(res <= 1.0f);
    return res;
}


float randn(void){
    static float _randn_state = NAN;
    if(finitef(_randn_state)){
        float res = _randn_state;
        _randn_state = NAN;
        return res;
    }

    float R = sqrtf(-2.0f * logf(randomf()));
    float theta = 2.0f * M_PI * randomf();
    _randn_state = R * cosf(theta);
    return R * sinf(theta);
}



const size_t nbase = 4;

int main(int argc, char * argv[]){
    if(argc==1){
        fputs("Usage: ./a.out seed ntimes beam_width nblock beamcut\n", stdout);
        exit(EXIT_SUCCESS);
    }
    const unsigned int seed = (argc > 1) ? atoi(argv[1]) : 0;
    printf("Seed is %u\n", seed);
    srandom(seed);

    const int ntimes = (argc > 2) ? atoi(argv[2]) : 1;
    assert(ntimes >= 0);

    const int beam_width = (argc > 3) ? atoi(argv[3]) : 5;
    assert(beam_width >= nbase + 1);

    const size_t test_nblock = (argc > 4) ? atoi(argv[4]) : 12000;
    assert(test_nblock >= 0);

    const float beamcut = (argc > 5) ? atof(argv[5]) : 0.0f;
    assert(beamcut >= 0.0f);

    const size_t nstate = nstate_from_nbase(nbase);
    const size_t ntrans = ntrans_from_nbase(nbase);
    float * test_score = calloc(ntrans * test_nblock, sizeof(float));
    for(size_t i=0 ; i < ntrans * test_nblock ; i++){
        const float r = randn();
        test_score[i] = r;
    }

    float * zerobwd = calloc(nstate * (test_nblock + 1), sizeof(float));

    for(int i=0 ; i < ntimes ; i++){
        base_t * seq = calloc(test_nblock, sizeof(base_t));
        float score = flipflop_beamsearch(test_score, nbase, test_nblock, zerobwd,
                beam_width, beamcut, seq);
        size_t seqlen = 0;
        for( ; seqlen < test_nblock && seq[seqlen] != -1 ; seqlen++);
        printf("Score is %f.  Seq length is %zu\n", score, seqlen);
        free(seq);
    }
/*
    for(size_t i=0 ; i < test_nblock && seq[i] != -1 ; i++){
        fputc(alphabet[seq[i]], stdout);
    }
    fputc('\n', stdout);


    for(size_t b=beam_width ; b <= beam_width + 10 ; b++){
        const float score = flipflop_beamsearch(test_score, nbase, test_nblock, b, seq);
        printf("Score for beam %zu is %f\n", b,  score);
    }
    */
    free(zerobwd);
    free(test_score);

}

#endif  /* DECODEUTIL_TEST */
