import numpy as np
from taiyaki import flipflopfings
from taiyaki.constants import DEFAULT_ALPHABET, LARGE_VAL


def map_to_crf_viterbi(scores, step_index, stay_index, localpen=LARGE_VAL):
    """Find highest scoring path corresponding to a given label sequence

    Args:
        scores (np float array): 2D array of CRF transition scores (log-space)
        step_index (np int array) : index of scores to use to step to the
                    next sequence position, corresponding to diagonal moves
                    in the alignment matrix, e.g. for a flipflop CRF and
                    the sequence ATTC the step_index would correspond to
                    the moves flipA->flipT, flipT->flopT, flopT->flipC
        stay_index (np int array) : index of scores to use to stay at same
                    sequence position, which should be length 1 longer than
                    the step_index (we assume start at position 0)
                    e.g. in a flipflop CRF these would be the
                    flipN->flipN and flopN->flopN states
        localpen (float): score for skipping over signal at the start or end
                    of the alignment

    Returns:
        tuple :(score of best path, best path)
    """
    N, M = len(scores), len(stay_index)
    assert len(step_index) == len(stay_index) - 1

    pscore = np.full(M, -LARGE_VAL)
    cscore = np.full(M, -LARGE_VAL)
    cscore[0] = 0

    start_score = 0.0
    end_score = -LARGE_VAL
    alignment_end = 0

    traceback = [np.zeros(M, dtype='u1')]

    for n in range(N):
        traceback.append(np.zeros(M, dtype='u1'))

        step_scores = scores[n, step_index]
        stay_scores = scores[n, stay_index]

        pscore, cscore = cscore, pscore

        # stay
        cstay = pscore + stay_scores

        # step
        cstep = pscore[:-1] + step_scores

        # start
        leave_start_score = start_score - localpen
        start_score = start_score + max(stay_scores[0], -localpen)

        # update cscore
        cscore[:] = cstay[:]
        cscore[1:] = np.maximum(cscore[1:], cstep)
        cscore[0] = max(cscore[0], start_score)
        traceback[n + 1][1:] = cstay[1:] < cstep
        traceback[n + 1][0] = 1 if leave_start_score > cstay[0] else 0

        # end
        remain_in_end_score = end_score + max(stay_scores[-1], -localpen)
        step_into_end_score = pscore[-1] - localpen
        end_score = max(remain_in_end_score, step_into_end_score)
        if step_into_end_score > remain_in_end_score:
            alignment_end = n

        traceback[-1] = np.packbits(traceback[-1])

    path = np.full(N + 1, -1, dtype=int)
    if cscore[-1] > end_score:
        # traceback starts at end of sequence
        n, m = N, M - 1
    else:
        # traceback starts in "end" state
        n, m = alignment_end, M - 1

    while n >= 0 and m >= 0:
        path[n] = m
        move = np.unpackbits(traceback[n])[m]
        m -= move
        n -= 1

    return max(cscore[-1], end_score), path


def flipflop_remap(transition_scores, sequence, alphabet=DEFAULT_ALPHABET,
                   localpen=LARGE_VAL):
    """Finds the best alignment between a matrix of flipflop transition scores
    and a sequence.

    Returns the score calculated for the best path, and an array of
    sequence positions that correspond to that path. The positions array
    has length 1 more than the scores matrix; this is because the scores
    matrix contains scores for transitions that will either move us to the
    next position, or stay at the same position.

    The entire sequence must be used in the alignment, but the scores might
    be clipped, depending on the value of localpen. This is acheived by
    introducing "start" and "end" states. The alignment must start in the
    "start" state, move out of "start" into the first position in the
    sequence, traverse the entire sequence, and then enter the "end" state.
    The alignment can stay in the "start" or "end" states by paying a cost
    of localpen while ignoring the next row of transition scores.
    Therefore, a large value of localpen will force the entire scores
    matrix to be used in the alignment ("global mapping"), while smaller
    values will lead to more clipping ("glocal mapping"). The time spent in
    the "start" and "end" states will be marked with -1s.


    The output positions array will have 3 sections:
      1. zero or more -1s for time spent in the "start" state
      2. a monotonic sequence of positions starting with 0 and ending with
         len(sequence) - 1
      3. zero or more -1s for time spend in the "end" state

    Args:
        scores (np float array) : an array of network outputs of shape
                                  (T, K) where K = 2 * nbase * (nbase + 1)
        sequence (str) : reference sequence to map to
        alphabet (str)  : alphabet of length nbase used in sequence
        localpen (float): score for staying in the start or end states

    Returns:
        np array: alignment score, array of sequence positions of length T + 1
    """
    nbase = len(alphabet)
    bases = np.array([alphabet.find(b) for b in sequence])
    flops = flipflopfings.flopmask(bases)

    stay_index = np.where(flops, bases + (2 * nbase + 1)
                          * nbase, bases + 2 * nbase * bases)
    from_base = (bases + flops * nbase)[:-1]
    to_base = np.maximum(bases, nbase * flops)[1:]
    step_index = from_base + 2 * nbase * to_base

    return map_to_crf_viterbi(
        transition_scores, step_index, stay_index, localpen=localpen)
