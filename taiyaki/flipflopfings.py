# Utilities for flip-flop coding
import numpy as np
from taiyaki.constants import DEFAULT_ALPHABET


def flopmask(labels):
    """Determine which labels are in even positions within runs of identical labels

    Args:
        labels: np array of digits representing bases (usually 0-3 for ACGT)
            or of bases (bytes)

    Returns:
        bool array fm such that fm[n] is True if labels[n] is in
            an even position in a run of identical symbols

    Examples:
        >>> x = np.array([1, 3, 2, 3, 3, 3, 3, 1, 1])
        >>> flopmask(x)
        array([False, False, False, False,  True, False,  True, False,  True])
    """
    move = np.ediff1d(labels, to_begin=1) != 0
    cumulative_flipflops = (1 - move).cumsum()
    offsets = np.maximum.accumulate(move * cumulative_flipflops)
    return (cumulative_flipflops - offsets) % 2 == 1


def flipflop_code(labels, alphabet_length=4):
    """Converts encoded bases to flip-flop codes

    Given a list of digits representing bases, add offset to those in even
    positions within runs of indentical bases.

    Args:
        labels: np array of digits representing bases (usually 0-3 for ACGT)
        alphabet_length: number of symbols in alphabet

    Returns:
        np array c such that c[n] = labels[n] + alphabet_length where labels[n] is in
            an even position in a run of identical symbols, or c[n] = labels[n]
            otherwise

    Examples:
        >>> x = np.array([1, 3, 2, 3, 3, 3, 3, 1, 1])
        >>> flipflop_code(x)
        array([1, 3, 2, 3, 7, 3, 7, 1, 5])
    """
    x = labels.copy()
    x[flopmask(x)] += alphabet_length
    return x


def path_to_str(path, alphabet=DEFAULT_ALPHABET, include_first_source=True):
    """ Convert flipflop path into a basecall string.

    Args:
        path: numpy vector of integers coding flip-flop states (0-7 for ACGT)
        alphabet: python str containing alphabet
        include_first_source: bool. Include the source state of the first
            transition in the path in the basecall. Guppy doesn't do this,
            so use False for (better) agreement with Guppy.

    Returns:
        str: the basecall
    """
    move = np.ediff1d(path, to_begin=1 if include_first_source else 0) != 0
    alphabet = np.frombuffer((alphabet * 2).encode(), dtype='u1')
    seq = alphabet[path[move]]
    return seq.tobytes().decode()


def extract_mod_weights(mod_weights, path, can_nmods):
    """Extract scores of modified bases for a given transition sequence


    Args:
        mod_weights: raw modified base scores, array of shape [T, sum(can_nmods)]
        path: a array of length T. Each element is in the range from 0 to
            2 times the alphabet length, specifying which state the basecaller
            is in at each time point, either a flip (0..alphabet length) or flop
            (> alpabet length). Note that a new base is added to the basecall only
            when the state changes, i.e. after collapsing all repeated elements,
            so the basecall usually has length less than T.
        can_nmods: array giving the number of possible modifications for each
            base in the alphabet

    Returns:
        array of mod scores of shape [basecall length, sum(can_nmods)]

    Note:
        the first base is always marked as unmodified
    """
    # skip initial base from base calling that was never "moved into"
    move = np.ediff1d(path, to_begin=0) != 0
    path_vals = path[move]
    # extract weights at basecall positions
    bc_mod_weights = mod_weights[move[1:]]
    curr_can_pos = 0
    mods_scores = []
    for base_i, can_nmod in enumerate(can_nmods):
        if can_nmod > 0:
            base_poss = np.where(np.equal(np.mod(
                path_vals, len(can_nmods)), base_i))[0]
        for mod_i in range(can_nmod):
            mod_i_scores = np.full(
                bc_mod_weights.shape[0] + 1, np.NAN)
            # first base is always unmodified since it is never "moved into"
            mod_i_scores[base_poss + 1] = bc_mod_weights[
                base_poss, curr_can_pos + 1 + mod_i]
            mods_scores.append(mod_i_scores)
        curr_can_pos += 1 + can_nmod
    mods_scores = np.stack(mods_scores, axis=1)

    return mods_scores


def nstate_flipflop(nbase):
    """ Number of transitions in output of flipflop network

    The flip-flop model over an alphabet of length L has 2L states:
    a 'flip' and 'flop' version of each letter. Within runs of the same
    base, the model alternates between the flip and flop versions.

    The model is allowed to make the following transitions:
        - stay in any state (no base is added)
        - move from flip X to flop X (for each letter X)
        - move from flip X to flip Y (for any letters X and Y != X)
        - move from flop X to flip Y (for any letters X and Y)

    In total there are :math:`2L` possible stays, and :math:`2L^2` possible
    moves, giving a total of :math:`2L(L + 1)` possible transitions.

    Args:
        nbase: Number of letters in alphabet

    Returns:
        Number of transitions
    """
    return 2 * nbase * (nbase + 1)


def nbase_flipflop(nstate):
    """  Number of letters in alphabet from flipflop network output size

    Args:
        nstate: Flipflop network output size

    Returns:
        Number of letters in alphabet
    """
    nbase_f = np.sqrt(0.25 + (0.5 * np.float32(nstate))) - 0.5
    assert np.mod(nbase_f, 1) == 0, (
        'Number of states not valid for flip-flop model. ' +
        'nstates: {}\tconverted nbases: {}').format(nstate, nbase_f)
    return int(np.round(nbase_f))
