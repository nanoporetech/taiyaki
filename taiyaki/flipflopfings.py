# Utilities for flip-flop coding
import numpy as np


def flopmask(labels):
    """Determine which labels are in even positions within runs of identical labels

    param labels : np array of digits representing bases (usually 0-3 for ACGT)
                   or of bases (bytes)
    returns: bool array fm such that fm[n] is True if labels[n] is in
             an even position in a run of identical symbols

    E.g.
    >> x=np.array([1,    3,      2,    3,      3,    3,     3,    1,      1])
    >> flopmask(x)
         array([False, False, False, False,  True, False,  True, False,  True])
    """
    move = np.ediff1d(labels, to_begin=1) != 0
    cumulative_flipflops = (1 - move).cumsum()
    offsets = np.maximum.accumulate(move * cumulative_flipflops)
    return (cumulative_flipflops - offsets) % 2 == 1


def flip_flop_code(labels, alphabet_length=4):
    """Given a list of digits representing bases, add offset to those in even
    positions within runs of indentical bases.
    param labels : np array of digits representing bases (usually 0-3 for ACGT)
    param alphabet_length : number of symbols in alphabet
    returns: np array c such that c[n] = labels[n] + alphabet_length where labels[n] is in
             an even position in a run of identical symbols, or c[n] = labels[n]
             otherwise

    E.g.
    >> x=np.array([1, 3, 2, 3, 3, 3, 3, 1, 1])
    >> flip_flop_code(x)
            array([1, 3, 2, 3, 7, 3, 7, 1, 5])
    """
    x = labels.copy()
    x[flopmask(x)] += alphabet_length
    return x
