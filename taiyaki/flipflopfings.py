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


def flipflop_code(labels, alphabet_length=4):
    """Given a list of digits representing bases, add offset to those in even
    positions within runs of indentical bases.
    param labels : np array of digits representing bases (usually 0-3 for ACGT)
    param alphabet_length : number of symbols in alphabet
    returns: np array c such that c[n] = labels[n] + alphabet_length where labels[n] is in
             an even position in a run of identical symbols, or c[n] = labels[n]
             otherwise

    E.g.
    >> x=np.array([1, 3, 2, 3, 3, 3, 3, 1, 1])
    >> flipflop_code(x)
            array([1, 3, 2, 3, 7, 3, 7, 1, 5])
    """
    x = labels.copy()
    x[flopmask(x)] += alphabet_length
    return x


def path_to_str(path, alphabet='ACGT'):
    """ Convert flipflop path into a basecall string """
    move = np.ediff1d(path, to_begin=1) != 0
    alphabet = np.frombuffer((alphabet * 2).encode(), dtype='u1')
    seq = alphabet[path[move]]
    return seq.tobytes().decode()


def cat_mod_code(labels, network):
    """ Given a numpy array of digits representing bases, convert to canonical
    flip-flop labels (defined by flipflopfings.flipflop_code) and
    modified category values (defined by alphabet.AlphabetInfo).

    :param labels: np array of digits representing bases
    :param network: `taiyaki.layers.Serial` object with
        `GlobalNormFlipFlopCatMod` last layer
    returns: two np arrays representing 1) canonical flip-flop labels and
        2) categorical modified base labels

    E.g. (using alphabet='ACGTZYXW', collapse_alphabet='ACGTCAAT')
    >> x = np.array([1, 5, 2, 4, 3, 3, 6, 7, 3])
    >> cat_mod_code(x)
          array(1, 0, 2, 1, 3, 7, 0, 3, 7), array(0, 1, 0, 1, 0, 0, 2, 1, 0)
    """
    assert is_cat_mod_model(network)
    ff_layer = network.sublayers[-1]
    mod_labels = np.ascontiguousarray(ff_layer.mod_labels[labels])
    can_labels = np.ascontiguousarray(ff_layer.can_labels[labels])
    ff_can_labels = flipflop_code(can_labels, ff_layer.ncan_base)
    return ff_can_labels, mod_labels


def nstate_flipflop(nbase):
    """  Number of states in output of flipflop network

    :param nbase: Number of letters in alphabet

    :returns: Number of states
    """
    return 2 * nbase * (nbase + 1)


def nbase_flipflop(nstate):
    """  Number of letters in alphabet from flipflop network output size

    :param nstate: Flipflop network output size

    :returns: Number of letters in alphabet
    """
    nbase_f = np.sqrt(0.25 + (0.5 * nstate)) - 0.5
    assert np.mod(nbase_f, 1) == 0, (
        'Number of states not valid for flip-flop model. ' +
        'nstates: {}\tconverted nbases: {}').format(nstate, nbase_f)
    return int(np.round(nbase_f))
