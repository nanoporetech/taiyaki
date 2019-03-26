DEFAULT_ALPHABET = b'ACGT'
DEFAULT_NBASE = len(DEFAULT_ALPHABET)
DOTROWLENGTH=50   #Length of a row of dots (polka) in training output
LARGE_LOG_VAL = 50000.0
SMALL_VAL = 1e-10


def nstate_flipflop(nbase):
    """  Number of states in output of flipflop network

    :param nbase: Number of letters in alphabet

    :returns: Number of states
    """
    return 2 * nbase * (nbase + 1)
