DEFAULT_ALPHABET = b'ACGT'
DEFAULT_NBASE = len(DEFAULT_ALPHABET)

LARGE_LOG_VAL = 50000.0
SMALL_VAL = 1e-10


def nstate_flipflop(nbase):
    """  Number of states in output of flipflop network

    :param nbase: Number of letters in alphabet

    :returns: Number of states
    """
    return 2 * nbase * (nbase + 1)
