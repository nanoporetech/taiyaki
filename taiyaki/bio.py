""" Module containing collection of functions for operating on sequences
represented as strings, and lists thereof.
"""
from taiyaki.iterators import product, window

# Base complements
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'X': 'X', 'N': 'N',
               'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'x': 'x', 'n': 'n',
               '-': '-'}


def reverse_complement(seq, compdict=_COMPLEMENT):
    """ Return reverse complement of a base sequence.

    :param seq: A string of bases.
    :param compdict: A dictionary containing base complements

    :returns: A string of bases.

    """
    return ''.join(compdict[b] for b in seq)[::-1]
