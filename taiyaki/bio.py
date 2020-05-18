""" Module containing collection of functions for operating on sequences
represented as strings, and lists thereof.
"""
import re
from Bio import SeqIO

from taiyaki.constants import DEFAULT_ALPHABET

# Base complements
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'X': 'X', 'N': 'N',
               'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'x': 'x', 'n': 'n',
               '-': '-'}


def complement(seq, compdict=_COMPLEMENT):
    """ Return complement of a base sequence.

    :param seq: A string of bases.
    :param compdict: A dictionary containing base complements

    :returns: A string of bases.

    """
    return ''.join(compdict[b] for b in seq)


def reverse_complement(seq, compdict=_COMPLEMENT):
    """ Return reverse complement of a base sequence.

    :param seq: A string of bases.
    :param compdict: A dictionary containing base complements

    :returns: A string of bases.

    """
    return complement(seq, compdict)[::-1]


def fasta_file_to_dict(fasta_file_name, filter_ambig=True, flatten_ambig=True,
                       alphabet=DEFAULT_ALPHABET):
    """Load records from fasta file as a dictionary
    :param filter_ambig:   Drop sequences with ambiguity character
    :param flatten_ambig:  Flatten all ambiguity characters to N's
    """
    notbase_regex = re.compile('[^{}]'.format(alphabet))

    references = {}
    with open(fasta_file_name, 'r') as fh:
        for ref in SeqIO.parse(fh, 'fasta'):
            refseq = str(ref.seq).upper()
            if len(refseq) == 0:
                continue
            if filter_ambig and notbase_regex.search(refseq) is not None:
                continue
            if flatten_ambig:
                refseq = notbase_regex.sub('N', refseq)
            references[ref.id] = refseq

    return references
