""" Module containing collection of functions for operating on sequences
represented as strings, and lists thereof.
"""
import re
import sys

from Bio import SeqIO

from taiyaki.constants import DEFAULT_ALPHABET

# Base complements
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'X': 'X', 'N': 'N',
               'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'x': 'x', 'n': 'n',
               '-': '-'}


def complement(seq, compdict=_COMPLEMENT):
    """ Return complement of a base sequence.

    Args:
        seq (str): Base sequence
        compdict (dict, optional): Base complements

    Returns:
        String of complemented bases.
    """
    return ''.join(compdict[b] for b in seq)


def reverse_complement(seq, compdict=_COMPLEMENT):
    """ Return reverse complement of a base sequence.

    Args:
        seq (str): Base sequence
        compdict (dict, optional): Base complements

    Returns:
        String of reverse complemented bases.
    """
    return complement(seq, compdict)[::-1]


def fasta_file_to_dict(fasta_file_name, filter_ambig=True, flatten_ambig=True,
                       alphabet=DEFAULT_ALPHABET):
    """ Load records from fasta file as a dictionary

    Args:
        fasta_file_name (str): Path to FASTA file
        filter_ambig (bool, optional): Drop sequences with characters not found
          in alphabet
        flatten_ambig (bool, optional): Flatten all characters not found in
          alphabet to N's
        alphabet (str, optional): Valid sequence alphabet found in
          fasta_file_name

    Returns:
        Dictionary contraining ID keys pointing to sequence values.
    """
    notbase_regex = re.compile('[^{}]'.format(alphabet))
    ambig_reads = []

    references = {}
    with open(fasta_file_name, 'r') as fh:
        for ref in SeqIO.parse(fh, 'fasta'):
            refseq = str(ref.seq)
            if len(refseq) == 0:
                continue
            if filter_ambig and notbase_regex.search(refseq) is not None:
                ambig_reads.append(ref.id)
                continue
            if flatten_ambig:
                refseq = notbase_regex.sub('N', refseq)
            references[ref.id] = refseq

    if len(ambig_reads) > 0:
        sys.stderr.write((
            '* {} reference seqeunces contain ambiguous bases not found in ' +
            'the provided alphabet and will be skipped.').format(
                len(ambig_reads)))

    return references
