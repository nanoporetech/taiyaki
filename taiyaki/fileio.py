from copy import deepcopy
from itertools import islice
import numpy as np
import os
import warnings

from gzip import open as gzopen
from bz2 import BZ2File as bzopen

from taiyaki.iterators import empty_iterator


_fval = {k: k for k in ['i', 'f', 'd', 's']}
_fval['b'] = 'i'


def _numpyfmt(a):
    """Return a list of formats with which to output a numpy array

    Args:
        a (numpy recarray) : numpy structured array

    Returns:
        list of strs : each is a format string to be used when
                       printing one of the columns of a
    """
    fmt = (np.dtype(s[1]).kind.lower() for s in a.dtype.descr)
    return ['%' + _fval.get(f, f) for f in fmt]


def file_has_fields(fname, fields=None):
    """Check that a tsv file has given fields

    Args:
        fname (str): filename to read. If the filename extension is
                     gz or bz2, the file is first decompressed.
        fields (list of str): list of required fields.

    Returns:
        bool : does the file have the fields?
    """

    # Allow a quick return
    req_fields = deepcopy(fields)
    if isinstance(req_fields, str):
        req_fields = [fields]
    if req_fields is None or len(req_fields) == 0:
        return True
    req_fields = set(req_fields)

    inspector = open
    ext = os.path.splitext(fname)[1]
    if ext == '.gz':
        inspector = gzopen
    elif ext == '.bz2':
        inspector = bzopen

    has_fields = None
    with inspector(fname, 'r') as fh:
        present_fields = set(fh.readline().rstrip('\n').split('\t'))
        has_fields = req_fields.issubset(present_fields)
    return has_fields


def read_chunks(fname, n_lines, n_chunks=None, header=True):
    """Yield successive chunks of a file

    Args:
        fname (str): file to read
        n_lines (int): number of lines per chunk
        n_chunks (int): number of chunks to read
        header (bool): if True one line is added to first chunk

    Yields:
        str : a chunk of the file
    """
    with open(fname) as fh:
        first = True
        yielded = 0
        while True:
            n = n_lines
            if first and header:
                n += 1
            sl = islice(fh, n)
            is_empty, sl = empty_iterator(sl)
            if is_empty:
                break
            else:
                yield sl
                yielded += 1
                if n_chunks is not None and yielded == n_chunks:
                    break


def take_a_peak(fname, n_lines=4):
    """Read the head of a file

    Args:
        fname (str): file to read
        n_lines (int): number of lines to read

    Yields:
        str : line of the file
    """
    with open(fname, 'r') as fh:
        for l in islice(fh, n_lines):
            yield l


def savetsv(fname, X, header=True):
    """Save a structured array to a .tsv file

    Args:
        fname (str or file handle) : where to save
            If the filename ends in ``.gz``, the file is automatically
            saved in compressed gzip format.  `loadtxt` understands
            gzipped files transparently.
        X (array_like) : Data to be saved to a text file.
    """
    if header:
        header = '\t'.join(X.dtype.names)
    else:
        header = ''
    fmt = '\t'.join(_numpyfmt(X))
    np.savetxt(fname, X, fmt=fmt, header=header, comments='', delimiter='\t')


def readtsv(fname, fields=None, **kwargs):
    """Read a tsv file into a numpy array with required field checking

    Args:
        fname (str): filename to read. If the filename extension is
                    gz or bz2, the file is first decompressed.
        fields (list of str) : list of required fields.

    Returns:
        numpy recarray : structured array containing data
    """

    if not file_has_fields(fname, fields):
        raise KeyError(
            'File {} does not contain requested required fields {}'.format(fname, fields))

    for k, v in {'names': True, 'delimiter': '\t', 'dtype': None, 'encoding': None}.items():
        if not (k in kwargs):
            kwargs[k] = v
    table = np.genfromtxt(fname, **kwargs)
    #  Numpy tricks to force single element to be array of one row
    return table.reshape(-1)


def readchunkedtsv(fname, chunk_size=100, **kwargs):
    """Read chunks of a .tsv file at a time.

    Args:
        fname (str): file to read
        chunk_size (int): length of resultant chunks
        **kwargs (dict): kwargs passed on to np.genfromtxt

    Yields:
        numpy recarray : structured array
    """
    for k in ['names', 'delimiter', 'dtype']:
        kwargs.pop(k, None)

    prototype = readtsv(take_a_peak(fname, chunk_size))
    dtype = prototype.dtype

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for i, chunk in enumerate(read_chunks(fname, chunk_size)):
            names = True if i == 0 else None
            try:
                yield np.genfromtxt(chunk, names=names, delimiter='\t', dtype=dtype, **kwargs)
            except:
                break
