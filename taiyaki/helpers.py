from Bio import SeqIO
from collections import Mapping, Sequence
import hashlib
import imp
import numpy as np
import os
import re
import sys
import torch

from taiyaki.fileio import readtsv
from taiyaki.variables import DEFAULT_ALPHABET




def _load_python_model(model_file, **model_kwargs):
    netmodule = imp.load_source('netmodule', model_file)
    network = netmodule.network(**model_kwargs)
    return network


def save_model(network, output, index=None):
    if index is None:
        basename = 'model_final'
    else:
        basename = 'model_checkpoint_{:05d}'.format(index)

    model_file = os.path.join(output, basename + '.checkpoint')
    torch.save(network, model_file)
    params_file = os.path.join(output, basename + '.params')
    torch.save(network.state_dict(), params_file)


def load_model(model_file, params_file=None, **model_kwargs):
    _, extension = os.path.splitext(model_file)

    if extension == '.py':
        network = _load_python_model(model_file, **model_kwargs)
    else:
        network = torch.load(model_file, map_location='cpu')

    if params_file is not None:
        param_dict = torch.load(params_file, map_location='cpu')
        network.load_state_dict(param_dict)

    return network


def guess_model_stride(net, input_shape=(720, 1, 1), device='cpu'):
    """ Infer the stride of a pytorch network by running it on some test input.
    Assume that net is already able to accept input on device specified"""
    out = net(torch.zeros(input_shape).to(device))
    return int(round(input_shape[0] / out.size()[0]))


def objwalk(obj, types=(object,), path=(), memo=None):
    """Recursively walk a python object yielding instances of specified types

    Ripped off from: https://goo.gl/brwMce
    """
    if memo is None:
        memo = set()

    if isinstance(obj, Mapping):
        children = obj.items()
    elif isinstance(obj, Sequence) and not isinstance(obj, (bytes, str)):
        children = enumerate(obj)
    elif hasattr(obj, "__dict__"):
        children = vars(obj).items()
    else:
        children = []

    if id(obj) not in memo:
        memo.add(id(obj))

        if isinstance(obj, types):
            yield path, obj

        for (path_component, value) in children:
            for res in objwalk(value, types, path + (path_component,), memo):
                yield res


def set_at_path(obj, path, val):
    """Set value on object following a path as built by objwalk"""
    if len(path) == 1:
        if isinstance(obj, Mapping):
            obj[path[0]] = val
        elif isinstance(obj, Sequence) and not isinstance(obj, (bytes, str)):
            obj[path[0]] = val
        else:
            setattr(obj, path[0], val)
    elif len(path) > 1:
        if isinstance(obj, Mapping):
            set_at_path(obj[path[0]], path[1:], val)
        elif isinstance(obj, Sequence) and not isinstance(obj, (bytes, str)):
            set_at_path(obj[path[0]], path[1:], val)
        else:
            set_at_path(obj.__dict__[path[0]], path[1:], val)


def get_kwargs(args, names):
    kwargs = {}
    for name in names:
        kwargs[name] = getattr(args, name)
    return kwargs


def trim_array(x, from_start, from_end):
    assert from_start >= 0
    assert from_end >= 0

    from_end = None if from_end == 0 else -from_end
    return x[from_start:from_end]


def subsample_array(x, length):
    if length is None:
        return x
    assert len(x) > length
    startpos = np.random.randint(0, len(x) - length + 1)
    return x[startpos : startpos + length]


def fasta_file_to_dict(fasta_file_name, allow_N=False, alphabet=DEFAULT_ALPHABET):
    """Load records from fasta file as a dictionary"""
    has_nonalphabet = re.compile('[^{}]'.format(alphabet))

    references = {}
    with open(fasta_file_name, 'r') as fh:
        for ref in SeqIO.parse(fh, 'fasta'):
            refseq = str(ref.seq)
            if len(refseq) == 0:
                continue
            if not allow_N and re.search(has_nonalphabet, refseq) is not None:
                continue
            references[ref.id] = refseq.encode('utf-8')

    return references


class ReadIndex(dict):
    """dict subclass mapping from read names (str or bytes) to index (int)"""
    @staticmethod
    def _force_str(s):
        return s.decode('utf-8') if isinstance(s, bytes) else s

    def __init__(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        items = ((ReadIndex._force_str(basename), read_id) for basename, read_id in data.items())
        super().__init__(items)

    def to_numpy(self):
        """Convert ReadIndex instance into a numpy structured array"""
        max_key_len = max(map(len, self.keys()))
        key_dtype = 'S{}'.format(max_key_len)
        dtype = [('read_name', key_dtype), ('read_id', '<u4')]
        return np.fromiter(self.items(), dtype=dtype)


def get_column_from_tsv(tsv_file_name, column):
    '''Load a column from a csv file'''

    if tsv_file_name is not None:
        data = readtsv(tsv_file_name, encoding='utf-8')
        assert column in data.dtype.names, "Strand file does not contain required field {}".format(column)
        return [x for x in data[column]]


def get_file_names(csv_file_name):
    '''Load strand file names from a csv file'''
    return get_column_from_tsv(csv_file_name, 'filename')


def get_read_ids(tsv_file_name):
    '''Load strand file names from a tsv file'''
    return get_column_from_tsv(tsv_file_name, 'read_id')


class ExponentialSmoother(object):

    def __init__(self, factor, val=0.0, weight=1e-30):
        assert 0.0 <= factor <= 1.0, "Smoothing factor was {}, should be between 0.0 and 1.0.\n".format(factor)
        self.factor = factor
        self.val = val
        self.weight = weight

    @property
    def value(self):
        return self.val / self.weight

    def update(self, val, weight=1.0):
        self.val = self.factor * self.val + (1.0 - self.factor) * val
        self.weight = self.factor * self.weight + (1.0 - self.factor) * weight


class Logger(object):

    def __init__(self, log_file_name, quiet=False):
        #
        # Can't have unbuffered text I/O at the moment hence 'b' mode below.
        # See currently open issue http://bugs.python.org/issue17404
        #
        self.fh = open(log_file_name, 'wb', 0)

        self.quiet = quiet

    def write(self, message):
        if not self.quiet:
            sys.stdout.write(message)
            sys.stdout.flush()
        try:
            self.fh.write(message.encode('utf-8'))
        except IOError as e:
            print("Failed to write to log\n Message: {}\n Error: {}".format(message, repr(e)))


def file_md5(filename, nblock=1024):
    hasher = hashlib.md5()
    block_size = nblock * hasher.block_size
    with open(filename, 'rb') as fh:
        for blk in iter((lambda: fh.read(block_size)), b''):
            hasher.update(blk)
    return hasher.hexdigest()


COLOURS = [91, 93, 95, 92, 35, 33, 94]


class Progress(object):
    """A dotty way of showing progress"""

    def __init__(self, fh=sys.stderr, every=1, maxlen=50):
        assert maxlen > 0
        self._count = 0
        self.every = every
        self._line_len = maxlen
        self.fh = fh

    def step(self):
        self._count += 1
        if self.count % self.every == 0:
            dotcount = self.count // self.every
            self.fh.write('\033[1;{}m.\033[m'.format(COLOURS[dotcount % len(COLOURS)]))
            if dotcount % self.line_len == 0:
                self.fh.write('{:8d}\n'.format(self.count))
            self.fh.flush()

    @property
    def line_len(self):
        return self._line_len

    @property
    def count(self):
        return self._count

    @property
    def nline(self):
        return (self.count // self.every) // self.line_len

    @property
    def is_newline(self):
        return self.count % (self.dotcount * self.line_len) == 0
