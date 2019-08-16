import datetime
import hashlib
import imp
import numpy as np
import os
import platform
import sys
import torch

from taiyaki import __version__
from taiyaki.fileio import readtsv


def _load_python_model(model_file, **model_kwargs):
    netmodule = imp.load_source('netmodule', model_file)
    network = netmodule.network(**model_kwargs)
    return network


def save_model(network, output, index=None, model_skeleton=None):
    """
    Save a trained model.

    Two versions are saved: the checkpoint file,
    which is a pickled version of the model as a class instance, and
    the params file which saves a state_dict.

    : param network  : a pytorch model
    : param output   : an output base (files will be saved as
                       <output>model_checkpoint_XXXXX.checkpoint
                       and <output>model_checkpoint_XXXXX.checkpoint.params
                       where XXXX is either an index number or 'final').
    : param index    : if this is supplied, then an index number is used.
                       otherwise 'final'.
    : param model_skeleton : a pytorch model having the class
                         structure that we want to use for saving.
                         For use with DistributedDataParallel (see
                         below)

    .. note ::
    The model_skeleton arg should be used when saving a
    DistributedDataParallel model.
    If model_skeleton is specified, then the parameters from the
    .module attribute in the <network> to be saved are placed into the
    model skeleton and the model skeleton is then saved.
    In this way, we can save a DistributedDataParallel model in the same
    format that we save ordinary models.
    """
    if model_skeleton is not None:
        model_skeleton.load_state_dict(network.module.state_dict())
        _network = model_skeleton
    else:
        _network = network
    if index is None:
        basename = 'model_final'
    else:
        basename = 'model_checkpoint_{:05d}'.format(index)

    model_file = os.path.join(output, basename + '.checkpoint')
    torch.save(_network, model_file)
    params_file = os.path.join(output, basename + '.params')
    torch.save(_network.state_dict(), params_file)


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


def guess_model_stride(net, input_shape=(720, 1, 1)):
    """ Infer the stride of a pytorch network by running it on some test input.
    """
    net_device = next(net.parameters()).device
    out = net(torch.zeros(input_shape).to(net_device))
    return int(round(input_shape[0] / out.size()[0]))


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


class WindowedExpSmoother(object):
    """ Smooth values using exponential decay over a fixed range of values

    :param alpha: exponential decay factor
    :param n_vals: maximum number of values in reported smoothed value
    """
    def __init__(self, alpha=0.95, n_vals=100):
        assert 0.0 <= alpha <= 1.0, (
            "Alpha was {}, should be between 0.0 and 1.0.\n".format(
                alpha))
        self.alpha = alpha
        self.weights = np.power(alpha, np.arange(n_vals))
        self.vals = np.full(n_vals, np.NAN)
        self.n_valid_vals = 0
        return

    @property
    def value(self):
        if self.n_valid_vals == 0: return np.NAN
        return np.average(
            self.vals[:self.n_valid_vals],
            weights=self.weights[:self.n_valid_vals])

    def update(self, val):
        self.vals[1:] = self.vals[:-1]
        self.vals[0] = val
        self.n_valid_vals += 1
        return


class Logger(object):

    def __init__(self, log_file_name=None, quiet=False):
        """Open file for logging training results.

        :param log_file_name: If log_file_name is None, then no file is used.
        :param quiet: If quiet = False, then log entries also echoed to stdout."""
        #
        # Can't have unbuffered text I/O at the moment hence 'b' mode below.
        # See currently open issue http://bugs.python.org/issue17404
        #
        if log_file_name is None:
            self.fh = None
        else:
            self.fh = open(log_file_name, 'wb', 0)

        self.quiet = quiet

    def write(self, message):
        if not self.quiet:
            sys.stdout.write(message)
            sys.stdout.flush()
        if self.fh is None:
            return
        try:
            self.fh.write(message.encode('utf-8'))
        except IOError as e:
            print("Failed to write to log\n Message: {}\n Error: {}".format(message, repr(e)))


class BatchLog:
    """Used to record three-column tsv file containing
    loss, gradient norm and gradient norm cap
    for every training step"""

    def __init__(self, output_dir, filename='batch.log'):
        """Open log in output_dir with given filename and write header line.

        :param output_dir: output directory
        :param filename: filename to use for batch log. Do nothing if filename is None.
        """
        # Can't have unbuffered text I/O at the moment hence 'b' mode below.
        # See currently open issue http://bugs.python.org/issue17404
        log_file_name = os.path.join(output_dir, filename)
        self.fh = open(log_file_name, 'wb', 0)
        self.write("loss\tgradientnorm\tgradientcap\n")

    def write(self, s):
        """Write a string to the log.
        """
        self.fh.write(s.encode('utf-8'))

    def record(self, loss, gradientnorm, gradientcap, nonestring="NaN"):
        """Write loss, gradient and cap to a row of the log.

        If gradientcap is None, then write nonestring in its place."""
        self.write("{:5.4f}\t{:5.4f}\t".format(loss, gradientnorm))
        if gradientcap is None:
            self.write("{}\n".format(nonestring))
        else:
            self.write("{:5.4f}\n".format(gradientcap))


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

    def __init__(self, fh=sys.stderr, every=1, maxlen=50, quiet=False):
        assert maxlen > 0
        self._count = 0
        self.every = every
        self._line_len = maxlen
        self.fh = fh
        self.quiet = quiet

    def step(self):
        self._count += 1
        if not self.quiet:
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


class open_file_or_stdout():
    """  Simple context manager that acts like `open`

    If filename is None, uses stdout.

    :param filename: Name or file to open, or None for stdout
    """
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        if self.filename is None:
            self.fh = sys.stdout
        else:
            self.fh = open(self.filename, 'w')
        return self.fh

    def __exit__(self, *args):
        if self.filename is not None:
            self.fh.close()


def set_torch_device(device):
    """  Set Pytorch device

    :param device: device string or cuda device number. E.g. 'cpu', 1, 'cuda:1'

    Raises exception if cuda device requested but cuda is not available
    """
    device = torch.device(device)
    if device.type == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        else:
            raise ValueError('GPU device requested but cannot be set (PyTorch not compiled with CUDA enabled?)')
    return device


def prepare_outdir(outdir, overwrite=False):
    """  Creates output directory
    Will overwrite if overwrite is true
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif not overwrite:
        raise FileExistsError('\"{}\" exists but --overwrite is false\n'.format(outdir))

    if not os.path.isdir(outdir):
        raise NotADirectoryError('\"{}\" is not directory'.format(outdir))


def formatted_env_info(device):
    """  Collect and format information about environment

    :param device: `torch.device`

    :returns: formatted string
    """
    info = ['* Taiyaki version {}'.format(__version__),
            '* Platform is {}'.format(platform.platform()),
            '* PyTorch version {}'.format(torch.__version__),
            '* CUDA version {} on device {}'.format(torch.version.cuda,
                                                    torch.cuda.get_device_name(device))
            if device.type == 'cuda' else '* Running on CPU',
            '* Command line:',
            '* "' + ' '.join(sys.argv),
            '* Started on {}'.format(datetime.datetime.now())]
    return '\n'.join(info) + '\n'
