# Command-line args used in more than one script defined here

from taiyaki.cmdargs import (AutoBool, DeviceAction, FileAbsent, FileExists,
                             Maybe, NonNegative, ParseToNamedTuple, Positive,
                             display_version_and_exit)
from taiyaki.constants import DEFAULT_ALPHABET
from taiyaki import __version__


def add_common_command_args(parser, arglist):
    """Given an argparse parser object and a list of keys such as
    ['input_strand_list', 'jobs'], add these command line args
    to the parser.

    Not all command line args used in the package are
    included in this func: only those that are used by more than
    one script and which have the same defaults.

    Some args are positional and some are optional.
    The optional ones are listed first below."""

    ALLOWED_ARGS = dict([
        #  Optional arguments
        ('adam',  lambda :
            parser.add_argument('--adam', nargs=2, metavar=('beta1', 'beta2'),
                                default=[0.9, 0.999], type=NonNegative(float),
                                help='Parameters beta1, beta2 for Exponential Decay Adaptive Momentum')),

        ('alphabet', lambda :
            parser.add_argument('--alphabet', default=DEFAULT_ALPHABET,
                                help='Canonical base alphabet')),

        ('device', lambda :
            parser.add_argument('--device', default='cpu', action=DeviceAction,
                                help='Integer specifying which GPU to use, or "cpu" to use CPU only. '
                                'Other accepted formats: "cuda" (use default GPU), "cuda:2" '
                                'or "cuda2" (use GPU 2).')),

        ('eps', lambda :
            parser.add_argument('--eps', default=1e-6, metavar='adjustment',
                                type=Positive(float), help='Small value to stabilise optimiser')),

        ('filter_max_dwell', lambda :
            parser.add_argument('--filter_max_dwell', default=10.0, metavar='multiple',
                                type=Maybe(Positive(float)),
                                help='Drop chunks with max dwell more than multiple of median (over chunks)')),

        ('filter_mean_dwell', lambda :
            parser.add_argument('--filter_mean_dwell', default=3.0, metavar='radius',
                                type=Maybe(Positive(float)),
                                help='Drop chunks with mean dwell more than radius deviations from the median (over chunks)')),

        ('input_strand_list', lambda :
            parser.add_argument('--input_strand_list', default=None, action=FileExists,
                                help='Strand list TSV file with columns filename_fast5 or read_id or both')),

        ('jobs', lambda :
            parser.add_argument('--jobs', default=1, metavar='n', type=Positive(int),
                                help='Number of threads to use when processing data')),

        ('limit', lambda :
            parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
                                help='Limit number of reads to process')),

        ('niteration', lambda :
            parser.add_argument('--niteration', metavar='batches', type=Positive(int),
                                default=100000, help='Maximum number of batches to train for')),

        ('outdir', lambda :
            parser.add_argument('--outdir', default='training',
                                help='Output directory, created when run.')),

        ('output', lambda :
            parser.add_argument('--output', default=None, metavar='filename',
                                action=FileAbsent, help='Write output to file')),

        ('overwrite', lambda :
            parser.add_argument('--overwrite', default=False, action=AutoBool,
                                help='Whether to overwrite any output files')),

        ('quiet', lambda :
            parser.add_argument('--quiet', default=False, action=AutoBool,
                                help="Don't print progress information to stdout")),

        ('recursive', lambda :
            parser.add_argument('--recursive', default=True, action=AutoBool,
                                help='Search for fast5s recursively within ' +
                                'input_folder. Otherwise only search first level.')),

        ('reverse', lambda :
            parser.add_argument('--reverse', default=False, action=AutoBool,
                                help='Reverse input sequence and current')),

        ('sample_nreads_before_filtering', lambda :
            parser.add_argument('--sample_nreads_before_filtering', metavar='n',
                                type=NonNegative(int), default=100000,
                                help='Sample n reads to decide on bounds for filtering before training. Set to 0 to do all.')),

        ('save_every', lambda :
            parser.add_argument('--save_every', metavar='x', type=Positive(int), default=1000,
                                help='Save model every x batches')),

        ('version', lambda :
            parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
                                help='Display version information.')),

        ('weight_decay', lambda :
            parser.add_argument('--weight_decay', default=0.0, metavar='penalty',
                                type=NonNegative(float),
                                help='Adam weight decay (L2 normalisation penalty)')),

        #  Positional arguments
        ('input_folder', lambda :
            parser.add_argument('input_folder', action=FileExists,
                                help='Directory containing single or multi-read fast5 files'))
    ])


    args_required = frozenset(arglist)
    args_allowed = frozenset(ALLOWED_ARGS.keys())
    assert len(args_required - args_allowed) == 0, \
        'Unsupported argument(s) found : {}'.format(args_required - args_allowed)

    for arg in args_required:
        ALLOWED_ARGS[arg]()
