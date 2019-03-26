# Command-line args used in more than one script defined here

from taiyaki.cmdargs import (AutoBool, DeviceAction, FileExists, Maybe, NonNegative,
                             ParseToNamedTuple, Positive, display_version_and_exit)
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

    ############################################################################
    #
    # Optional arguments
    #
    ############################################################################

    if 'adam' in arglist:
        parser.add_argument('--adam', nargs=2, metavar=('beta1', 'beta2'),
                            default=(0.9, 0.999), type=(NonNegative(float),
                                                        NonNegative(float)), action=ParseToNamedTuple,
                            help='Parameters beta1, beta2 for Exponential Decay Adaptive Momentum')

    if 'chunk_logging_threshold' in arglist:
        parser.add_argument('--chunk_logging_threshold', default=10.0, metavar='multiple',
                            type=NonNegative(float),
                            help='If loss > (threshold * smoothed loss) for a batch, then log chunks to ' +
                                 'output/chunklog.tsv. Set to zero to log all, including rejected chunks')

    if 'device' in arglist:
        parser.add_argument('--device', default='cpu', action=DeviceAction,
                            help='Integer specifying which GPU to use, or "cpu" to use CPU only. '
                            'Other accepted formats: "cuda" (use default GPU), "cuda:2" '
                            'or "cuda2" (use GPU 2).')

    if 'filter_max_dwell' in arglist:
        parser.add_argument('--filter_max_dwell', default=10.0, metavar='multiple',
                            type=Maybe(Positive(float)),
                            help='Drop chunks with max dwell more than multiple of median (over chunks)')
        
    if 'filter_mean_dwell' in arglist:
        parser.add_argument('--filter_mean_dwell', default=3.0, metavar='radius',
                            type=Maybe(Positive(float)),
                            help='Drop chunks with mean dwell more than radius deviations from the median (over chunks)')

    if 'input_strand_list' in arglist:
        parser.add_argument('--input_strand_list', default=None, action=FileExists,
                            help='Strand list TSV file with columns filename_fast5 or read_id or both')

    if 'jobs' in arglist:
        parser.add_argument('--jobs', default=1, metavar='n', type=Positive(int),
                            help='Number of threads to use when processing data')

    if 'limit' in arglist:
        parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
                            help='Limit number of reads to process')

    if 'niteration' in arglist:
        parser.add_argument('--niteration', metavar='batches', type=Positive(int),
                            default=50000, help='Maximum number of batches to train for')

    if 'overwrite' in arglist:
        parser.add_argument('--overwrite', default=False, action=AutoBool,
                            help='Whether to overwrite any output files')

    if 'quiet' in arglist:
        parser.add_argument('--quiet', default=False, action=AutoBool,
                            help="Don't print progress information to stdout")

    if 'recursive' in arglist:
        parser.add_argument('--recursive', default=False, action=AutoBool,
                            help='Search for fast5s recursively within ' +
                            'input_folder. Default only search first level.')

    if 'sample_nreads_before_filtering' in arglist:
        parser.add_argument('--sample_nreads_before_filtering', metavar='n', type=NonNegative(int), default=1000,
                            help='Sample n reads to decide on bounds for filtering before training. Set to 0 to do all.')

    if 'save_every' in arglist:
        parser.add_argument('--save_every', metavar='x', type=Positive(int), default=5000,
                            help='Save model every x batches')

    if 'version' in arglist:
        parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
                            help='Display version information.')

    if 'weight_decay' in arglist:parser.add_argument('--weight_decay', default=0.0, metavar='penalty',
                                                     type=NonNegative(float),
                                                     help='Adam weight decay (L2 normalisation penalty)')




    ############################################################################
    #
    # Positional arguments
    #
    ############################################################################

    if 'input_folder' in arglist:
        parser.add_argument('input_folder', action=FileExists,
                            help='Directory containing single or multi-read fast5 files')
