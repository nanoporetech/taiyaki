import argparse

from taiyaki import __version__
from taiyaki.cmdargs import (
    AutoBool, DeviceAction, display_version_and_exit, FileExists, Maybe,
    NonNegative, ParseToNamedTuple, Positive)


def get_train_flipflop_parser():
    parser = argparse.ArgumentParser(
        description='Train flip-flop neural network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    mdl_grp = parser.add_argument_group('Model Arguments')
    mdl_grp.add_argument(
        '--size', default=256, metavar='neurons',
        type=Positive(int), help='Base layer size for model')
    mdl_grp.add_argument(
        '--stride', default=5, metavar='samples',
        type=Positive(int), help='Stride for model')
    mdl_grp.add_argument(
        '--winlen', default=19, type=Positive(int),
        help='Length of window over data')

    trn_grp = parser.add_argument_group('Training Arguments')
    trn_grp.add_argument(
        '--adam', nargs=2, metavar=('beta1', 'beta2'),
        default=[0.9, 0.999], type=NonNegative(float),
        help='Parameters beta1, beta2 for Exponential Decay ' +
        'Adaptive Momentum')
    trn_grp.add_argument(
        '--eps', default=1e-6, metavar='adjustment',
        type=Positive(float), help='Small value to stabilise optimiser')
    trn_grp.add_argument(
        '--niteration', metavar='batches', type=Positive(int),
        default=100000, help='Maximum number of batches to train for')
    trn_grp.add_argument(
        '--weight_decay', default=0.0, metavar='penalty',
        type=NonNegative(float),
        help='Adam weight decay (L2 normalisation penalty)')
    trn_grp.add_argument(
        '--gradient_cap_fraction', default=0.05, metavar='f',
        type=Maybe(NonNegative(float)),
        help='Cap L2 norm of gradient so that a fraction f of gradients ' +
        'are capped. Use --gradient_cap_fraction None for no capping.')
    trn_grp.add_argument(
        '--lr_cosine_iters', default=90000, metavar='n', type=Positive(float),
        help='Learning rate decreases from max to min like cosine function ' +
        'over n batches')
    trn_grp.add_argument(
        '--lr_frac_decay', default=None, metavar='k', type=Positive(int),
        help='If specified, use fractional learning rate ' +
        'schedule, rate=lr_max*k/(k+t)')
    trn_grp.add_argument(
        '--lr_max', default=4.0e-3, metavar='rate', type=Positive(float),
        help='Max (and starting) learning rate')
    trn_grp.add_argument(
        '--lr_min', default=1.0e-4, metavar='rate', type=Positive(float),
        help='Min (and final) learning rate')
    trn_grp.add_argument(
        '--seed', default=None, metavar='integer', type=Positive(int),
        help='Set random number seed')
    trn_grp.add_argument(
        '--sharpen', default=(1.0, 1.0, 25000), nargs=3,
        metavar=('min', 'max', 'niter'), action=ParseToNamedTuple,
        type=(Positive(float), Positive(float), Positive(int)),
        help='Increase sharpening factor linearly from "min" to ' +
        '"max" over "niter" iterations')
    trn_grp.add_argument(
        '--warmup_batches', type=int, default=200,
        help='For the first n batches, warm up at a low learning rate.')
    trn_grp.add_argument(
        '--lr_warmup', type=float, default=None,
        help="Learning rate used for warmup. Defaults to lr_min")

    data_grp = parser.add_argument_group('Data Arguments')
    data_grp.add_argument(
        '--filter_max_dwell', default=10.0, metavar='multiple',
        type=Maybe(Positive(float)),
        help='Drop chunks with max dwell more than multiple of median ' +
        '(over chunks)')
    data_grp.add_argument(
        '--filter_mean_dwell', default=3.0, metavar='radius',
        type=Maybe(Positive(float)),
        help='Drop chunks with mean dwell more than radius deviations ' +
        'from the median (over chunks)')
    data_grp.add_argument(
        '--limit', default=None, type=Maybe(Positive(int)),
        help='Limit number of reads to process')
    data_grp.add_argument(
        '--reverse', default=False, action=AutoBool,
        help='Reverse input sequence and current')
    data_grp.add_argument(
        '--sample_nreads_before_filtering', metavar='n',
        type=NonNegative(int), default=100000,
        help='Sample n reads to decide on bounds for filtering before ' +
        'training. Set to 0 to do all.')
    data_grp.add_argument(
        '--chunk_len_min', default=3000, metavar='samples', type=Positive(int),
        help='Min length of each chunk in samples (chunk lengths are ' +
        'random between min and max)')
    data_grp.add_argument(
        '--chunk_len_max', default=8000, metavar='samples', type=Positive(int),
        help='Max length of each chunk in samples (chunk lengths are ' +
        'random between min and max)')
    data_grp.add_argument(
        '--include_reporting_strands', default=False, action=AutoBool,
        help='Include reporting strands in training. Default: Hold ' +
        'training strands out of training.')
    data_grp.add_argument(
        '--input_strand_list', default=None, action=FileExists,
        help='Strand summary file containing column read_id. Filenames in ' +
        'file are ignored.')
    data_grp.add_argument(
        '--min_sub_batch_size', default=128, metavar='chunks',
        type=Positive(int),
        help='Number of chunks to run in parallel per sub-batch for ' +
        'chunk_len = chunk_len_max. Actual length of sub-batch used is ' +
        '(min_sub_batch_size * chunk_len_max / chunk_len).')
    data_grp.add_argument(
        '--reporting_percent_reads', default=1, metavar='sub_batches',
        type=Positive(float),
        help='Percent of reads to use for std loss reporting')
    data_grp.add_argument(
        '--reporting_strand_list', action=FileExists,
        help='Strand summary file containing column read_id. All other ' +
        'fields are ignored. If not provided reporting strands will be ' +
        'randomly selected.')
    data_grp.add_argument(
        '--reporting_sub_batches', default=10, metavar='sub_batches',
        type=Positive(int),
        help='Number of sub-batches to use for std loss reporting')
    data_grp.add_argument(
        '--standardize', default=True, action=AutoBool,
        help='Standardize currents for each read')
    data_grp.add_argument(
        '--sub_batches', default=1, metavar='sub_batches', type=Positive(int),
        help='Number of sub-batches per batch')

    cmp_grp = parser.add_argument_group('Compute Arguments')
    cmp_grp.add_argument(
        '--device', default='cpu', action=DeviceAction,
        help='Integer specifying which GPU to use, or "cpu" to use CPU only. '
        'Other accepted formats: "cuda" (use default GPU), "cuda:2" '
        'or "cuda2" (use GPU 2).')
    # Argument local_rank is used only by when the script is run in multi-GPU
    # mode using torch.distributed.launch. See the README.
    cmp_grp.add_argument(
        '--local_rank', type=int, default=None, help=argparse.SUPPRESS)

    out_grp = parser.add_argument_group('Output Arguments')
    out_grp.add_argument(
        '--full_filter_status', default=False, action=AutoBool,
        help='Output full chunk filtering statistics. Default: only ' +
        'proportion of filtered chunks.')
    out_grp.add_argument(
        '--outdir', default='training',
        help='Output directory, created when run.')
    out_grp.add_argument(
        '--overwrite', default=False, action=AutoBool,
        help='Whether to overwrite any output files')
    out_grp.add_argument(
        '--quiet', default=False, action=AutoBool,
        help="Don't print progress information to stdout")
    out_grp.add_argument(
        '--save_every', metavar='x', type=Positive(int), default=1000,
        help='Save model every x batches')

    mod_grp = parser.add_argument_group('Modified Base Arguments')
    mod_grp.add_argument(
        '--mod_factor', type=float, default=1.0,
        help='Relative modified base weight (compared to canonical ' +
        'transitions) in loss/gradient (only applicable for modified base ' +
        'models).')

    misc_grp = parser.add_argument_group('Miscellaneous  Arguments')
    misc_grp.add_argument(
        '--version', nargs=0, action=display_version_and_exit,
        metavar=__version__,
        help='Display version information.')

    parser.add_argument(
        'model', action=FileExists,
        help='File to read python model (or checkpoint) from')
    parser.add_argument(
        'input', action=FileExists,
        help='file containing mapped reads')

    return parser
