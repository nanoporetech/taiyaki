#!/usr/bin/env python3
from collections import defaultdict, namedtuple
import numpy as np
import os
from shutil import copyfile
import sys
import time

import torch

from taiyaki import (
    chunk_selection, constants, ctc, flipflopfings, helpers, layers,
    mapped_signal_files, maths, optim, signal_mapping)
from taiyaki.constants import DOTROWLENGTH
from taiyaki.helpers import guess_model_stride
from _bin_argparse import get_train_flipflop_parser


# Note also set flag in taiyaki/ctc/ctc.pyx when profiling code
_DO_PROFILE = False
_DEBUG_MULTIGPU = False

RESOURCE_INFO = namedtuple('RESOURCE_INFO', (
    'is_multi_gpu', 'is_lead_process', 'device'))

MOD_INFO = namedtuple('MOD_INFO', ('mod_factor', 'mod_cat_weights'))

NETWORK_METADATA = namedtuple('NETWORK_METADATA', (
    'reverse', 'standardize', 'is_cat_mod', 'can_mods_offsets',
    'can_labels', 'mod_labels'))
NETWORK_METADATA.__new__.__defaults__ = (None, None, None)
NETWORK_INFO = namedtuple('NETWORK_INFO', (
    'net', 'net_clone', 'metadata', 'stride'))

OPTIM_INFO = namedtuple('OPTIM_INFO', (
    'optimiser', 'lr_warmup', 'lr_scheduler', 'rolling_quantile'))

TRAIN_PARAMS = namedtuple('TRAIN_PARAMS', (
    'niteration', 'sharpen', 'chunk_len_min', 'chunk_len_max',
    'min_sub_batch_size', 'sub_batches', 'save_every',
    'outdir', 'full_filter_status'))

LOG_POLKA_TMPLT = (
    ' {:5d} {:7.5f} {:7.5f}  {:5.2f}s ({:.2f} ksample/s {:.2f} ' +
    'kbase/s) lr={:.2e}')


def parse_network_metadata(network):
    if layers.is_cat_mod_model(network):
        return NETWORK_METADATA(
            network.metadata['reverse'], network.metadata['standardize'],
            True, network.sublayers[-1].can_mods_offsets,
            network.sublayers[-1].can_labels,
            network.sublayers[-1].mod_labels)
    return NETWORK_METADATA(
        network.metadata['reverse'], network.metadata['standardize'], False)


def compute_grad_norm(network, norm_type=2):
    """ Compute the norm of the gradients in a network. Code adapted from
    `torch.nn.utils.clip_grad_norm_`, but without clipping.

    Args:
        network: A taiyaki neural network object.
        norm_type: Norm type as defined in `torch.norm`.

    Returns:
        Float norm computed from network
    """
    parameters = list(filter(lambda p: p.grad is not None,
                             network.parameters()))
    if len(parameters) == 0:
        return 0.0
    return float(torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type)
                     for p in parameters]), norm_type))


# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(description='Train flip-flop neural network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

mdl_grp = parser.add_argument_group('Model Arguments')
mdl_grp.add_argument('--size', default=256, metavar='neurons',
                     type=Positive(int), help='Base layer size for model')
mdl_grp.add_argument('--stride', default=5, metavar='samples',
                     type=Positive(int), help='Stride for model')
mdl_grp.add_argument('--winlen', default=19, type=Positive(int),
                     help='Length of window over data')

trn_grp = parser.add_argument_group('Training Arguments')
add_common_command_args(
    trn_grp, """adam eps niteration weight_decay""".split())
trn_grp.add_argument('--gradient_cap_fraction', default=0.05, metavar='f',
                     type=Maybe(NonNegative(float)),
                     help='Cap L2 norm of gradient so that a fraction f of ' +
                     'gradients are capped. ' +
                     'Use --gradient_cap_fraction None for no capping.')
trn_grp.add_argument('--lr_cosine_iters', default=90000, metavar='n',
                     type=Positive(float),
                     help='Learning rate decreases from max to min ' +
                     'like cosine function over n batches')
trn_grp.add_argument('--lr_frac_decay', default=None, metavar='k',
                     type=Positive(int),
                     help='If specified, use fractional learning rate ' +
                     'schedule, rate=lr_max*k/(k+t)')
trn_grp.add_argument('--lr_max', default=4.0e-3, metavar='rate',
                     type=Positive(float),
                     help='Max (and starting) learning rate')
trn_grp.add_argument('--lr_min', default=1.0e-4, metavar='rate',
                     type=Positive(float), help='Min (and final) learning rate')
trn_grp.add_argument('--seed', default=None, metavar='integer',
                     type=Positive(int),
                     help='Set random number seed and deterministic flags ' +
                     'in pytorch.')
trn_grp.add_argument('--sharpen', default=(1.0, 1.0, 25000), nargs=3,
                     metavar=('min', 'max', 'niter'), action=ParseToNamedTuple,
                     type=(Positive(float), Positive(float), Positive(int)),
                     help='Increase sharpening factor linearly from "min" to ' +
                          '"max" over "niter" iterations')
trn_grp.add_argument('--warmup_batches', type=int, default=200,
                     help='For the first n batches, ' +
                     'warm up at a low learning rate.')
trn_grp.add_argument('--lr_warmup', type=float, default=None,
                     help="Learning rate used for warmup. Defaults to lr_min")

data_grp = parser.add_argument_group('Data Arguments')
add_common_command_args(data_grp, """filter_max_dwell filter_mean_dwell limit
                                     reverse sample_nreads_before_filtering""".split())
data_grp.add_argument('--chunk_len_min', default=3000, metavar='samples',
                      type=Positive(int),
                      help='Min length of each chunk in samples' +
                      ' (chunk lengths are random between min and max)')
data_grp.add_argument('--chunk_len_max', default=8000, metavar='samples',
                      type=Positive(int),
                      help='Max length of each chunk in samples ' +
                      '(chunk lengths are random between min and max)')
data_grp.add_argument('--include_reporting_strands',
                      default=False, action=AutoBool,
                      help='Include reporting strands in training. Default: ' +
                      'Hold training strands out of training.')
data_grp.add_argument('--input_strand_list', default=None, action=FileExists,
                      help='Strand summary file containing column read_id. ' +
                      'Filenames in file are ignored.')
data_grp.add_argument('--min_sub_batch_size', default=128, metavar='chunks',
                      type=Positive(int),
                      help='Number of chunks to run in parallel per ' +
                      'sub-batch for chunk_len = chunk_len_max. Actual ' +
                      'length of sub-batch used is ' +
                      '(min_sub_batch_size * chunk_len_max / chunk_len).')
data_grp.add_argument('--reporting_percent_reads', default=1,
                      metavar='sub_batches', type=Positive(float),
                      help='Percent of reads to use for std loss reporting')
data_grp.add_argument('--reporting_strand_list', action=FileExists,
                      help='Strand summary file containing column read_id. ' +
                      'All other fields are ignored. If not provided ' +
                      'reporting strands will be randomly selected.')
data_grp.add_argument('--reporting_sub_batches', default=10,
                      metavar='sub_batches', type=Positive(int),
                      help='Number of sub-batches to use for std loss reporting')
data_grp.add_argument('--standardize', default=True, action=AutoBool,
                      help='Standardize currents for each read')
data_grp.add_argument('--sub_batches', default=1, metavar='sub_batches',
                      type=Positive(int),
                      help='Number of sub-batches per batch')

cmp_grp = parser.add_argument_group('Compute Arguments')
add_common_command_args(cmp_grp, set(("device",)))
# Argument local_rank is used only by when the script is run in multi-GPU
# mode using torch.distributed.launch. See the README.
cmp_grp.add_argument('--local_rank', type=int, default=None,
                     help=argparse.SUPPRESS)

out_grp = parser.add_argument_group('Output Arguments')
add_common_command_args(out_grp, """outdir overwrite quiet
                                    save_every""".split())
out_grp.add_argument('--full_filter_status', default=False, action=AutoBool,
                     help='Output full chunk filtering statistics. ' +
                     'Default: only proportion of filtered chunks.')

mod_grp = parser.add_argument_group('Modified Base Arguments')
mod_grp.add_argument('--mod_factor', type=float, default=1.0,
                     help='Relative modified base weight (compared to ' +
                     'canonical transitions) in loss/gradient (only ' +
                     'applicable for modified base models).')

misc_grp = parser.add_argument_group('Miscellaneous  Arguments')
add_common_command_args(misc_grp, set(("version",)))

parser.add_argument('model', action=FileExists,
                    help='File to read python model (or checkpoint) from')
parser.add_argument('input', action=FileExists,
                    help='file containing mapped reads')


def prepare_random_batches(device, read_data, batch_chunk_len, sub_batch_size,
                           target_sub_batches, alphabet_info, filter_params,
                           network, network_metadata, log,
                           select_strands_randomly=True, first_strand_index=0):
    total_sub_batches = 0
    if net_info.metadata.reverse:
        revop = np.flip
    else:
        revop = np.array

    while total_sub_batches < target_sub_batches:

        # Chunk_batch is a list of dicts
        chunk_batch, batch_rejections = chunk_selection.sample_chunks(
            read_data, sub_batch_size, batch_chunk_len, filter_params,
            standardize=net_info.metadata.standardize,
            select_strands_randomly=select_strands_randomly,
            first_strand_index=first_strand_index)
        first_strand_index += sum(batch_rejections.values())
        if len(chunk_batch) < sub_batch_size:
            log.write(('* Warning: only {} chunks passed filters ' +
                       '(asked for {}).\n').format(
                           len(chunk_batch), sub_batch_size))

        if not all(chunk.seq_len > 0.0 for chunk in chunk_batch):
            raise Exception('Error: zero length sequence')

        # Shape of input tensor must be:
        #     (timesteps) x (batch size) x (input channels)
        # in this case:
        #     batch_chunk_len x sub_batch_size x 1
        stacked_current = np.vstack([
            revop(chunk.current) for chunk in chunk_batch]).T
        indata = torch.tensor(stacked_current, device=device,
                              dtype=torch.float32).unsqueeze(2)

        # Prepare seqs, seqlens and (if necessary) mod_cats
        seqs, seqlens = [], []
        mod_cats = [] if net_info.metadata.is_cat_mod else None
        for chunk in chunk_batch:
            chunk_labels = revop(chunk.sequence)
            seqlens.append(len(chunk_labels))
            if net_info.metadata.is_cat_mod:
                chunk_mod_cats = np.ascontiguousarray(
                    net_info.metadata.mod_labels[chunk_labels])
                mod_cats.append(chunk_mod_cats)
                # convert chunk_labels to canonical base labels
                chunk_labels = np.ascontiguousarray(
                    net_info.metadata.can_labels[chunk_labels])
            chunk_seq = flipflopfings.flipflop_code(
                chunk_labels, alphabet_info.ncan_base)
            seqs.append(chunk_seq)

        seqs = torch.tensor(
            np.concatenate(seqs), dtype=torch.float32, device=device)
        seqlens = torch.tensor(seqlens, dtype=torch.long, device=device)
        if net_info.metadata.is_cat_mod:
            mod_cats = torch.tensor(
                np.concatenate(mod_cats), dtype=torch.long, device=device)

        total_sub_batches += 1

        yield indata, seqs, seqlens, mod_cats, sub_batch_size, batch_rejections


def calculate_loss(
        net_info, batch_gen, sharpen, mod_cat_weights=None,
        mod_factor_t=None, calc_grads=False):
    total_chunk_count = total_fval = total_samples = total_bases = \
        n_subbatches = 0
    rejection_dict = defaultdict(int)
    for (indata, seqs, seqlens, mod_cats, sub_batch_size,
         batch_rejections) in batch_gen:
        n_subbatches += 1
        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        total_chunk_count += sub_batch_size

        with torch.set_grad_enabled(calc_grads):
            outputs = net_info.net(indata)
            if net_info.metadata.is_cat_mod:
                lossvector = ctc.cat_mod_flipflop_loss(
                    outputs, seqs, seqlens, mod_cats,
                    net_info.metadata.can_mods_offsets,
                    mod_cat_weights, mod_factor_t, sharpen)
            else:
                lossvector = ctc.crf_flipflop_loss(
                    outputs, seqs, seqlens, sharpen)

            non_zero_seqlens = (seqlens > 0.0).float().sum()
            # In multi-GPU mode, gradients are synchronised when
            # loss.backward() is called. We need to make sure we are
            # calculating a gradient that can be synchronised across processes
            # - so loss must be per-block-in-batch
            loss = lossvector.sum() / non_zero_seqlens
            fval = float(loss)
        total_fval += fval
        total_samples += int(indata.nelement())
        total_bases += int(seqlens.sum())

        if calc_grads:
            loss.backward()

    if calc_grads:
        for p in net_info.net.parameters():
            if p.grad is not None:
                p.grad /= n_subbatches

    return total_chunk_count, total_fval / n_subbatches, \
        total_samples, total_bases, rejection_dict


def parse_init_args(args):
    is_multi_gpu = (args.local_rank is not None)
    is_lead_process = (not is_multi_gpu) or args.local_rank == 0

    if is_multi_gpu:
        # Use distributed parallel processing to run one process per GPU
        try:
            torch.distributed.init_process_group(backend='nccl')
        except Exception:
            raise Exception(
                'Unable to start multiprocessing group. The most likely ' +
                'reason is that the script is running with local_rank set ' +
                'but without the set-up for distributed operation. ' +
                'local_rank should be used only by torch.distributed.' +
                'launch. See the README.')
        device = helpers.set_torch_device(args.local_rank)
        if args.seed is not None:
            args.seed = args.seed + args.local_rank
    else:
        device = helpers.set_torch_device(args.device)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if is_lead_process:
        helpers.prepare_outdir(args.outdir, args.overwrite)
        if args.model.endswith('.py'):
            copyfile(args.model, os.path.join(args.outdir, 'model.py'))
        batchlog = helpers.BatchLog(args.outdir)
        logfile = os.path.join(args.outdir, 'model.log')
    else:
        logfile = batchlog = None
    log = helpers.Logger(logfile, args.quiet)
    log.write(helpers.formatted_env_info(device))

    return RESOURCE_INFO(is_multi_gpu, is_lead_process, device), log, batchlog


def load_data(args, log, res_info):
    log.write('* Loading data from {}\n'.format(args.input))
    log.write('* Per read file MD5 {}\n'.format(helpers.file_md5(args.input)))

    if args.input_strand_list is not None:
        read_ids = list(set(helpers.get_read_ids(args.input_strand_list)))
        log.write(('* Will train from a subset of {} strands, determined ' +
                   'by read_ids in input strand list\n').format(len(read_ids)))
    else:
        log.write('* Reads not filtered by id\n')
        read_ids = 'all'

    if args.limit is not None:
        log.write('* Limiting number of strands to {}\n'.format(args.limit))

    with mapped_signal_files.HDF5Reader(args.input) as per_read_file:
        alphabet_info = per_read_file.get_alphabet_information()
        # load list of signal_mapping.SignalMapping objects
        read_data = per_read_file.get_multiple_reads(
            read_ids, max_reads=args.limit)
    log.write('* Using alphabet definition: {}\n'.format(str(alphabet_info)))

    if len(read_data) == 0:
        log.write('* No reads remaining for training, exiting.\n')
        exit(1)
    log.write('* Loaded {} reads.\n'.format(len(read_data)))

    # prepare modified base paramter tensors
    mod_factor_t = torch.tensor(
        args.mod_factor, dtype=torch.float32).to(res_info.device)
    # mod cat inv freq weighting is currently disabled. Compute and set this
    # value to enable mod cat weighting
    mod_cat_weights = np.ones(alphabet_info.nbase, dtype=np.float32)
    mod_info = MOD_INFO(mod_factor_t, mod_cat_weights)

    return read_data, alphabet_info, mod_info


def load_network(args, alphabet_info, res_info, log):
    log.write('* Reading network from {}\n'.format(args.model))
    model_kwargs = {
        'stride': args.stride,
        'winlen': args.winlen,
        'insize': 1,
        'size': args.size,
        'alphabet_info': alphabet_info
    }

    if res_info.is_lead_process:
        # Under pytorch's DistributedDataParallel scheme, we
        # need a clone of the start network to use as a template for saving
        # checkpoints. Necessary because DistributedParallel makes the class
        # structure different.
        net_clone = helpers.load_model(args.model, **model_kwargs)
        log.write('* Network has {} parameters.\n'.format(
            sum(p.nelement()
                for p in net_clone.parameters())))
        if hasattr(net_clone, 'metadata'):
            #  Check model metadata is consistent with command-line options
            if net_clone.metadata['reverse'] != args.reverse:
                sys.stderr.write((
                    '* WARNING: Commandline specifies {} orientation ' +
                    'but model trained in opposite direction!\n').format(
                        'reverse' if args.reverse else 'forward'))
                net_clone.metadata['reverse'] = args.reverse
            if net_clone.metadata['standardize'] != \
               args.standardize:
                sys.stderr.write('* WARNING: Model and command-line ' +
                                 'standardization are inconsistent.\n')
                net_clone.metadata[
                    'standardize'] = args.standardize

        else:
            net_clone.metadata = {
                'reverse': args.reverse,
                'standardize': args.standardize,
                'version': layers.MODEL_VERSION
            }

        if not alphabet_info.is_compatible_model(net_clone):
            sys.stderr.write(
                '* ERROR: Model and mapped signal files contain ' +
                'incompatible alphabet definitions (including modified ' +
                'bases).')
            sys.exit(1)
        if layers.is_cat_mod_model(net_clone):
            log.write('* Loaded categorical modified base model.\n')
            if not alphabet_info.contains_modified_bases():
                sys.stderr.write(
                    '* ERROR: Modified bases model specified, but mapped ' +
                    'signal file does not contain modified bases.')
                sys.exit(1)
        else:
            log.write('* Loaded standard (canonical bases-only) model.\n')
            if alphabet_info.contains_modified_bases():
                sys.stderr.write(
                    '* ERROR: Standard (canonical bases only) model ' +
                    'specified, but mapped signal file does contains ' +
                    'modified bases.')
                sys.exit(1)
        log.write('* Dumping initial model\n')
        helpers.save_model(net_clone, args.outdir, 0)
    else:
        net_clone = None

    if res_info.is_multi_gpu:
        # so that processes 1,2,3.. don't try to load before process 0 has
        # saved
        torch.distributed.barrier()
        log.write('* MultiGPU process {}'.format(args.local_rank))
        log.write(': loading initial model saved by process 0\n')
        saved_startmodel_path = os.path.join(
            args.outdir, 'model_checkpoint_00000.checkpoint')
        network = helpers.load_model(saved_startmodel_path).to(res_info.device)
        network_metadata = parse_network_metadata(network)
        # Wrap network for training in the DistributedDataParallel structure
        network = torch.nn.parallel.DistributedDataParallel(
            network, device_ids=[args.local_rank],
            output_device=args.local_rank)
    else:
        log.write('* Loading model onto device\n')
        network = net_clone.to(res_info.device)
        network_metadata = parse_network_metadata(network)
        net_clone = None

    log.write('* Estimating filter parameters from training data\n')
    stride = guess_model_stride(network)
    optimiser = torch.optim.AdamW(
        network.parameters(), lr=args.lr_max, betas=args.adam,
        weight_decay=args.weight_decay, eps=args.eps)

    if args.lr_warmup is None:
        lr_warmup = args.lr_min
    else:
        lr_warmup = args.lr_warmup

    if args.lr_frac_decay is not None:
        lr_scheduler = optim.ReciprocalLR(
            optimiser, args.lr_frac_decay, args.warmup_batches, lr_warmup)
        log.write('* Learning rate schedule lr_max*k/(k+t)')
        log.write(', k={}, t=iterations.\n'.format(args.lr_frac_decay))
    else:
        lr_scheduler = optim.CosineFollowedByFlatLR(
            optimiser, args.lr_min, args.lr_cosine_iters, args.warmup_batches,
            lr_warmup)
        log.write('* Learning rate goes like cosine from lr_max to lr_min ')
        log.write('over {} iterations.\n'.format(args.lr_cosine_iters))
    log.write('* At start, train for {} '.format(args.warmup_batches))
    log.write('batches at warm-up learning rate {:3.2}\n'.format(lr_warmup))

    if args.gradient_cap_fraction is None:
        log.write('* No gradient capping\n')
        rolling_quantile = None
    else:
        rolling_quantile = maths.RollingQuantile(args.gradient_cap_fraction)
        log.write('* Gradient L2 norm cap will be upper' +
                  ' {:3.2f} quantile of the last {} norms.\n'.format(
                      args.gradient_cap_fraction, rolling_quantile.window))

    net_info = NETWORK_INFO(
        net=network, net_clone=net_clone, metadata=network_metadata,
        stride=stride)
    optim_info = OPTIM_INFO(
        optimiser=optimiser, lr_warmup=lr_warmup, lr_scheduler=lr_scheduler,
        rolling_quantile=rolling_quantile)

    return net_info, optim_info


def compute_filter_params(args, net_info, read_data, log):
    # Get parameters for filtering by sampling a subset of the reads
    # Result is a tuple median mean_dwell, mad mean_dwell
    # Choose a chunk length in the middle of the range for this, forcing
    # to be multiple of stride
    sampling_chunk_len = (args.chunk_len_min + args.chunk_len_max) // 2
    sampling_chunk_len = (
        sampling_chunk_len // net_info.stride) * net_info.stride
    filter_params = chunk_selection.sample_filter_parameters(
        read_data, args.sample_nreads_before_filtering, sampling_chunk_len,
        args.filter_mean_dwell, args.filter_max_dwell)
    log.write((
        '* Sampled {} chunks: median(mean_dwell)={:.2f}, mad(mean_dwell)=' +
        '{:.2f}\n').format(
            args.sample_nreads_before_filtering,
            filter_params.median_meandwell, filter_params.mad_meandwell))

    return filter_params


def extract_reporting_data(
        args, read_data, res_info, alphabet_info, filter_params, net_info,
        log):
    # Generate list of batches for standard loss reporting
    all_read_ids = [read.read_id for read in read_data]
    if args.reporting_strand_list is not None:
        # get reporting read ids in from strand list
        reporting_read_ids = set(helpers.get_read_ids(
            args.reporting_strand_list)).intersection(all_read_ids)
    else:
        # randomly select reporting read ids (at least one for tiny test runs)
        num_report_reads = max(
            1, int(len(read_data) * args.reporting_percent_reads / 100))
        reporting_read_ids = set(np.random.choice(
            all_read_ids, size=num_report_reads, replace=False))
    # generate reporting reads list
    report_read_data = [read for read in read_data
                        if read.read_id in reporting_read_ids]
    if not args.include_reporting_strands:
        # if holding strands out remove these reads from read_data
        read_data = [read for read in read_data
                     if read.read_id not in reporting_read_ids]
        log.write(('* Standard loss reporting from {} validation reads ' +
                   'held out of training. \n').format(len(report_read_data)))
    reporting_chunk_len = (args.chunk_len_min + args.chunk_len_max) // 2
    reporting_batch_list = list(prepare_random_batches(
        res_info.device, report_read_data, reporting_chunk_len,
        args.min_sub_batch_size, args.reporting_sub_batches, alphabet_info,
        filter_params, net_info, log, select_strands_randomly=False))
    log.write((
        '* Standard loss report: chunk length = {} & sub-batch size = {} ' +
        'for {} sub-batches. \n').format(
            reporting_chunk_len, args.min_sub_batch_size,
            args.reporting_sub_batches))

    return reporting_batch_list


def parse_train_params(args):
    train_params = TRAIN_PARAMS(
        args.niteration, args.sharpen, args.chunk_len_min, args.chunk_len_max,
        args.min_sub_batch_size, args.sub_batches, args.save_every,
        args.outdir, args.full_filter_status)
    return train_params


def train_model(
        train_params, net_info, optim_info, res_info, read_data, alphabet_info,
        filter_params, mod_info, reporting_batch_list, log, batchlog):
    # Set cap at very large value (before we have any gradient stats).
    gradient_cap = constants.LARGE_VAL
    score_smoothed = helpers.WindowedExpSmoother()
    total_bases = total_samples = total_chunks = 0
    # To count the numbers of different sorts of chunk rejection
    rejection_dict = defaultdict(int)
    time_last = time.time()
    log.write('* Training\n')
    for curr_iter in range(train_params.niteration):
        sharpen = train_params.sharpen.min + (
            train_params.sharpen.max - train_params.sharpen.min) * \
            min(1.0, curr_iter / train_params.sharpen.niter)

        # Chunk length is chosen randomly in the range given but forced to
        # be a multiple of the stride
        batch_chunk_len = (
            np.random.randint(train_params.chunk_len_min,
                              train_params.chunk_len_max + 1) //
            net_info.stride) * net_info.stride
        # We choose the size of a sub-batch so that the size of the data in
        # the sub-batch is about the same as args.min_sub_batch_size chunks of
        # length args.chunk_len_max
        sub_batch_size = int(
            train_params.min_sub_batch_size * train_params.chunk_len_max /
            batch_chunk_len + 0.5)
        main_batch_gen = prepare_random_batches(
            res_info.device, read_data, batch_chunk_len, sub_batch_size,
            train_params.sub_batches, alphabet_info, filter_params, net_info,
            log)

        # take optimiser step
        optim_info.optimiser.zero_grad()
        chunk_count, fval, chunk_samples, chunk_bases, batch_rejections = \
            calculate_loss(network, network_metadata,
                           main_batch_gen, sharpen,
                           mod_cat_weights,
                           mod_factor_t, calc_grads=True)

        if args.gradient_cap_fraction is None:
            gradnorm_uncapped = compute_grad_norm(network)
        else:
            gradnorm_uncapped = torch.nn.utils.clip_grad_norm_(
                network.parameters(), gradient_cap)
            gradient_cap = rolling_quantile.update(gradnorm_uncapped)

        optimizer.step()
        if is_lead_process:
            batchlog.record(fval, gradnorm_uncapped,
                            None if args.gradient_cap_fraction is None else gradient_cap)

        total_chunks += chunk_count
        total_samples += chunk_samples
        total_bases += chunk_bases
        score_smoothed.update(fval)
        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v
        if (curr_iter + 1) % train_params.save_every == 0 and \
           res_info.is_lead_process:
            helpers.save_model(
                net_info.net, train_params.outdir,
                (curr_iter + 1) // train_params.save_every,
                net_info.net_clone)
            log.write('C')
        else:
            log.write('.')
        if (curr_iter + 1) % DOTROWLENGTH == 0:
            log_polka(
                net_info, reporting_batch_list, train_params, mod_info,
                optim_info, time_last, score_smoothed, curr_iter,
                total_samples, total_bases, rejection_dict, res_info, log)
            time_last = time.time()
            total_bases = total_samples = 0

        # step learning rate scheduler
        optim_info.lr_scheduler.step()

    if res_info.is_lead_process:
        helpers.save_model(
            net_info.net, train_params.outdir,
            model_skeleton=net_info.net_clone)


if _DO_PROFILE:
    train_model_wrapper = train_model

    def train_model(*args):
        import cProfile
        cProfile.runctx('train_model_wrapper(*args)', globals(), locals(),
                        filename='train_flipflop.prof')


def log_polka(
        net_info, reporting_batch_list, train_params, mod_info, optim_info,
        time_last, score_smoothed, curr_iter, total_samples, total_bases,
        rejection_dict, res_info, log):
    # compute validation loss and log polka information
    _, rloss, _, _, _ = calculate_loss(
        net_info, reporting_batch_list, train_params.sharpen.max,
        mod_info.mod_cat_weights, mod_info.mod_factor)
    time_delta = time.time() - time_last
    log.write(LOG_POLKA_TMPLT.format(
        (curr_iter + 1) // DOTROWLENGTH, score_smoothed.value, rloss,
        time_delta, total_samples / 1000.0 / time_delta,
        total_bases / 1000.0 / time_delta,
        optim_info.lr_scheduler.get_lr()[0]))
    # Write summary of chunk rejection reasons
    if train_params.full_filter_status:
        for k, v in rejection_dict.items():
            log.write(" {}:{} ".format(k, v))
    else:
        n_tot = n_fail = 0
        for k, v in rejection_dict.items():
            n_tot += v
            if k != signal_mapping.Chunk.rej_str_pass:
                n_fail += v
        log.write("  {:.1%} chunks filtered".format(n_fail / n_tot))
    log.write("\n")

    if _DEBUG_MULTIGPU:
        for p in net_info.net.parameters():
            v = p.data.reshape(-1)[:5].to('cpu')
            u = p.data.reshape(-1)[-5:].to('cpu')
            break
        if res_info.local_rank is not None:
            log.write("* GPU{} params:".format(res_info.local_rank))
            log.write("{}...{}\n".format(v, u))


def main(args):
    res_info, log, batchlog = parse_init_args(args)
    read_data, alphabet_info, mod_info = load_data(args, log, res_info)
    net_info, optim_info = load_network(args, alphabet_info, res_info, log)
    filter_params = compute_filter_params(args, net_info, read_data, log)
    reporting_batch_list = extract_reporting_data(
        args, read_data, res_info, alphabet_info, filter_params, net_info, log)
    train_params = parse_train_params(args)
    train_model(
        train_params, net_info, optim_info, res_info, read_data, alphabet_info,
        filter_params, mod_info, reporting_batch_list, log, batchlog)


if __name__ == '__main__':
    main(get_train_flipflop_parser().parse_args())
