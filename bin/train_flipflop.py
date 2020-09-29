#!/usr/bin/env python3
from collections import defaultdict, namedtuple
import math
import numpy as np
import os
from shutil import copyfile
import sys
import time

import torch

from taiyaki import (
    chunk_selection, ctc, flipflopfings, helpers, layers,
    mapped_signal_files, maths, signal_mapping)
from taiyaki.constants import (
    DOTROWLENGTH, MODEL_LOG_FILENAME, BATCH_LOG_FILENAME, VAL_LOG_FILENAME)
from taiyaki.helpers import get_model_device, guess_model_stride
from _bin_argparse import get_train_flipflop_parser


# Note also set flag in taiyaki/ctc/ctc.pyx when profiling code
_DO_PROFILE = False
_DEBUG_MULTIGPU = False
_MAKE_TORCH_DETERMINISTIC = False

RESOURCE_INFO = namedtuple('RESOURCE_INFO', (
    'is_multi_gpu', 'is_lead_process', 'device'))

MOD_INFO = namedtuple('MOD_INFO', ('mod_cat_weights', 'mod_factor'))

LOGS = namedtuple('LOGS', ('main', 'batch', 'validation'))
LOGS.__new__.__defaults__ = (None, None, None)

NETWORK_METADATA = namedtuple('NETWORK_METADATA', (
    'reverse', 'standardize', 'is_cat_mod', 'can_mods_offsets',
    'can_labels', 'mod_labels'))
NETWORK_METADATA.__new__.__defaults__ = (None, None, None)
NETWORK_INFO = namedtuple('NETWORK_INFO', (
    'net', 'net_clone', 'metadata', 'stride'))

OPTIM_INFO = namedtuple('OPTIM_INFO', (
    'optimiser', 'lr_warmup', 'lr_scheduler', 'rolling_mads'))

TRAIN_PARAMS = namedtuple('TRAIN_PARAMS', (
    'niteration', 'sharpen', 'chunk_len_min', 'chunk_len_max',
    'min_sub_batch_size', 'sub_batches', 'save_every',
    'outdir', 'full_filter_status'))

BATCH_FIELDS = [
    'iter', 'loss', 'gradientmax', 'gradientcap', 'learning_rate',
    'chunk_len']
BATCH_TMPLT = '\t'.join('{}' for _ in BATCH_FIELDS) + '\n'
BATCH_HEADER = BATCH_TMPLT.format(*BATCH_FIELDS)
VAL_FIELDS = ['iter', 'loss']
VAL_TMPLT = '\t'.join('{}' for _ in VAL_FIELDS) + '\n'
VAL_HEADER = VAL_TMPLT.format(*VAL_FIELDS)

MAIN_LOG_POLKA_TMPLT = (
    ' {:5d} {:7.5f}   {:5.2f}s ({:.2f} ksample/s {:.2f} ' +
    'kbase/s) lr={:.2e}')
MAIN_LOG_VAL_TMPLT = (
    'iteration: {} validation_loss: {:7.5f} ({:5.2} Mbase in {:5.2f} s, ' +
    '{:.2f} kbase/s)\n')


def parse_network_metadata(network):
    if layers.is_cat_mod_model(network):
        return NETWORK_METADATA(
            network.metadata['reverse'], network.metadata['standardize'],
            True, network.sublayers[-1].can_mods_offsets,
            network.sublayers[-1].can_labels,
            network.sublayers[-1].mod_labels)
    return NETWORK_METADATA(
        network.metadata['reverse'], network.metadata['standardize'], False)


def prepare_random_batches(
        read_data, batch_chunk_len, sub_batch_size, target_sub_batches,
        alphabet_info, filter_params, net_info, log,
        select_strands_randomly=True, first_strand_index=0, pin=True):
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
        indata = torch.tensor(stacked_current, device='cpu',
                              dtype=torch.float32).unsqueeze(2)
        if pin and torch.cuda.is_available():
            indata = indata.pin_memory()

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
            np.concatenate(seqs), dtype=torch.long, device='cpu')
        seqlens = torch.tensor(seqlens, dtype=torch.long, device='cpu')
        if net_info.metadata.is_cat_mod:
            mod_cats = torch.tensor(
                np.concatenate(mod_cats), dtype=torch.long, device='cpu')

        total_sub_batches += 1

        yield indata, seqs, seqlens, mod_cats, sub_batch_size, batch_rejections


def calculate_loss(
        net_info, batch_gen, sharpen, mod_cat_weights=None, mod_factor=None,
        calc_grads=False):
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
            outputs = net_info.net(indata.to(
                get_model_device(net_info.net), non_blocking=True))
            if net_info.metadata.is_cat_mod:
                lossvector = ctc.cat_mod_flipflop_loss(
                    outputs, seqs, seqlens, mod_cats,
                    net_info.metadata.can_mods_offsets,
                    mod_cat_weights * mod_factor, sharpen)
            else:
                lossvector = ctc.crf_flipflop_loss(
                    outputs, seqs, seqlens, sharpen)

            # In multi-GPU mode, gradients are synchronised when
            # loss.backward() is called. We need to make sure we are
            # calculating a gradient that can be synchronised across processes
            # - so loss must be per-block-in-batch
            loss = lossvector.mean()

        if calc_grads:
            loss.backward()

        fval = float(loss)
        total_fval += fval
        total_samples += int(indata.nelement())
        total_bases += int(seqlens.sum())

    if calc_grads:
        for p in net_info.net.parameters():
            if p.grad is not None:
                p.grad /= n_subbatches

    return total_chunk_count, total_fval / n_subbatches, \
        total_samples, total_bases, rejection_dict


def apply_clipping(net_info, grad_max_threshs):
    parameters = [p for p in net_info.net.parameters() if p.requires_grad]
    grad_maxs = [
        float(torch.max(torch.abs(param_group.grad.detach())))
        for param_group in parameters]
    if grad_max_threshs is not None:
        for grp_gm, grp_gmt, grp_params in zip(
                grad_maxs, grad_max_threshs, parameters):
            if grp_gm > grp_gmt:
                # clip norm by value
                grp_params.grad.data.clamp_(min=-grp_gmt, max=grp_gmt)
    return grad_maxs


def parse_init_args(args):
    is_multi_gpu = (args.local_rank is not None)
    is_lead_process = (not is_multi_gpu) or args.local_rank == 0

    # if seed is provided use this else generate random seed value
    seed = (
        np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32)
        if args.seed is None else args.seed)

    main_log_fn = os.path.join(args.outdir, MODEL_LOG_FILENAME)
    if is_lead_process:
        helpers.prepare_outdir(args.outdir, args.overwrite)
        if args.model.endswith('.py'):
            copyfile(args.model, os.path.join(args.outdir, 'model.py'))
        # note buffering=1 to enforce line buffering and enable
        # inspection/plotting during a run
        logs = LOGS(
            main=helpers.Logger(main_log_fn, args.quiet),
            batch=open(os.path.join(args.outdir, BATCH_LOG_FILENAME),
                       'w', buffering=1),
            validation=open(os.path.join(args.outdir, VAL_LOG_FILENAME),
                            'w', buffering=1))
        logs.batch.write(BATCH_HEADER)
        logs.validation.write(VAL_HEADER)

        if args.save_every % DOTROWLENGTH != 0:
            # Illegal save_every, change
            se2 = int(math.ceil(args.save_every / DOTROWLENGTH)) * DOTROWLENGTH
            logs.main.write('* --save_every {} not a multiple of {}, rounding '
                            'to {}'.format(args.save_every, DOTROWLENGTH, se2))
            args.save_every = se2

        if args.chunk_len_min > args.chunk_len_max:
            # Illegal chunk length parameters
            raise ValueError('--chunk_len_min greater than --chunk_len_max')

        logs.main.write('* Using random seed: {}\n'.format(seed))
    else:
        logs = LOGS(main=helpers.Logger(main_log_fn, args.quiet))

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
        # offset seeds so different GPUs get different data streams
        seed += args.local_rank
    else:
        device = helpers.set_torch_device(args.device)
    logs.main.write(helpers.formatted_env_info(device))

    # set random seed for this process
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _MAKE_TORCH_DETERMINISTIC and device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return RESOURCE_INFO(is_multi_gpu, is_lead_process, device), logs


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

    # mod cat inv freq weighting is currently disabled. Compute and set this
    # value to enable mod cat weighting
    # prepare modified base paramter tensors
    if args.mod_prior_factor is None:
        mod_cat_weights = np.ones(alphabet_info.nbase, dtype=np.float32)
    else:
        mod_cat_weights = alphabet_info.compute_log_odds_weights(
            read_data, args.num_mod_weight_reads)
        log.write('* Computed modbase log odds priors:  {}\n'.format(
            '  '.join('{}:{:.4f}'.format(*x)
                      for x in zip(alphabet_info.alphabet, mod_cat_weights))))
        if args.mod_prior_factor != 1.0:
            mod_cat_weights = np.power(mod_cat_weights, args.mod_prior_factor)
            log.write('* Applied mod_prior_factor to modbase log odds ' +
                      'priors:  {}\n'.format(
                          '  '.join('{}:{:.4f}'.format(*x)
                                    for x in zip(alphabet_info.alphabet,
                                                 mod_cat_weights))))
    mod_info = MOD_INFO(mod_cat_weights, args.mod_factor)

    return read_data, alphabet_info, mod_info


def load_network(args, alphabet_info, res_info, log):
    log.write('* Reading network from {}\n'.format(args.model))
    if res_info.is_lead_process:
        # Under pytorch's DistributedDataParallel scheme, we
        # need a clone of the start network to use as a template for saving
        # checkpoints. Necessary because DistributedParallel makes the class
        # structure different.
        model_kwargs = {
            'stride': args.stride,
            'winlen': args.winlen,
            'insize': 1,
            'size': args.size,
            'alphabet_info': alphabet_info
        }
        model_metadata = {'reverse': args.reverse,
                          'standardize': args.standardize}
        net_clone = helpers.load_model(
            args.model, model_metadata=model_metadata, **model_kwargs)
        log.write('* Network has {} parameters.\n'.format(
            sum(p.nelement() for p in net_clone.parameters())))

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
        if layers.is_delta_model(net_clone) and model_metadata.standardize:
            log.write(
                '*' * 60 + '\n* WARNING: Delta-scaling models trained ' +
                'with --standardize are not compatible with Guppy.\n' +
                '*' * 60)
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

    lr_warmup = args.lr_min if args.lr_warmup is None else args.lr_warmup
    adam_beta1, _ = args.adam
    if args.warmup_batches >= args.niteration:
        sys.stderr.write('* Error: --warmup_batches must be < --niteration\n')
        sys.exit(1)
    warmup_fraction = args.warmup_batches / args.niteration
    # Pytorch OneCycleLR crashes if pct_start==1 (i.e. warmup_fraction==1)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        args.lr_max,
        total_steps=args.niteration,
        # pct_start is really fractional, not percent
        pct_start=warmup_fraction,
        div_factor=args.lr_max / lr_warmup,
        final_div_factor=lr_warmup / args.lr_min,
        cycle_momentum=(args.min_momentum is not None),
        base_momentum=adam_beta1 if args.min_momentum is None \
        else args.min_momentum,
        max_momentum=adam_beta1
    )
    log.write(('* Learning rate increases from {:.2e} to {:.2e} over {} ' +
               'iterations using cosine schedule.\n').format(
                   lr_warmup, args.lr_max, args.warmup_batches))
    log.write(('* Then learning rate decreases from {:.2e} to {:.2e} over ' +
               '{} iterations using cosine schedule.\n').format(
                   args.lr_max, args.lr_min,
                   args.niteration - args.warmup_batches))

    if args.gradient_clip_num_mads is None:
        log.write('* No gradient clipping\n')
        rolling_mads = None
    else:
        nparams = len([p for p in network.parameters() if p.requires_grad])
        if nparams == 0:
            rolling_mads = None
            log.write('* No gradient clipping due to missing parameters\n')
        else:
            rolling_mads = maths.RollingMAD(
                nparams, args.gradient_clip_num_mads)
            log.write((
                '* Gradients will be clipped (by value) at {:3.2f} MADs ' +
                'above the median of the last {} gradient maximums.\n').format(
                    rolling_mads.n_mads, rolling_mads.window))

    net_info = NETWORK_INFO(
        net=network, net_clone=net_clone, metadata=network_metadata,
        stride=stride)
    optim_info = OPTIM_INFO(
        optimiser=optimiser, lr_warmup=lr_warmup, lr_scheduler=lr_scheduler,
        rolling_mads=rolling_mads)

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
        args.filter_mean_dwell, args.filter_max_dwell,
        args.filter_min_pass_fraction, net_info.stride,
        args.filter_path_buffer)
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
        report_read_data, reporting_chunk_len,
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
        filter_params, mod_info, reporting_batch_list, logs):
    # Set cap at very large value (before we have any gradient stats).
    grad_max_threshs = None
    grad_max_thresh_str = 'NaN'
    score_smoothed = helpers.WindowedExpSmoother()
    total_bases = total_samples = total_chunks = 0
    # To count the numbers of different sorts of chunk rejection
    rejection_dict = defaultdict(int)
    time_last = time.time()
    logs.main.write('* Training\n')
    for curr_iter in range(train_params.niteration):
        sharpen = float(train_params.sharpen.min + (
            train_params.sharpen.max - train_params.sharpen.min) *
            min(1.0, curr_iter / train_params.sharpen.niter))
        mod_factor = float(mod_info.mod_factor.start + (
            mod_info.mod_factor.final - mod_info.mod_factor.start) *
            min(1.0, curr_iter / mod_info.mod_factor.niter))

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
            read_data, batch_chunk_len, sub_batch_size,
            train_params.sub_batches, alphabet_info, filter_params, net_info,
            logs.main)

        # take optimiser step
        optim_info.optimiser.zero_grad()
        chunk_count, fval, chunk_samples, chunk_bases, batch_rejections = \
            calculate_loss(
                net_info, main_batch_gen, sharpen, mod_info.mod_cat_weights,
                mod_factor, calc_grads=True)
        grad_maxs = apply_clipping(net_info, grad_max_threshs)
        optim_info.optimiser.step()
        if optim_info.rolling_mads is not None:
            grad_max_threshs = optim_info.rolling_mads.update(grad_maxs)
        # record step information
        if res_info.is_lead_process:
            grad_max_thresh_str = ','.join((
                'NA' if l_gmt is None else str(float(l_gmt))
                for l_gmt in grad_maxs))
            logs.batch.write(BATCH_TMPLT.format(
                curr_iter + 1, fval, ','.join(map(str, grad_maxs)),
                grad_max_thresh_str, optim_info.lr_scheduler.get_last_lr()[0],
                batch_chunk_len))

        total_chunks += chunk_count
        total_samples += chunk_samples
        total_bases += chunk_bases
        score_smoothed.update(fval)
        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        logs.main.write('.')

        if (curr_iter + 1) % DOTROWLENGTH == 0:
            log_polka(
                net_info, train_params,
                optim_info, time_last, score_smoothed, curr_iter,
                total_samples, total_bases, rejection_dict, res_info,
                logs.main)
            time_last = time.time()
            total_bases = total_samples = 0

        if (curr_iter + 1) % train_params.save_every == 0:
            #  Save model and validate
            if res_info.is_lead_process:
                saved_filename = helpers.save_model(
                    net_info.net, train_params.outdir,
                    (curr_iter + 1) // train_params.save_every,
                    net_info.net_clone)
                logs.main.write("Model saved to {}.\n".format(saved_filename))

                log_validation(net_info, reporting_batch_list, train_params,
                               mod_info, curr_iter, logs)
            time_last = time.time()

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
        net_info, train_params, optim_info,
        time_last, score_smoothed, curr_iter, total_samples, total_bases,
        rejection_dict, res_info, log):
    # Log polka information
    time_delta = time.time() - time_last
    log.write(MAIN_LOG_POLKA_TMPLT.format(
        (curr_iter + 1) // DOTROWLENGTH, score_smoothed.value,
        time_delta, total_samples / 1000.0 / time_delta,
        total_bases / 1000.0 / time_delta,
        optim_info.lr_scheduler.get_last_lr()[0]))
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


def log_validation(
        net_info, reporting_batch_list, train_params, mod_info, curr_iter,
        logs):
    t0 = time.time()
    _, rloss, _, total_bases, _ = calculate_loss(
        net_info, reporting_batch_list, train_params.sharpen.max,
        mod_info.mod_cat_weights, mod_info.mod_factor.final)
    dt = time.time() - t0
    kbases = total_bases / 1e3
    logs.main.write(MAIN_LOG_VAL_TMPLT.format(
        curr_iter + 1, rloss, kbases / 1e3, dt, kbases / dt))
    logs.validation.write(VAL_TMPLT.format(curr_iter + 1, rloss))


def main(args):
    res_info, logs = parse_init_args(args)
    read_data, alphabet_info, mod_info = load_data(args, logs.main, res_info)
    net_info, optim_info = load_network(
        args, alphabet_info, res_info, logs.main)
    filter_params = compute_filter_params(args, net_info, read_data, logs.main)
    reporting_batch_list = extract_reporting_data(
        args, read_data, res_info, alphabet_info, filter_params, net_info,
        logs.main)
    train_params = parse_train_params(args)
    train_model(
        train_params, net_info, optim_info, res_info, read_data, alphabet_info,
        filter_params, mod_info, reporting_batch_list, logs)


if __name__ == '__main__':
    main(get_train_flipflop_parser().parse_args())
