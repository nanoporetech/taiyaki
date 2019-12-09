#!/usr/bin/env python3
import argparse
from collections import defaultdict
import numpy as np
import os
from shutil import copyfile
import sys
import time

import torch


from taiyaki import (chunk_selection, constants, ctc, flipflopfings, helpers,
                     layers, mapped_signal_files, maths, optim)
from taiyaki.cmdargs import AutoBool, FileExists, Maybe, NonNegative, Positive
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.constants import DOTROWLENGTH
from taiyaki.helpers import guess_model_stride


# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(description='Train flip-flop neural network',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

mdl_grp = parser.add_argument_group('Model Arguments')
mdl_grp.add_argument('--size', default=256, metavar='neurons',
                     type=Positive(int), help='Base layer size for model')
mdl_grp.add_argument('--stride', default=2, metavar='samples',
                     type=Positive(int), help='Stride for model')
mdl_grp.add_argument('--winlen', default=19, type=Positive(int),
                     help='Length of window over data')

trn_grp = parser.add_argument_group('Training Arguments')
add_common_command_args(trn_grp, """adam eps niteration weight_decay""".split())
trn_grp.add_argument('--gradient_cap_fraction', default=0.05, metavar = 'f',
                     type=Maybe(NonNegative(float)),
                     help='Cap L2 norm of gradient so that a fraction f of ' +
                     'gradients are capped. ' +
                     'Use --gradient_cap_fraction None for no capping.')
trn_grp.add_argument('--lr_cosine_iters', default=40000, metavar='n',
                    type=Positive(float),
                    help='Learning rate decreases from max to min ' +
                         'like cosine function over n batches')
trn_grp.add_argument('--lr_frac_decay', default=None, metavar='k',
                    type=Positive(int),
                    help='If specified, use fractional learning rate ' +
                          'schedule, rate=lr_max*k/(k+t)')
trn_grp.add_argument('--lr_max', default=2.0e-3, metavar='rate',
                    type=Positive(float),
                    help='Max (and starting) learning rate')
trn_grp.add_argument('--lr_min', default=1.0e-4, metavar='rate',
                    type=Positive(float), help='Min (and final) learning rate')
trn_grp.add_argument('--seed', default=None, metavar='integer',
                     type=Positive(int),
                     help='Set random number seed')
trn_grp.add_argument('--sharpen', default=1.0, metavar='factor',
                     type=Positive(float), help='Sharpening factor')
trn_grp.add_argument('--warmup_batches', type=int, default=200,
                     help = 'For the first n batches, ' +
                     'warm up at a low learning rate.')
trn_grp.add_argument('--lr_warmup', type=float, default=None,
                     help = "Learning rate used for warmup. Defaults to lr_min")

data_grp = parser.add_argument_group('Data Arguments')
add_common_command_args(data_grp, """filter_max_dwell filter_mean_dwell limit
                                     reverse sample_nreads_before_filtering""".split())
data_grp.add_argument('--chunk_len_min', default=2000, metavar='samples',
                      type=Positive(int),
                      help='Min length of each chunk in samples' +
                      ' (chunk lengths are random between min and max)')
data_grp.add_argument('--chunk_len_max', default=4000, metavar='samples',
                      type=Positive(int),
                      help='Max length of each chunk in samples ' +
                      '(chunk lengths are random between min and max)')
data_grp.add_argument('--input_strand_list', default=None, action=FileExists,
                      help='Strand summary file containing column read_id. '+
                      'Filenames in file are ignored.')
data_grp.add_argument('--min_sub_batch_size', default=96, metavar='chunks',
                      type=Positive(int),
                      help='Number of chunks to run in parallel per ' +
                      'sub-batch for chunk_len = chunk_len_max. Actual ' +
                      'length of sub-batch used is ' +
                      '(min_sub_batch_size * chunk_len_max / chunk_len).')
data_grp.add_argument('--sub_batches', default=1, metavar='sub_batches',
                      type=Positive(int),
                      help='Number of sub-batches per batch')

cmp_grp = parser.add_argument_group('Compute Arguments')
add_common_command_args(cmp_grp, set(("device",)))
#Argument local_rank is used only by when the script is run in multi-GPU
#mode using torch.distributed.launch. See the README.
cmp_grp.add_argument('--local_rank', type=int, default=None,
                     help = argparse.SUPPRESS)

out_grp = parser.add_argument_group('Output Arguments')
add_common_command_args(out_grp, """outdir overwrite quiet
                                    save_every""".split())
out_grp.add_argument('--full_filter_status', default=False, action=AutoBool,
                     help='Output full chunk filtering statistics. ' +
                     'Default: only proportion of filtered chunks.')
out_grp.add_argument('--reporting_sub_batches', default=10,
                     metavar='sub_batches', type=Positive(int),
                     help='Number of sub-batches to use for std loss reporting')

mod_grp = parser.add_argument_group('Modified Base Arguments')
mod_grp.add_argument('--mod_factor', type=float, default=0.1,
                     help='Relative modified base weight (compared to ' +
                     'canonical transitions) in loss/gradient (only ' +
                     'applicable for modified base models).')

misc_grp = parser.add_argument_group('Miscellaneous  Arguments')
add_common_command_args(misc_grp, set(("version",)))

parser.add_argument('model', action=FileExists,
                    help='File to read python model (or checkpoint) from')
parser.add_argument('input', action=FileExists,
                    help='file containing mapped reads')


def is_cat_mod_model(network):
    return isinstance(network.sublayers[-1], layers.GlobalNormFlipFlopCatMod)


def prepare_random_batches(device, read_data, batch_chunk_len, sub_batch_size,
                           target_sub_batches, alphabet_info, reverse,
                           filter_params, network, network_is_catmod, log):
    total_sub_batches = 0
    if reverse:
        revop = np.flip
    else:
        revop = np.array

    while total_sub_batches < target_sub_batches:

        # Chunk_batch is a list of dicts
        chunk_batch, batch_rejections = \
            chunk_selection.assemble_batch(read_data, sub_batch_size,
                                           batch_chunk_len, filter_params)
        if len(chunk_batch) < sub_batch_size:
            log.write('* Warning: only {} chunks passed filters (asked for {}).\n'.format(len(chunk_batch), sub_batch_size))

        if not all(len(d['sequence']) > 0.0 for d in chunk_batch):
            raise Exception('Error: zero length sequence')

        # Shape of input tensor must be:
        #     (timesteps) x (batch size) x (input channels)
        # in this case:
        #     batch_chunk_len x sub_batch_size x 1
        stacked_current = np.vstack([revop(d['current']) for d in chunk_batch]).T
        indata = torch.tensor( stacked_current, device=device,
                                            dtype=torch.float32 ).unsqueeze(2)

        # Prepare seqs, seqlens and (if necessary) mod_cats
        seqs, seqlens = [], []
        mod_cats = [] if network_is_catmod else None
        for chunk in chunk_batch:
            chunk_labels = revop(chunk['sequence'])
            seqlens.append(len(chunk_labels))
            if network_is_catmod:
                chunk_mod_cats = np.ascontiguousarray(
                    network.sublayers[-1].mod_labels[chunk_labels])
                mod_cats.append(chunk_mod_cats)
                # convert chunk_labels to canonical base labels
                chunk_labels = np.ascontiguousarray(
                    network.sublayers[-1].can_labels[chunk_labels])
            chunk_seq = flipflopfings.flipflop_code(
                chunk_labels, alphabet_info.ncan_base)
            seqs.append(chunk_seq)

        seqs = torch.tensor(
            np.concatenate(seqs), dtype=torch.float32, device=device)
        seqlens = torch.tensor(seqlens, dtype=torch.long, device=device)
        if network_is_catmod:
            mod_cats = torch.tensor(
                np.concatenate(mod_cats), dtype=torch.long, device=device)

        total_sub_batches += 1

        yield indata, seqs, seqlens, mod_cats, sub_batch_size, batch_rejections


def calculate_loss( network, network_is_catmod, batch_gen, sharpen,
                    can_mods_offsets = None, mod_cat_weights = None,
                    mod_factor_t = None, calc_grads = False ):

    total_chunk_count = 0
    total_fval = 0
    total_samples = 0
    total_bases = 0

    rejection_dict = defaultdict(int)

    n_subbatches = 0
    for (indata, seqs, seqlens, mod_cats, sub_batch_size,
         batch_rejections) in batch_gen:
        n_subbatches += 1
        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        total_chunk_count += sub_batch_size

        with torch.set_grad_enabled(calc_grads):
            outputs = network(indata)
            if network_is_catmod:
                lossvector = ctc.cat_mod_flipflop_loss(
                    outputs, seqs, seqlens, mod_cats, can_mods_offsets,
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
        for p in network.parameters():
            if p.grad is not None:
                p.grad /= n_subbatches

    return total_chunk_count, total_fval / n_subbatches, \
        total_samples, total_bases, rejection_dict


def main():
    args = parser.parse_args()
    is_multi_gpu = (args.local_rank is not None)
    is_lead_process = (not is_multi_gpu) or args.local_rank == 0

    if is_multi_gpu:
        #Use distributed parallel processing to run one process per GPU
        try:
            torch.distributed.init_process_group(backend='nccl')
        except:
            raise Exception("Unable to start multiprocessing group. " +
              "The most likely reason is that the script is running with " +
              "local_rank set but without the set-up for distributed " +
              "operation. local_rank should be used " +
              "only by torch.distributed.launch. See the README.")
        device = helpers.set_torch_device(args.local_rank)
        if args.seed is not None:
            #Make sure processes get different random picks of training data
            np.random.seed(args.seed + args.local_rank)
    else:
        device = helpers.set_torch_device(args.device)
        np.random.seed(args.seed)

    if is_lead_process:
        helpers.prepare_outdir(args.outdir, args.overwrite)
        if args.model.endswith('.py'):
            copyfile(args.model, os.path.join(args.outdir, 'model.py'))
        batchlog = helpers.BatchLog(args.outdir)
        logfile = os.path.join(args.outdir, 'model.log')
    else:
        logfile = None

    log = helpers.Logger(logfile, args.quiet)
    log.write(helpers.formatted_env_info(device))

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
        read_data = per_read_file.get_multiple_reads(
            read_ids, max_reads=args.limit)
        # read_data now contains a list of reads
        # (each an instance of the Read class defined in
        # mapped_signal_files.py, based on dict)
    log.write('* Using alphabet definition: {}\n'.format(str(alphabet_info)))

    if len(read_data) == 0:
        log.write('* No reads remaining for training, exiting.\n')
        exit(1)
    log.write('* Loaded {} reads.\n'.format(len(read_data)))

    log.write('* Reading network from {}\n'.format(args.model))
    model_kwargs = {
        'stride': args.stride,
        'winlen': args.winlen,
        # Number of input features to model e.g. was >1 for event-based
        # models (level, std, dwell)
        'insize': 1,
        'size' : args.size,
        'alphabet_info': alphabet_info
    }

    if is_lead_process:
        # Under pytorch's DistributedDataParallel scheme, we
        # need a clone of the start network to use as a template for saving
        # checkpoints. Necessary because DistributedParallel makes the class
        # structure different.
        network_save_skeleton = helpers.load_model(args.model, **model_kwargs)
        log.write('* Network has {} parameters.\n'.format(
                  sum([p.nelement() for p in network_save_skeleton.parameters()])))
        if not alphabet_info.is_compatible_model(network_save_skeleton):
            sys.stderr.write(
                '* ERROR: Model and mapped signal files contain incompatible ' +
                'alphabet definitions (including modified bases).')
            sys.exit(1)
        if is_cat_mod_model(network_save_skeleton):
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
        helpers.save_model(network_save_skeleton, args.outdir, 0)

    if is_multi_gpu:
        #so that processes 1,2,3.. don't try to load before process 0 has saved
        torch.distributed.barrier()
        log.write('* MultiGPU process {}'.format(args.local_rank))
        log.write(': loading initial model saved by process 0\n')
        saved_startmodel_path = os.path.join(args.outdir,
                                     'model_checkpoint_00000.checkpoint')
        network = helpers.load_model(saved_startmodel_path).to(device)
        network_is_catmod = is_cat_mod_model(network)
        # Wrap network for training in the DistributedDataParallel structure
        network = torch.nn.parallel.DistributedDataParallel(network,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank)
    else:
        network = network_save_skeleton.to(device)
        network_is_catmod = is_cat_mod_model(network)
        network_save_skeleton = None

    stride = guess_model_stride(network)
    # Get parameters for filtering by sampling a subset of the reads
    # Result is a tuple median mean_dwell, mad mean_dwell
    # Choose a chunk length in the middle of the range for this, forcing
    # to be multiple of stride
    sampling_chunk_len = (args.chunk_len_min + args.chunk_len_max) // 2
    sampling_chunk_len = (sampling_chunk_len // stride) * stride
    filter_params = chunk_selection.sample_filter_parameters(
        read_data, args.sample_nreads_before_filtering, sampling_chunk_len,
        args.filter_mean_dwell, args.filter_max_dwell)

    log.write("* Sampled {} chunks".format(args.sample_nreads_before_filtering))
    log.write(": median(mean_dwell)={:.2f}".format(
        filter_params.median_meandwell))
    log.write(", mad(mean_dwell)={:.2f}\n".format(
        filter_params.mad_meandwell))

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr_max,
                                 betas=args.adam,
                                 weight_decay=args.weight_decay,
                                 eps=args.eps)

    if args.lr_warmup is None:
        lr_warmup = args.lr_min
    else:
        lr_warmup = args.lr_warmup

    if args.lr_frac_decay is not None:
        lr_scheduler = optim.ReciprocalLR(optimizer, args.lr_frac_decay,
                                          args.warmup_batches, lr_warmup)
        log.write('* Learning rate schedule lr_max*k/(k+t)')
        log.write(', k={}, t=iterations.\n'.format(args.lr_frac_decay))
    else:
        lr_scheduler = optim.CosineFollowedByFlatLR(optimizer, args.lr_min,
                                                    args.lr_cosine_iters,
                                                    args.warmup_batches,
                                                    lr_warmup)
        log.write('* Learning rate goes like cosine from lr_max to lr_min ')
        log.write('over {} iterations.\n'.format(args.lr_cosine_iters))
    log.write('* At start, train for {} '.format(args.warmup_batches))
    log.write('batches at warm-up learning rate {:3.2}\n'.format(lr_warmup))

    score_smoothed = helpers.WindowedExpSmoother()

    # prepare modified base paramter tensors
    mod_factor_t = torch.tensor(args.mod_factor, dtype=torch.float32).to(device)
    can_mods_offsets = (network.sublayers[-1].can_mods_offsets
                        if network_is_catmod else None)
    # mod cat inv freq weighting is currently disabled. Compute and set this
    # value to enable mod cat weighting
    mod_cat_weights = np.ones(alphabet_info.nbase, dtype=np.float32)

    #Generating list of batches for standard loss reporting
    reporting_chunk_len = (args.chunk_len_min + args.chunk_len_max) // 2
    reporting_batch_list=list(
        prepare_random_batches(device, read_data, reporting_chunk_len,
                               args.min_sub_batch_size,
                               args.reporting_sub_batches, alphabet_info,
                               args.reverse, filter_params, network,
                               network_is_catmod, log))

    log.write( ('* Standard loss report: chunk length = {} & sub-batch size ' +
                '= {} for {} sub-batches. \n').format(reporting_chunk_len,
                args.min_sub_batch_size, args.reporting_sub_batches) )

    #Set cap at very large value (before we have any gradient stats).
    gradient_cap = constants.LARGE_VAL
    if args.gradient_cap_fraction is None:
        log.write('* No gradient capping\n')
    else:
        rolling_quantile = maths.RollingQuantile(args.gradient_cap_fraction)
        log.write('* Gradient L2 norm cap will be upper' +
                  ' {:3.2f} quantile of the last {} norms.\n'.format(
                          args.gradient_cap_fraction, rolling_quantile.window))


    total_bases = 0
    total_samples = 0
    total_chunks = 0
    # To count the numbers of different sorts of chunk rejection
    rejection_dict = defaultdict(int)

    t0 = time.time()
    log.write('* Training\n')


    for i in range(args.niteration):

        # Chunk length is chosen randomly in the range given but forced to
        # be a multiple of the stride
        batch_chunk_len = (np.random.randint(
            args.chunk_len_min, args.chunk_len_max + 1) // stride) * stride

        # We choose the size of a sub-batch so that the size of the data in
        # the sub-batch is about the same as args.min_sub_batch_size chunks of
        # length args.chunk_len_max
        sub_batch_size = int( args.min_sub_batch_size * args.chunk_len_max /
                              batch_chunk_len + 0.5)

        optimizer.zero_grad()

        main_batch_gen = prepare_random_batches(device, read_data,
                                                batch_chunk_len, sub_batch_size,
                                                args.sub_batches, alphabet_info,
                                                args.reverse, filter_params,
                                                network, network_is_catmod, log)

        chunk_count, fval, chunk_samples, chunk_bases, batch_rejections = \
                            calculate_loss( network, network_is_catmod,
                                            main_batch_gen, args.sharpen,
                                            can_mods_offsets, mod_cat_weights,
                                            mod_factor_t, calc_grads = True )

        gradnorm_uncapped = torch.nn.utils.clip_grad_norm_(
                                      network.parameters(), gradient_cap)
        if args.gradient_cap_fraction is not None:
            gradient_cap = rolling_quantile.update(gradnorm_uncapped)

        optimizer.step()
        if is_lead_process:
            batchlog.record(fval, gradnorm_uncapped,
                  None if args.gradient_cap_fraction is None else gradient_cap)

        total_chunks += chunk_count
        total_samples += chunk_samples
        total_bases += chunk_bases

        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        score_smoothed.update(fval)

        if (i + 1) % args.save_every == 0 and is_lead_process:
            helpers.save_model(network, args.outdir,
                               (i + 1) // args.save_every,
                               network_save_skeleton)
            log.write('C')
        else:
            log.write('.')


        if (i + 1) % DOTROWLENGTH == 0:

            _, rloss, _, _, _ = calculate_loss( network, network_is_catmod,
                                                reporting_batch_list,
                                                args.sharpen, can_mods_offsets,
                                                mod_cat_weights, mod_factor_t )

            # In case of super batching, additional functionality must be
            # added here
            learning_rate = lr_scheduler.get_lr()[0]
            tn = time.time()
            dt = tn - t0
            t = (' {:5d} {:7.5f} {:7.5f}  {:5.2f}s ({:.2f} ksample/s {:.2f} ' +
                 'kbase/s) lr={:.2e}')
            log.write(t.format((i + 1) // DOTROWLENGTH,
                               score_smoothed.value, rloss, dt,
                               total_samples / 1000.0 / dt,
                               total_bases / 1000.0 / dt, learning_rate))
            # Write summary of chunk rejection reasons
            if args.full_filter_status:
                for k, v in rejection_dict.items():
                    log.write(" {}:{} ".format(k, v))
            else:
                n_tot = n_fail = 0
                for k, v in rejection_dict.items():
                    n_tot += v
                    if k != 'pass':
                        n_fail += v
                log.write("  {:.1%} chunks filtered".format(n_fail / n_tot))
            log.write("\n")
            total_bases = 0
            total_samples = 0
            t0 = tn

            # Uncomment the lines below to check synchronisation of models
            # between processes in multi-GPU operation
            #for p in network.parameters():
            #    v = p.data.reshape(-1)[:5].to('cpu')
            #    u = p.data.reshape(-1)[-5:].to('cpu')
            #    break
            #if args.local_rank is not None:
            #    log.write("* GPU{} params:".format(args.local_rank))
            #log.write("{}...{}\n".format(v,u))

        lr_scheduler.step()


    if is_lead_process:
        helpers.save_model(network, args.outdir,
                           model_skeleton=network_save_skeleton)


if __name__ == '__main__':
    main()
