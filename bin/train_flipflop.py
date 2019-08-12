#!/usr/bin/env python3
import argparse
from collections import defaultdict
import datetime
import numpy as np
import os
import platform
from shutil import copyfile
import sys
import time

import torch


from taiyaki import (chunk_selection, constants, ctc, flipflopfings, helpers,
                     mapped_signal_files, maths, optim)
from taiyaki import __version__
from taiyaki.cmdargs import AutoBool, FileExists, Maybe, NonNegative, Positive
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.constants import DOTROWLENGTH


# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(description='Train a flip-flop neural network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_common_command_args(parser, """adam device eps filter_max_dwell filter_mean_dwell
                                   limit lr_cosine_iters niteration overwrite quiet save_every
                                   sample_nreads_before_filtering version weight_decay""".split())

parser.add_argument('--chunk_len_min', default=2000, metavar='samples', type=Positive(int),
                    help='Min length of each chunk in samples (chunk lengths are random between min and max)')
parser.add_argument('--chunk_len_max', default=4000, metavar='samples', type=Positive(int),
                    help='Max length of each chunk in samples (chunk lengths are random between min and max)')
parser.add_argument('--full_filter_status', default=False, action=AutoBool,
                    help='Output full chunk filtering statistics. ' +
                    'Default: only proportion of filtered chunks.')
parser.add_argument('--gradient_cap_fraction', default=0.05, metavar = 'f', type=Maybe(NonNegative(float)),
                    help='Cap L2 norm of gradient so that a fraction f of gradients are capped.' +
                         'Use --gradient_cap_fraction None for no capping')
parser.add_argument('--input_strand_list', default=None, action=FileExists,
                    help='Strand summary file containing column read_id. Filenames in file are ignored.')
parser.add_argument('--lr_cosine_iters', default=40000, metavar='n', type=Positive(float),
                    help='Learning rate decreases from max to min like cosine function over n batches')
parser.add_argument('--lr_frac_decay', default=None, metavar='k', type=Positive(int),
                    help='If specified, use fractional learning rate schedule, rate=lr_max*k/(k+t)')
parser.add_argument('--lr_max', default=2.0e-3, metavar='rate',
                    type=Positive(float),
                    help='Max (and starting) learning rate')
parser.add_argument('--lr_min', default=1.0e-4, metavar='rate',
                    type=Positive(float), help='Min (and final) learning rate')
parser.add_argument('--min_sub_batch_size', default=96, metavar='chunks', type=Positive(int),
                    help='Number of chunks to run in parallel per sub-batch for' +
                    'chunk_len = chunk_len_max. Actual length of sub-batch' +
                    'used is (min_sub_batch_size * chunk_len_max / chunk_len).')
parser.add_argument('--reporting_sub_batches', default=10, metavar='sub_batches', type=Positive(int),
                    help='Number of sub-batches to use for standard loss reporting')
parser.add_argument('--seed', default=None, metavar='integer', type=Positive(int),
                    help='Set random number seed')
parser.add_argument('--sharpen', default=1.0, metavar='factor',
                    type=Positive(float), help='Sharpening factor')
parser.add_argument('--size', default=256, metavar='neurons',
                    type=Positive(int), help='Base layer size for model')
parser.add_argument('--stride', default=2, metavar='samples', type=Positive(int),
                    help='Stride for model')
parser.add_argument('--sub_batches', default=1, metavar='sub_batches', type=Positive(int),
                    help='Number of sub-batches per batch')
parser.add_argument('--warmup_batches', type=int, default=200,
                    help = "For the first n batches, warm up at a low learning rate.")
parser.add_argument('--lr_warmup', type=float, default=None,
                    help = "Learning rate used for warmup. Defaults to lr_min")
parser.add_argument('--winlen', default=19, type=Positive(int),
                    help='Length of window over data')

parser.add_argument('model', action=FileExists,
                    help='File to read python model description (or checkpoint) from')

parser.add_argument('output', help='Prefix for output files')
parser.add_argument('input', action=FileExists,
                    help='file containing mapped reads')


def prepare_random_batches( device, read_data, batch_chunk_len, sub_batch_size,
                            target_sub_batches, nbase, filter_parameters, args,
                            log = None ):

    total_sub_batches = 0

    while total_sub_batches < target_sub_batches:

        # Chunk_batch is a list of dicts
        chunk_batch, batch_rejections = \
            chunk_selection.assemble_batch( read_data, sub_batch_size,
                                            batch_chunk_len, filter_parameters,
                                            args, log )

        if not all(len(d['sequence']) > 0.0 for d in chunk_batch):
            raise Exception('Error: zero length sequence')

        # Shape of input tensor must be:
        #     (timesteps) x (batch size) x (input channels)
        # in this case:
        #     batch_chunk_len x sub_batch_size x 1
        stacked_current = np.vstack([d['current'] for d in chunk_batch]).T
        indata = torch.tensor( stacked_current, device=device,
                                            dtype=torch.float32 ).unsqueeze(2)

        # Sequence input tensor is just a 1D vector, and so is seqlens
        seqs = torch.tensor( np.concatenate(
        [flipflopfings.flipflop_code(d['sequence'], nbase) for d in chunk_batch]),
                                                device=device, dtype=torch.long )
        seqlens = torch.tensor( [len(d['sequence']) for d in chunk_batch],
                                                device=device, dtype=torch.long )

        total_sub_batches += 1

        yield indata, seqs, seqlens, sub_batch_size, batch_rejections


def calculate_loss( network, batch_gen, sharpen, calc_grads = False ):

    total_chunk_count = 0
    total_non_zero_seqlens = 0
    total_fval = 0
    total_samples = 0
    total_bases = 0

    rejection_dict = defaultdict(int)

    for indata, seqs, seqlens, sub_batch_size, batch_rejections in batch_gen:

        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        total_chunk_count += sub_batch_size

        with torch.set_grad_enabled(calc_grads):
            outputs = network(indata)
            lossvector = ctc.crf_flipflop_loss(outputs, seqs, seqlens, sharpen)
            loss = lossvector.sum()
            fval = float(loss)

        total_non_zero_seqlens += (seqlens > 0.0).float().sum()
        total_fval += fval

        total_samples += int(indata.nelement())
        total_bases += int(seqlens.sum())

        if calc_grads:
            loss.backward()

        # Doing this deletion leads to less CUDA memory usage.
        #del indata, seqs, seqlens, outputs, lossvector, loss
        #if device.type == 'cuda':
        #    torch.cuda.empty_cache()

    if calc_grads:
        for p in network.parameters():
            if p.grad is not None:
                p.grad /= total_non_zero_seqlens

    return total_chunk_count, total_fval/total_non_zero_seqlens, \
                                    total_samples, total_bases, rejection_dict



def main():
    args = parser.parse_args()

    np.random.seed(args.seed)

    device = helpers.set_torch_device(args.device)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    elif not args.overwrite:
        sys.stderr.write('Error: Output directory {} exists but --overwrite ' +
                         'is false\n'.format(args.output))
        exit(1)
    if not os.path.isdir(args.output):
        sys.stderr.write(
            'Error: Output location {} is not directory\n'.format(args.output))
        exit(1)

    copyfile(args.model, os.path.join(args.output, 'model.py'))
    batchlog = helpers.BatchLog(args.output)
    log = helpers.Logger(os.path.join(args.output, 'model.log'), args.quiet)
    log.write('* Taiyaki version {}\n'.format(__version__))
    log.write('* Platform is {}\n'.format(platform.platform()))
    log.write('* PyTorch version {}\n'.format(torch.__version__))
    if device.type == 'cuda':
        log.write('* CUDA version {}\n'.format(torch.version.cuda))
        log.write('* CUDA device {}\n'.format(torch.cuda.get_device_name(device)))
    else:
        log.write('* Running on CPU\n')
    log.write('* Command line:\n')
    log.write('* "' + ' '.join(sys.argv) + '"\n')
    log.write('* Started on {}\n'.format(datetime.datetime.now()))
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
        alphabet, _, _ = per_read_file.get_alphabet_information()
        read_data = per_read_file.get_multiple_reads(
            read_ids, max_reads=args.limit)
        # read_data now contains a list of reads
        # (each an instance of the Read class defined in
        # mapped_signal_files.py, based on dict)

    if len(read_data) == 0:
        log.write('* No reads remaining for training, exiting.\n')
        exit(1)
    log.write('* Loaded {} reads.\n'.format(len(read_data)))

    # Get parameters for filtering by sampling a subset of the reads
    # Result is a tuple median mean_dwell, mad mean_dwell
    # Choose a chunk length in the middle of the range for this
    sampling_chunk_len = (args.chunk_len_min + args.chunk_len_max) // 2
    filter_parameters = chunk_selection.sample_filter_parameters(
        read_data, args.sample_nreads_before_filtering, sampling_chunk_len,
        args, log )

    medmd, madmd = filter_parameters

    log.write("* Sampled {} chunks: median(mean_dwell)={:.2f}, mad(mean_dwell)={:.2f}\n".format(
              args.sample_nreads_before_filtering, medmd, madmd))
    log.write('* Reading network from {}\n'.format(args.model))
    nbase = len(alphabet)
    model_kwargs = {
        'stride': args.stride,
        'winlen': args.winlen,
        # Number of input features to model e.g. was >1 for event-based
        # models (level, std, dwell)
        'insize': 1,
        'size' : args.size,
        'outsize': flipflopfings.nstate_flipflop(nbase)
    }
    network = helpers.load_model(args.model, **model_kwargs).to(device)
    log.write('* Network has {} parameters.\n'.format(
        sum([p.nelement() for p in network.parameters()])))

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr_max,
                                 betas=args.adam, weight_decay=args.weight_decay,
                                 eps=args.eps)

    if args.lr_warmup is None:
        lr_warmup = args.lr_min
    else:
        lr_warmup = args.lr_warmup

    if args.lr_frac_decay is not None:
        lr_scheduler = optim.ReciprocalLR(optimizer, args.lr_frac_decay, args.warmup_batches, lr_warmup)
        log.write('* Learning rate schedule lr_max*k/(k+t) , k={}, t=iterations.\n'.format(args.lr_frac_decay))
    else:
        lr_scheduler = optim.CosineFollowedByFlatLR(optimizer, args.lr_min, args.lr_cosine_iters, args.warmup_batches, lr_warmup)
        log.write('* Learning rate schedule decreases like cosine from lr_max to lr_min over {} iterations.\n'.format(args.lr_cosine_iters))
    log.write('* Before schedule starts, train for {} batches at warm-up learning rate {:3.2}\n'.format(args.warmup_batches, lr_warmup))

    score_smoothed = helpers.WindowedExpSmoother()

    #Generating list of batches for standard loss reporting
    reporting_batch_list=list(
        prepare_random_batches( device, read_data, args.chunk_len_max,
                                args.min_sub_batch_size, args.reporting_sub_batches,
                                nbase, filter_parameters, args, log ) )

    log.write( ('* Standard loss reporting: chunk length = {} & sub-batch size ' +
                '= {} for {} sub-batches. \n').format( args.chunk_len_max,
                args.min_sub_batch_size, args.reporting_sub_batches) )


    gradient_cap = constants.LARGE_VAL #Cap at very large value (before we have any gradient stats).
    if args.gradient_cap_fraction is None:
        log.write('* No gradient capping\n')
    else:
        rolling_quantile = maths.RollingQuantile(args.gradient_cap_fraction)
        log.write('* Gradient L2 norm cap will be upper' +
                  ' {:3.2f} quantile of the last {} norms.\n'.format(args.gradient_cap_fraction, rolling_quantile.window))



    log.write('* Dumping initial model\n')
    helpers.save_model(network, args.output, 0)



    total_bases = 0
    total_samples = 0
    total_chunks = 0
    # To count the numbers of different sorts of chunk rejection
    rejection_dict = defaultdict(int)

    t0 = time.time()
    log.write('* Training\n')



    for i in range(args.niteration):

        lr_scheduler.step()

        # Chunk length is chosen randomly in the range given but forced to
        # be a multiple of the stride
        batch_chunk_len = (np.random.randint(
            args.chunk_len_min, args.chunk_len_max + 1) //
                           args.stride) * args.stride

        # We choose the size of a sub-batch so that the size of the data in
        # the sub-batch is about the same as args.min_sub_batch_size chunks of
        # length args.chunk_len_max
        sub_batch_size = int( args.min_sub_batch_size * args.chunk_len_max /
                              batch_chunk_len + 0.5)

        optimizer.zero_grad()

        main_batch_gen = prepare_random_batches( device, read_data,
                                                 batch_chunk_len, sub_batch_size,
                                                 args.sub_batches, nbase,
                                                 filter_parameters, args, log )

        chunk_count, fval, chunk_samples, chunk_bases, batch_rejections = \
                            calculate_loss( network, main_batch_gen,
                                            args.sharpen, calc_grads = True )

        gradnorm_uncapped = torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_cap)
        if args.gradient_cap_fraction is not None:
            gradient_cap = rolling_quantile.update(gradnorm_uncapped)

        optimizer.step()
        batchlog.record(fval, gradnorm_uncapped, None if args.gradient_cap_fraction is None else gradient_cap)

        total_chunks += chunk_count
        total_samples += chunk_samples
        total_bases += chunk_bases

        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        score_smoothed.update(fval)

        if (i + 1) % args.save_every == 0:
            helpers.save_model(network, args.output, (i + 1) // args.save_every)
            log.write('C')
        else:
            log.write('.')


        if (i + 1) % DOTROWLENGTH == 0:

            _, rloss, _, _, _ = calculate_loss( network, reporting_batch_list,
                                                args.sharpen )

            # In case of super batching, additional functionality must be
            # added here
            learning_rate = lr_scheduler.get_lr()[0]
            tn = time.time()
            dt = tn - t0
            t = (' {:5d} {:5.3f} {:5.3f}  {:5.2f}s ({:.2f} ksample/s {:.2f} kbase/s) ' +
                 'lr={:.2e}')
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

    helpers.save_model(network, args.output)


if __name__ == '__main__':
    main()
