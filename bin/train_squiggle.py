#!/usr/bin/env python3
import argparse
import numpy as np
import os
import time
import torch

from collections import defaultdict

from taiyaki import (activation, chunk_selection, helpers, layers,
                     mapped_signal_files, signal_mapping)
from taiyaki.cmdargs import AutoBool, FileExists, Maybe, Positive, proportion
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.constants import DOTROWLENGTH, MODEL_LOG_FILENAME
from taiyaki.squiggle_match import squiggle_match_loss, embed_sequence


def get_parser():
    parser = argparse.ArgumentParser(
        description='Train a model to predict ionic current levels ' +
        'from sequence',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_common_command_args(
        parser, """adam device eps filter_max_dwell filter_mean_dwell limit
        niteration outdir overwrite quiet reverse save_every
        sample_nreads_before_filtering version weight_decay""".split())

    parser.add_argument(
        '--batch_size', default=100, metavar='chunks', type=Positive(int),
        help='Number of chunks to run in parallel')
    parser.add_argument(
        '--back_prob', default=1e-15, metavar='probability',
        type=proportion, help='Probability of backwards move')
    parser.add_argument(
        '--depth', metavar='layers', default=4, type=Positive(int),
        help='Number of residual convolution layers')
    parser.add_argument(
        '--drop_slip', default=5, type=Maybe(Positive(int)), metavar='length',
        help='Drop chunks with slips greater than given length (None = off)')
    parser.add_argument(
        '--filter_path_buffer', default=1.1, metavar='ratio', type=float,
        help='Drop chunks with small ratio of signal length to bases * ' +
        'model stride, which would restrict potential CTC paths.')
    parser.add_argument(
        '--filter_min_pass_fraction', default=0.5, metavar='fraction',
        type=Maybe(Positive(float)),
        help='Halt if fraction of chunks passing tests is less than this')
    parser.add_argument(
        '--full_filter_status', default=False, action=AutoBool,
        help='Output full chunk filtering statistics. ' +
        'Default: only proportion of filtered chunks.')
    parser.add_argument(
        '--input_strand_list', default=None, action=FileExists,
        help='Strand summary file containing column read_id. Filenames in ' +
        'file are ignored.')
    parser.add_argument(
        '--lr_decay', default=5000, metavar='n', type=Positive(float),
        help='Learning rate for batch i is lr_max / (1.0 + i / n)')
    parser.add_argument(
        '--lr_max', default=1.0e-4, metavar='rate', type=Positive(float),
        help='Max (and starting) learning rate')
    parser.add_argument(
        '--sd', default=0.5, metavar='value', type=Positive(float),
        help='Standard deviation to initialise with')
    parser.add_argument(
        '--seed', default=None, metavar='integer', type=Positive(int),
        help='Set random number seed')
    parser.add_argument(
        '--size', metavar='n', default=32, type=Positive(int),
        help='Size of layers in convolution network')
    parser.add_argument(
        '--target_len', metavar='n', default=300, type=Positive(int),
        help='Target length of sequence')
    parser.add_argument(
        '--winlen', metavar='n', default=9, type=Positive(int),
        help='Window for convolution network')
    parser.add_argument(
        'input', action=FileExists,
        help='HDF5 file containing mapped reads')

    return parser


def create_convolution(size, depth, winlen):
    conv_actfun = activation.tanh
    return layers.Serial(
        [layers.Convolution(3, size, winlen, stride=1, fun=conv_actfun)] +
        [layers.Residual(layers.Convolution(
            size, size, winlen, stride=1, fun=conv_actfun))
         for _ in range(depth)] +
        [layers.Convolution(size, 3, winlen, stride=1, fun=activation.linear)]
    )


def main():
    args = get_parser().parse_args()
    np.random.seed(args.seed)

    helpers.prepare_outdir(args.outdir, args.overwrite)

    device = helpers.set_torch_device(args.device)

    log = helpers.Logger(
        os.path.join(args.outdir, MODEL_LOG_FILENAME), args.quiet)
    log.write(helpers.formatted_env_info(device))

    if args.input_strand_list is not None:
        read_ids = list(set(helpers.get_read_ids(args.input_strand_list)))
        log.write(
            '* Will train from a subset of {} strands\n'.format(len(read_ids)))
    else:
        log.write('* Reads not filtered by id\n')
        read_ids = 'all'

    if args.limit is not None:
        log.write('* Limiting number of strands to {}\n'.format(args.limit))

    with mapped_signal_files.HDF5Reader(args.input) as per_read_file:
        alphabet_info = per_read_file.get_alphabet_information()
        assert alphabet_info.nbase == 4, (
            'Squiggle prediction with modified base training data is ' +
            'not currenly supported.')
        # load list of signal_mapping.SignalMapping objects
        read_data = per_read_file.get_multiple_reads(
            read_ids, max_reads=args.limit)

    if len(read_data) == 0:
        log.write('* No reads remaining for training, exiting.\n')
        exit(1)
    log.write('* Loaded {} reads.\n'.format(len(read_data)))

    stride = 1
    # Get parameters for filtering by sampling a subset of the reads
    # Result is a tuple median mean_dwell, mad mean_dwell
    filter_parameters = chunk_selection.sample_filter_parameters(
        read_data, args.sample_nreads_before_filtering, args.target_len,
        args.filter_mean_dwell, args.filter_max_dwell,
        args.filter_min_pass_fraction, stride,
        args.filter_path_buffer)

    log.write(("* Sampled {} chunks: median(mean_dwell)={:.2f}, " +
               "mad(mean_dwell)={:.2f}\n").format(
                   args.sample_nreads_before_filtering,
                   filter_parameters.median_meandwell,
                   filter_parameters.mad_meandwell))

    conv_net = create_convolution(args.size, args.depth, args.winlen)
    if not hasattr(conv_net, 'metadata'):
        conv_net.metadata = {
            'reverse': False,
            'standardize': True,
            'version': layers.MODEL_VERSION
        }
    nparam = sum([p.data.detach().numpy().size for p in conv_net.parameters()])
    log.write('* Created network.  {} parameters\n'.format(nparam))
    log.write('* Depth {} layers ({} residual layers)\n'.format(
        args.depth + 2, args.depth))
    log.write('* Window width {}\n'.format(args.winlen))
    log.write('* Context +/- {} bases\n'.format(
        (args.depth + 2) * (args.winlen // 2)))

    conv_net = conv_net.to(device)

    optimizer = torch.optim.AdamW(
        conv_net.parameters(), lr=args.lr_max, betas=args.adam,
        weight_decay=args.weight_decay, eps=args.eps)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda i: args.lr_decay / (i + args.lr_decay))

    # To count the numbers of different sorts of chunk rejection
    rejection_dict = defaultdict(lambda: 0)
    t0 = time.time()
    score_smoothed = helpers.WindowedExpSmoother()
    total_chunks = 0

    if args.reverse:
        revop = np.flip
    else:
        revop = np.array

    for i in range(args.niteration):
        # If the logging threshold is 0 then we log all chunks, including those
        # rejected, so pass the log object into assemble_batch
        # chunk_batch is a list of dicts.
        chunk_batch, batch_rejections = chunk_selection.sample_chunks(
            read_data, args.batch_size, args.target_len, filter_parameters,
            args.filter_min_pass_fraction,
            chunk_len_means_sequence_len=True)
        if len(chunk_batch) < args.batch_size:
            log.write('* Warning: only {} chunks passed filters.\n'.format(
                len(chunk_batch)))

        total_chunks += len(chunk_batch)
        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        # Shape of input needs to be seqlen x batchsize x embedding_dimension
        embedded_matrix = [embed_sequence(
            revop(chunk.sequence), alphabet=None) for chunk in chunk_batch]
        seq_embed = torch.tensor(embedded_matrix).permute(1, 0, 2).to(device)
        # Shape of labels is a flat vector
        batch_signal = torch.tensor(np.concatenate(
            [revop(chunk.current) for chunk in chunk_batch])).to(device)
        # Shape of lens is also a flat vector
        batch_siglen = torch.tensor(
            [chunk.sig_len for chunk in chunk_batch]).to(device)

        # print("First 10 elements of first sequence in batch",
        #       seq_embed[:10,0,:])
        # print("First 10 elements of signal batch",batch_signal[:10])
        # print("First 10 lengths",batch_siglen[:10])

        optimizer.zero_grad()

        predicted_squiggle = conv_net(seq_embed)
        batch_loss = squiggle_match_loss(
            predicted_squiggle, batch_signal, batch_siglen, args.back_prob)
        fval = batch_loss.sum() / float(batch_siglen.sum())

        fval.backward()
        optimizer.step()

        score_smoothed.update(float(fval))

        if (i + 1) % args.save_every == 0:
            helpers.save_model(conv_net, args.outdir,
                               (i + 1) // args.save_every)
            log.write('C')
        else:
            log.write('.')

        if (i + 1) % DOTROWLENGTH == 0:
            tn = time.time()
            dt = tn - t0
            t = ' {:5d} {:7.5f}  {:5.2f}s'
            log.write(t.format((i + 1) // DOTROWLENGTH,
                               score_smoothed.value, dt))
            t0 = tn
            # Write summary of chunk rejection reasons
            if args.full_filter_status:
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

        lr_scheduler.step()

    helpers.save_model(conv_net, args.outdir)


if __name__ == '__main__':
    main()
