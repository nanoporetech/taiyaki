#!/usr/bin/env python3


import argparse
import numpy as np
import os
import sys
import time
import torch

from collections import defaultdict
from taiyaki import chunk_selection, helpers, mapped_signal_files, optim, variables
import taiyaki.common_cmdargs as common_cmdargs
from taiyaki.cmdargs import (FileExists, Maybe, NonNegative, Positive, proportion)
from taiyaki import activation, layers
#from taiyaki.optim import Adamski
from taiyaki.squiggle_match import squiggle_match_loss, embed_sequence
from taiyaki import __version__


parser = argparse.ArgumentParser(
    description='Train a model to predict ionic current levels from sequence',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

common_cmdargs.add_common_command_args(parser, """adam chunk_logging_threshold device filter_max_dwell filter_mean_dwell
                                                  limit niteration overwrite quiet save_every
                                                  sample_nreads_before_filtering version weight_decay""".split())

parser.add_argument('--batch_size', default=100, metavar='chunks', type=Positive(int),
                    help='Number of chunks to run in parallel')
parser.add_argument('--back_prob', default=1e-15, metavar='probability',
                    type=proportion, help='Probability of backwards move')
parser.add_argument('--depth', metavar='layers' , default=4, type=Positive(int),
                    help='Number of residual convolution layers')
parser.add_argument('--drop_slip', default=5, type=Maybe(Positive(int)), metavar='length',
                    help='Drop chunks with slips greater than given length (None = off)')
parser.add_argument('--input_strand_list', default=None, action=FileExists,
                    help='Strand summary file containing column read_id. Filenames in file are ignored.')
parser.add_argument('--lr_decay', default=5000, metavar='n', type=Positive(float),
                     help='Learning rate for batch i is lr_max / (1.0 + i / n)')
parser.add_argument('--lr_max', default=1.0e-4, metavar='rate',
                            type=Positive(float),
                            help='Max (and starting) learning rate')
parser.add_argument('--sd', default=0.5, metavar='value', type=Positive(float),
                    help='Standard deviation to initialise with')
parser.add_argument('--seed', default=None, metavar='integer', type=Positive(int),
                    help='Set random number seed')
parser.add_argument('--size', metavar='n', default=32, type=Positive(int),
                    help='Size of layers in convolution network')
parser.add_argument('--smooth', default=0.45, metavar='factor', type=proportion,
                    help='Smoothing factor for reporting progress')
parser.add_argument('--target_len', metavar='n', default=300, type=Positive(int),
                    help='Target length of sequence')
parser.add_argument('--winlen', metavar='n', default=7, type=Positive(int),
                    help='Window for convolution network')
parser.add_argument('input', action=FileExists, help='HDF5 file containing mapped reads')
parser.add_argument('output', help='Prefix for output files')


def create_convolution(size, depth, winlen):
    conv_actfun = activation.tanh
    return layers.Serial(
        [layers.Convolution(3, size, winlen, stride=1, fun=conv_actfun)] +
        [layers.Residual(layers.Convolution(size, size, winlen, stride=1, fun=conv_actfun)) for _ in range(depth)] +
        [layers.Convolution(size, 3, winlen, stride=1, fun=activation.linear)]
    )




if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif not args.overwrite:
        sys.stderr.write('Error: Output directory {} exists but --overwrite is false\n'.format(args.output))
        exit(1)
    if not os.path.isdir(args.output):
        sys.stderr.write('Error: Output location {} is not directory\n'.format(args.output))
        exit(1)

    log = helpers.Logger(os.path.join(args.output, 'model.log'), args.quiet)
    log.write('# Taiyaki version {}\n'.format(__version__))
    log.write('# Command line\n')
    log.write(' '.join(sys.argv) + '\n')

    if args.input_strand_list is not None:
        read_ids = list(set(helpers.get_read_ids(args.input_strand_list)))
        log.write('* Will train from a subset of {} strands\n'.format(len(read_ids)))
    else:
        log.write('* Reads not filtered by id\n')
        read_ids = 'all'

    if args.limit is not None:
        log.write('* Limiting number of strands to {}\n'.format(args.limit))

    with mapped_signal_files.HDF5(args.input, "r") as per_read_file:
        read_data = per_read_file.get_multiple_reads(read_ids, max_reads=args.limit)
        # read_data now contains a list of reads
        # (each an instance of the Read class defined in mapped_signal_files.py, based on dict)

    log.write('* Loaded {} reads.\n'.format(len(read_data)))
    
    # Create a logging file to save details of chunks.
    # If args.chunk_logging_threshold is set to 0 then we log all chunks including those rejected.
    chunk_log = chunk_selection.ChunkLog(args.output)

    # Get parameters for filtering by sampling a subset of the reads
    # Result is a tuple median mean_dwell, mad mean_dwell
    filter_parameters = chunk_selection.sample_filter_parameters(read_data,
                                                                 args.sample_nreads_before_filtering,
                                                                 args.target_len,
                                                                 args,
                                                                 log,
                                                                 chunk_log=chunk_log)

    medmd, madmd = filter_parameters
    log.write("* Sampled {} chunks: median(mean_dwell)={:.2f}, mad(mean_dwell)={:.2f}\n".format(
              args.sample_nreads_before_filtering, medmd, madmd))

    conv_net = create_convolution(args.size, args.depth, args.winlen)
    nparam = sum([p.data.detach().numpy().size for p in conv_net.parameters()])
    log.write('# Created network.  {} parameters\n'.format(nparam))
    log.write('# Depth {} layers ({} residual layers)\n'.format(args.depth + 2, args.depth))
    log.write('# Window width {}\n'.format(args.winlen))
    log.write('# Context +/- {} bases\n'.format((args.depth + 2) * (args.winlen // 2)))

    device = torch.device(args.device)
    conv_net = conv_net.to(device)



    optimizer = torch.optim.Adam(conv_net.parameters(), lr=args.lr_max,
                                 betas=args.adam, weight_decay=args.weight_decay)
    
    lr_scheduler = optim.ReciprocalLR(optimizer, args.lr_decay)
    
    rejection_dict = defaultdict(lambda : 0)  # To count the numbers of different sorts of chunk rejection
    t0 = time.time()
    score_smoothed = helpers.ExponentialSmoother(args.smooth)
    total_chunks = 0
    
    for i in range(args.niteration):
        lr_scheduler.step()
        # If the logging threshold is 0 then we log all chunks, including those rejected, so pass the log
        # object into assemble_batch
        if args.chunk_logging_threshold == 0:
            log_rejected_chunks = chunk_log
        else:
            log_rejected_chunks = None
        # chunk_batch is a list of dicts.
        chunk_batch, batch_rejections = chunk_selection.assemble_batch(read_data, args.batch_size, args.target_len,
                                                                      filter_parameters, args, log,
                                                                      chunk_log=log_rejected_chunks,
                                                                      chunk_len_means_sequence_len=True)

        total_chunks += len(chunk_batch)
        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        # Shape of input needs to be seqlen x batchsize x embedding_dimension
        embedded_matrix = [embed_sequence(d['sequence'], alphabet=None) for d in chunk_batch]
        seq_embed = torch.tensor(embedded_matrix).permute(1,0,2).to(device)
        # Shape of labels is a flat vector
        batch_signal = torch.tensor(np.concatenate([d['current'] for d in chunk_batch])).to(device)
        # Shape of lens is also a flat vector
        batch_siglen = torch.tensor([len(d['current']) for d in chunk_batch]).to(device)

        #print("First 10 elements of first sequence in batch",seq_embed[:10,0,:])
        #print("First 10 elements of signal batch",batch_signal[:10])
        #print("First 10 lengths",batch_siglen[:10])

        optimizer.zero_grad()

        predicted_squiggle = conv_net(seq_embed)
        batch_loss = squiggle_match_loss(predicted_squiggle, batch_signal, batch_siglen, args.back_prob)
        fval = batch_loss.sum() / float(batch_siglen.sum())

        fval.backward()
        optimizer.step()

        score_smoothed.update(float(fval))

        # Check for poison chunk and save losses and chunk locations if we're poisoned
        # If args.chunk_logging_threshold set to zero then we log everything
        if fval / score_smoothed.value >= args.chunk_logging_threshold:
            chunk_log.write_batch(i, chunk_batch, batch_loss)

        if (i + 1) % args.save_every == 0:
            helpers.save_model(conv_net, args.output, (i + 1) // args.save_every)
            log.write('C')
        else:
            log.write('.')


        if (i + 1) % variables.DOTROWLENGTH == 0:
            tn = time.time()
            dt = tn - t0
            t = ' {:5d} {:5.3f}  {:5.2f}s'
            log.write(t.format((i + 1) // variables.DOTROWLENGTH, score_smoothed.value, dt))
            t0 = tn
            # Write summary of chunk rejection reasons
            for k, v in rejection_dict.items():
                log.write(" {}:{} ".format(k, v))
            log.write("\n")


    helpers.save_model(conv_net, args.output)
