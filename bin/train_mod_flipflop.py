#!/usr/bin/env python3
import argparse
from collections import defaultdict
import numpy as np
import os
import platform
from shutil import copyfile
import sys
import time

import torch
import taiyaki.common_cmdargs as common_cmdargs
from taiyaki.cmdargs import FileExists, Positive, proportion, Maybe, AutoBool
from taiyaki.flipflopfings import flipflop_code

from taiyaki import (alphabet, chunk_selection, ctc, helpers,
                     mapped_signal_files, layers, optim)
from taiyaki import __version__


# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Train a flip-flop neural network basecaller',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

common_cmdargs.add_common_command_args(parser, """
adam device eps filter_max_dwell filter_mean_dwell
limit lr_max niteration overwrite quiet save_every
sample_nreads_before_filtering version weight_decay""".split())

parser.add_argument(
    '--chunk_len_max', default=4000, metavar='samples', type=Positive(int),
    help='Max length of each chunk in samples (chunk lengths are random ' +
    'between min and max)')
parser.add_argument(
    '--chunk_len_min', default=2000, metavar='samples', type=Positive(int),
    help='Min length of each chunk in samples (chunk lengths are random ' +
    'between min and max)')
parser.add_argument(
    '--full_filter_status', default=False, action=AutoBool,
    help='Output full chunk filtering statistics. ' +
    'Default: only proportion of filtered chunks.')
parser.add_argument(
    '--input_strand_list', default=None, action=FileExists,
    help='Strand summary file containing column read_id. Filenames in file ' +
    'are ignored.')
parser.add_argument(
    '--lr_cosine_iters', default=40000, metavar='n', type=Positive(float),
    help='Learning rate decreases from max to min like cosine function ' +
    'over n batches')
parser.add_argument(
    '--lr_max', default=1.0e-3, metavar='rate', type=Positive(float),
    help='Max (and starting) learning rate')
parser.add_argument(
    '--lr_min', default=4.0e-4, metavar='rate',
    type=Positive(float), help='Min (and final) learning rate')
parser.add_argument(
    '--min_batch_size', default=50, metavar='chunks', type=Positive(int),
    help='Number of chunks to run in parallel for chunk_len = chunk_len_max.' +
    'Actual batch size used is (min_batch_size / chunk_len) * chunk_len_max')
parser.add_argument(
    '--mod_factor', type=float, default=1.0,
    help='Relative modified base weight (compared to canonical ' +
    'transitions) in loss/grad.')
parser.add_argument(
    '--num_inv_freq_reads', default=1000, type=Positive(int),
    help='Sample N reads for modified base inverse scaling')
parser.add_argument(
    '--outdir', default='training',
    help='Output directory, created when run. Default: %(default)s')
parser.add_argument(
    '--scale_mod_loss', default=False, action=AutoBool,
    help='Scale mod parameter loss/gradients by inverse of category frequency')
parser.add_argument(
    '--sharpen', default=1.0, metavar='factor',
    type=Positive(float), help='Sharpening factor')
parser.add_argument(
    '--seed', default=None, metavar='integer', type=Positive(int),
    help='Set random number seed')
parser.add_argument(
    '--size', default=256, metavar='neurons',
    type=Positive(int), help='Base layer size for model')
parser.add_argument(
    '--stride', default=2, metavar='samples', type=Maybe(Positive(int)),
    help='Stride for model')
parser.add_argument(
    '--winlen', default=19, type=Positive(int),
    help='Length of window over data')

parser.add_argument(
    'model',
    help='File to read python model description from')
parser.add_argument(
    'input', action=FileExists,
    help='HDF5 file containing chunks')


def save_model(network, output, index=None):
    if index is None:
        basename = 'model_final'
    else:
        basename = 'model_checkpoint_{:05d}'.format(index)

    model_file = os.path.join(output, basename + '.checkpoint')
    torch.save(network, model_file)
    params_file = os.path.join(output, basename + '.params')
    torch.save(network.state_dict(), params_file)

def _load_data(args, log):
    if args.input_strand_list is not None:
        read_ids = list(set(helpers.get_read_ids(args.input_strand_list)))
        log.write('* Will train from a subset of {} strands, determined ' +
                  'by read_ids in input strand list\n'.format(len(read_ids)))
    else:
        log.write('* Will train from all strands\n')
        read_ids = 'all'

    if args.limit is not None:
        log.write('* Limiting number of strands to {}\n'.format(args.limit))

    with mapped_signal_files.HDF5Reader(args.input) as per_read_file:
        (bases_alphabet, collapse_alphabet,
         mod_long_names) = per_read_file.get_alphabet_information()
        read_data = per_read_file.get_multiple_reads(
            read_ids, max_reads=args.limit)
        # read_data now contains a list of reads
        # (each an instance of the Read class defined in
        # mapped_signal_files.py, based on dict)

    if len(read_data) == 0:
        log.write('* No reads remaining for training, exiting.\n')
        exit(1)
    log.write('* Loaded {} reads.\n'.format(len(read_data)))

    alphabet_info  = alphabet.AlphabetInfo(
        bases_alphabet, collapse_alphabet, mod_long_names, do_reorder=False)
    log.write('* Using alphabet definition: {}\n'.format(str(alphabet_info)))

    return read_data, alphabet_info

def _setup_and_logs(args):
    device = helpers.set_torch_device(args.device)

    np.random.seed(args.seed)

    helpers.prepare_outdir(args.outdir, args.overwrite)

    copyfile(args.model, os.path.join(args.outdir, 'model.py'))

    log = helpers.Logger(os.path.join(args.outdir, 'model.log'), args.quiet)
    loss_log = helpers.Logger(
        os.path.join(args.outdir, 'model.all_loss.txt'), True)
    log.write('* Taiyaki version {}\n'.format(__version__))
    log.write('* Platform is {}\n'.format(platform.platform()))
    log.write('* PyTorch version {}\n'.format(torch.__version__))
    if device.type == 'cuda':
        log.write('* CUDA version {}\n'.format(torch.version.cuda))
        log.write('* CUDA device {}\n'.format(torch.cuda.get_device_name(device)))
    else:
        log.write('* Running on CPU\n')
    log.write('* Command line\n')
    log.write(' '.join(sys.argv) + '\n')
    log.write('* Loading data from {}\n'.format(args.input))
    log.write('* Per read file MD5 {}\n'.format(helpers.file_md5(args.input)))

    return log, loss_log, device


def main():
    args = parser.parse_args()
    log, loss_log, device = _setup_and_logs(args)

    read_data, alphabet_info =  _load_data(args, log)
    # Get parameters for filtering by sampling a subset of the reads
    # Result is a tuple median mean_dwell, mad mean_dwell
    # Choose a chunk length in the middle of the range for this
    filter_parameters = chunk_selection.sample_filter_parameters(
        read_data, args.sample_nreads_before_filtering,
        (args.chunk_len_min + args.chunk_len_max) // 2,
        args, log)
    log.write(("* Sampled {} chunks: median(mean_dwell)={:.2f}, " +
               "mad(mean_dwell)={:.2f}\n").format(
                   args.sample_nreads_before_filtering, *filter_parameters))

    log.write('* Reading network from {}\n'.format(args.model))
    model_kwargs = {
        'insize': 1,
        'winlen': args.winlen,
        'stride': args.stride,
        'size' : args.size,
        'alphabet_info': alphabet_info
    }
    network = helpers.load_model(args.model, **model_kwargs).to(device)
    if not isinstance(network.sublayers[-1], layers.GlobalNormFlipFlopCatMod):
        log.write(
            'ERROR: Model must end with GlobalNormCatModFlipFlop layer, ' +
            'not {}.\n'.format(str(network.sublayers[-1])))
        sys.exit(1)
    can_mods_offsets = network.sublayers[-1].can_mods_offsets
    flipflop_can_labels = network.sublayers[-1].can_labels
    flipflop_mod_labels = network.sublayers[-1].mod_labels
    flipflop_ncan_base = network.sublayers[-1].ncan_base
    log.write('* Loaded categorical modifications flip-flop model.\n')
    log.write('* Network has {} parameters.\n'.format(
        sum([p.nelement() for p in network.parameters()])))

    optimizer = torch.optim.Adam(
        network.parameters(), lr=args.lr_max, betas=args.adam,
        weight_decay=args.weight_decay, eps=args.eps)
    lr_scheduler = optim.CosineFollowedByFlatLR(
        optimizer, args.lr_min, args.lr_cosine_iters)

    if args.scale_mod_loss:
        try:
            mod_cat_weights = alphabet_info.compute_mod_inv_freq_weights(
                read_data, args.num_inv_freq_reads)
            log.write('* Modified base weights: {}\n'.format(
                str(mod_cat_weights)))
        except NotImplementedError:
            log.write('* WARNING: Some mods not found when computing inverse ' +
                      'frequency weights. Consider raising ' +
                      '[--num_inv_freq_reads].\n')
            mod_cat_weights = np.ones(alphabet_info.nbase, dtype=np.float32)
    else:
        mod_cat_weights = np.ones(alphabet_info.nbase, dtype=np.float32)

    log.write('* Dumping initial model\n')
    save_model(network, args.outdir, 0)

    total_bases = 0
    total_chunks = 0
    total_samples = 0
    # To count the numbers of different sorts of chunk rejection
    rejection_dict = defaultdict(int)
    score_smoothed = helpers.WindowedExpSmoother()
    t0 = time.time()
    log.write('* Training\n')
    for i in range(args.niteration):
        lr_scheduler.step()
        mod_factor_t = torch.tensor(args.mod_factor, dtype=torch.float32)

        # Chunk length is chosen randomly in the range given but forced to
        # be a multiple of the stride
        batch_chunk_len = (np.random.randint(
            args.chunk_len_min, args.chunk_len_max + 1) //
                           args.stride) * args.stride
        # We choose the batch size so that the size of the data in the batch
        # is about the same as args.min_batch_size chunks of length
        # args.chunk_len_max
        target_batch_size = int(args.min_batch_size * args.chunk_len_max /
                                batch_chunk_len + 0.5)
        # ...but it can't be more than the number of reads.
        batch_size = min(target_batch_size, len(read_data))

        # chunk_batch is a list of dicts.
        chunk_batch, batch_rejections = chunk_selection.assemble_batch(
            read_data, batch_size, batch_chunk_len, filter_parameters, args,
            log)
        total_chunks += len(chunk_batch)
        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        # Shape of input tensor must be:
        #     (timesteps) x (batch size) x (input channels)
        # in this case:
        #     batch_chunk_len x batch_size x 1
        stacked_current = np.vstack([d['current'] for d in chunk_batch]).T
        indata = torch.tensor(
            stacked_current, device=device, dtype=torch.float32).unsqueeze(2)
        seqs, mod_cats, seqlens = [], [], []
        for chunk in chunk_batch:
            chunk_labels = chunk['sequence']
            seqlens.append(len(chunk_labels))
            chunk_seq = flipflop_code(np.ascontiguousarray(flipflop_can_labels[chunk_labels]),
                                       flipflop_ncan_base)
            chunk_mod_cats = np.ascontiguousarray(flipflop_mod_labels[chunk_labels])
            seqs.append(chunk_seq)
            mod_cats.append(chunk_mod_cats)
        seqs, mod_cats = np.concatenate(seqs), np.concatenate(mod_cats)
        seqs = torch.tensor(seqs, dtype=torch.float32, device=device)
        seqlens = torch.tensor(seqlens, dtype=torch.long, device=device)
        mod_cats = torch.tensor(mod_cats, dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = network(indata)
        lossvector = ctc.cat_mod_flipflop_loss(
            outputs, seqs, seqlens, mod_cats, can_mods_offsets,
            mod_cat_weights, mod_factor_t, args.sharpen)
        loss = lossvector.sum() / (seqlens > 0.0).float().sum()
        loss.backward()
        optimizer.step()

        fval = float(loss)
        score_smoothed.update(fval)

        total_bases += int(seqlens.sum())
        total_samples += int(indata.nelement())

        del indata, seqs, mod_cats, seqlens, outputs, loss, lossvector
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        loss_log.write('{}\t{:.10f}\t{:.10f}\n'.format(
            i, fval, score_smoothed.value))
        if (i + 1) % args.save_every == 0:
            save_model(network, args.outdir, (i + 1) // args.save_every)
            log.write('C')
        else:
            log.write('.')

        if (i + 1) % 50 == 0:
            # In case of super batching, additional functionality must be
            # added here
            learning_rate = lr_scheduler.get_lr()[0]
            tn = time.time()
            dt = tn - t0
            log.write((' {:5d} {:5.3f}  {:5.2f}s ({:.2f} ksample/s ' +
                       '{:.2f} kbase/s) lr={:.2e}').format(
                           (i + 1) // 50, score_smoothed.value, dt,
                           total_samples / 1000 / dt, total_bases / 1000 / dt,
                           learning_rate))
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

    save_model(network, args.outdir)

    return

if __name__ == '__main__':
    main()
