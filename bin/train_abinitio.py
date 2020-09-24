#!/usr/bin/env python3
import argparse
from Bio import SeqIO
import h5py
import numpy as np
import os
import pickle
from shutil import copyfile
import time

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from taiyaki import alphabet, constants, ctc, flipflopfings, helpers, maths
from taiyaki.cmdargs import FileExists, Maybe, NonNegative, Positive
from taiyaki.constants import MODEL_LOG_FILENAME
from taiyaki.common_cmdargs import add_common_command_args


def get_parser():
    parser = argparse.ArgumentParser(
        description='Train a flip-flop neural network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_common_command_args(
        parser, """adam alphabet device eps limit niteration
        outdir overwrite quiet save_every version
        weight_decay""".split())

    parser.add_argument(
        '--batch_size', default=128, metavar='chunks',
        type=Positive(int), help='Number of chunks to run in parallel')
    parser.add_argument(
        '--gradient_cap_fraction', default=0.05, metavar='f',
        type=Maybe(NonNegative(float)),
        help='Cap L2 norm of gradient so that a fraction f of gradients ' +
        'are capped. Use --gradient_cap_fraction None for no capping.')
    parser.add_argument(
        '--lr_max', default=4.0e-3, metavar='rate',
        type=Positive(float), help='Initial learning rate')
    parser.add_argument(
        '--size', default=96, metavar='neurons',
        type=Positive(int), help='Base layer size for model')
    parser.add_argument(
        '--seed', default=None, metavar='integer', type=Positive(int),
        help='Set random number seed')
    parser.add_argument(
        '--stride', default=2, metavar='samples', type=Positive(int),
        help='Stride for model')
    parser.add_argument(
        '--winlen', default=19, type=Positive(int),
        help='Length of window over data')

    parser.add_argument(
        'model', action=FileExists,
        help='File to read python model description from')
    parser.add_argument(
        'chunks', action=FileExists,
        help='file containing chunks')
    parser.add_argument(
        'reference', action=FileExists,
        help='file containing fasta reference')

    return parser


def convert_seq(s, alphabet):
    """Convert str sequence to flip-flop integer codes

    Args:
        s (str) : base sequence (e.g. 'ACCCTGGA')
        alphabet (str): alphabet of bases for coding (e.g. 'ACGT')

    Returns:
        np i4 array : flip-flop coded sequence (e.g. 01513260)
    """
    buf = np.array(list(s))
    for i, b in enumerate(alphabet):
        buf[buf == b] = i
    buf = buf.astype('i4')
    assert np.all(buf < len(alphabet)
                  ), "Alphabet violates assumption in convert_seq"
    return flipflopfings.flipflop_code(buf, len(alphabet))


def save_model(network, outdir, index=None):
    """ Save a model with name indicating how far we have got with training.

    Args:
        network (pytorch Module) : model to be saved
        outdir (str) : directory to save in
        index (int or None): number of iterations or None if training finished
    """
    if index is None:
        basename = 'model_final'
    else:
        basename = 'model_checkpoint_{:05d}'.format(index)

    model_file = os.path.join(outdir, basename + '.checkpoint')
    torch.save(network, model_file)
    params_file = os.path.join(outdir, basename + '.params')
    torch.save(network.state_dict(), params_file)


if __name__ == '__main__':
    args = get_parser().parse_args()

    np.random.seed(args.seed)

    device = helpers.set_torch_device(args.device)

    helpers.prepare_outdir(args.outdir, args.overwrite)

    copyfile(args.model, os.path.join(args.outdir, 'model.py'))

    log = helpers.Logger(
        os.path.join(args.outdir, MODEL_LOG_FILENAME), args.quiet)
    log.write(helpers.formatted_env_info(device))
    log.write('* Loading data from {}\n'.format(args.chunks))
    log.write('* Per read file MD5 {}\n'.format(helpers.file_md5(args.chunks)))

    if args.limit is not None:
        log.write('* Limiting number of strands to {}\n'.format(args.limit))

    with h5py.File(args.chunks, 'r', libver='v110') as h5:
        chunks = h5['chunks'][:args.limit]
    log.write('* Loaded {} reads from {}.\n'.format(len(chunks), args.chunks))

    if os.path.splitext(args.reference)[1] == '.pkl':
        #  Read preprocessed sequences from pickle
        with open(args.reference, 'rb') as fh:
            seq_dict = pickle.load(fh)
        log.write(
            '* Loaded preprocessed references from {}.\n'.format(
                args.reference))
    else:
        #  Read sequences from .fa / .fasta file
        seq_dict = {int(seq.id): convert_seq(str(seq.seq), args.alphabet)
                    for seq in SeqIO.parse(args.reference, "fasta")}
        log.write('* Loaded references from {}.\n'.format(args.reference))
        #  Write pickle for future
        pickle_name = os.path.splitext(args.reference)[0] + '.pkl'
        with open(pickle_name, 'wb') as fh:
            pickle.dump(seq_dict, fh)
        log.write((
            '* Written pickle of processed references to {} for ' +
            'future use.\n').format(pickle_name))

    log.write('* Reading network from {}\n'.format(args.model))
    alphabet_info = alphabet.AlphabetInfo(args.alphabet, args.alphabet)

    model_kwargs = {
        'size': args.size,
        'stride': args.stride,
        'winlen': args.winlen,
        # Number of input features to model e.g. was >1 for event-based models
        # (level, std, dwell)
        'insize': 1,
        'alphabet_info': alphabet_info
    }
    model_metadata = {
        'reverse': False,
        'standardize': True
    }
    network = helpers.load_model(
        args.model, model_metadata=model_metadata, **model_kwargs).to(device)
    log.write('* Network has {} parameters.\n'.format(
        sum([p.nelement() for p in network.parameters()])))

    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr_max,
                                  betas=args.adam, eps=args.eps,
                                  weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, args.niteration)

    score_smoothed = helpers.WindowedExpSmoother()

    log.write('* Dumping initial model\n')
    save_model(network, args.outdir, 0)

    total_bases = 0
    total_samples = 0
    total_chunks = 0

    t0 = time.time()
    log.write('* Training\n')

    # Set cap at very large value (before we have any gradient stats).
    gradient_cap = constants.LARGE_VAL
    if args.gradient_cap_fraction is None:
        log.write('* No gradient capping\n')
    else:
        rolling_quantile = maths.RollingQuantile(args.gradient_cap_fraction)
        log.write('* Gradient L2 norm cap will be upper' +
                  ' {:3.2f} quantile of the last {} norms.\n'.format(
                      args.gradient_cap_fraction, rolling_quantile.window))

    for i in range(args.niteration):

        idx = np.random.choice(
            len(chunks), size=args.batch_size, replace=False)
        indata = chunks[idx].transpose(1, 0)
        indata = torch.tensor(
            indata[..., np.newaxis], device=device, dtype=torch.float32)
        seqs = [seq_dict[i] for i in idx]

        seqlens = torch.tensor([len(seq) for seq in seqs],
                               dtype=torch.long, device=device)
        seqs = torch.tensor(np.concatenate(
            seqs), device=device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = network(indata)
        lossvector = ctc.crf_flipflop_loss(outputs, seqs, seqlens, 1.0)
        loss = lossvector.sum() / (seqlens > 0.0).float().sum()
        loss.backward()

        gradnorm_uncapped = torch.nn.utils.clip_grad_norm_(
            network.parameters(), gradient_cap)
        if args.gradient_cap_fraction is not None:
            gradient_cap = rolling_quantile.update(gradnorm_uncapped)
        optimizer.step()

        fval = float(loss)
        score_smoothed.update(fval)

        total_bases += int(seqlens.sum())
        total_samples += int(indata.nelement())

        # Doing this deletion leads to less CUDA memory usage.
        del indata, seqs, seqlens, outputs, loss, lossvector
        if device.type == 'cuda':
            torch.cuda.empty_cache()

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
            t = (' {:5d} {:7.5f}  {:5.2f}s ({:.2f} ksample/s {:.2f} ' +
                 'kbase/s) lr={:.2e}\n')
            log.write(t.format((i + 1) // 50, score_smoothed.value,
                               dt, total_samples / 1000.0 / dt,
                               total_bases / 1000.0 / dt, learning_rate))
            total_bases = 0
            total_samples = 0
            t0 = tn

        lr_scheduler.step()

    save_model(network, args.outdir)
