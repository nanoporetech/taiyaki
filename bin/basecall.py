#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import time
import torch

from ont_fast5_api import fast5_interface
from taiyaki.cmdargs import FileExists, NonNegative, Positive
from taiyaki import common_cmdargs
from taiyaki.cupy_extensions.flipflop import flipflop_make_trans, flipflop_viterbi
import taiyaki.fast5utils as fast5utils
from taiyaki.flipflopfings import path_to_str
from taiyaki.helpers import load_model, guess_model_stride
from taiyaki.maths import med_mad
from taiyaki.signal import Signal
from taiyaki.variables import DEFAULT_ALPHABET


parser = argparse.ArgumentParser(
    description="Basecall reads using a taiyaki model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

common_cmdargs.add_common_command_args(parser, 'device input_folder input_strand_list limit recursive version'.split())

parser.add_argument("--alphabet", default=DEFAULT_ALPHABET.decode(), help="Alphabet used by basecaller")
parser.add_argument("--chunk_size", type=Positive(int), default=1000, help="Size of signal chunks sent to GPU")
parser.add_argument("--overlap", type=NonNegative(int), default=100, help="Overlap between signal chunks sent to GPU")
parser.add_argument("model", action=FileExists, help="Model checkpoint file to use for basecalling")


def med_mad_norm(x, dtype='f4'):
    """ Normalise a numpy array using median and MAD """
    med, mad = med_mad(x)
    normed_x = (x - med) / mad
    return normed_x.astype(dtype)


def get_signal(read_filename, read_id):
    """ Get raw signal from read tuple (as returned by fast5utils.iterate_fast5_reads) """
    try:
        with fast5_interface.get_fast5_file(read_filename, 'r') as f5file:
            read = f5file.get_read(read_id)
            sig = Signal(read)
            return sig.dacs

    except Exception as e:
        sys.stderr.write('Unable to obtain signal for {} from {}.\n{}\n'.format(
            read_id, read_filename, repr(e)))
        return None


def chunk_read(signal, chunk_size, overlap):
    """ Divide signal into overlapping chunks """
    if len(signal) < chunk_size:
        return signal[:, None, None], np.array([0]), np.array([len(signal)])

    chunk_ends = np.arange(chunk_size, len(signal), chunk_size - overlap, dtype=int)
    chunk_ends = np.concatenate([chunk_ends, [len(signal)]], 0)
    chunk_starts = chunk_ends - chunk_size
    nchunks = len(chunk_ends)

    chunks = np.empty((chunk_size, nchunks, 1), dtype='f4')
    for i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
        chunks[:, i, 0] = signal[start:end]

    # We will use chunk_starts and chunk_ends to stitch the basecalls together
    return chunks, chunk_starts, chunk_ends


def stitch_paths(best_paths, chunk_starts, chunk_ends, stride):
    """ Stitch together Viterbi paths from overlapping chunks """
    nchunks = best_paths.shape[1]

    if nchunks == 1:
        return baths_paths[:, 0]
    else:
        # first chunk
        start = chunk_starts[0] // stride
        end = (chunk_ends[0] + chunk_starts[1]) // (2 * stride)
        stitched_paths = [best_paths[start:end, 0]]

        # middle chunks
        for i in range(1, nchunks - 1):
            start = (chunk_ends[i - 1] - chunk_starts[i]) // (2 * stride)
            end = (chunk_ends[i] + chunk_starts[i + 1] - 2 * chunk_starts[i]) // (2 * stride)
            stitched_paths.append(best_paths[start:end, i])

        # last chunk
        start = (chunk_starts[-2] - chunk_ends[-1]) // (2 * stride)
        end = (chunk_ends[-1] - chunk_starts[-1]) // stride
        stitched_paths.append(best_paths[start:end, -1])

        return torch.cat(stitched_paths, 0)


if __name__ == '__main__':

    args = parser.parse_args()

    assert args.device != 'cpu', "Flipflop basecalling in taiyaki requires a GPU and for cupy to be installed"
    device = torch.device(args.device)
    model = load_model(args.model).to(device)
    stride = guess_model_stride(model, device=device)

    fast5_reads = fast5utils.iterate_fast5_reads(
        args.input_folder, limit=args.limit, strand_list=args.input_strand_list,
        recursive=args.recursive)

    for read_filename, read_id in fast5_reads:
        signal = get_signal(read_filename, read_id)
        if signal is None:
            continue
        normed_signal = med_mad_norm(signal)
        chunks, chunk_starts, chunk_ends = chunk_read(normed_signal, args.chunk_size, args.overlap)
        with torch.no_grad():
            out = model(torch.tensor(chunks, device=device))
            trans, _, _ = flipflop_make_trans(out)
            _, _, chunk_best_paths = flipflop_viterbi(trans)
            best_path = stitch_paths(chunk_best_paths, chunk_starts, chunk_ends, stride)
            basecall = path_to_str(best_path.cpu().numpy(), alphabet=args.alphabet)
            print(">{}\n{}".format(read_id, basecall))
