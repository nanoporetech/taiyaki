#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import time
import torch

from ont_fast5_api import fast5_interface
from taiyaki.cmdargs import FileExists, NonNegative, Positive
from taiyaki import basecall_helper, common_cmdargs
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
        chunks, chunk_starts, chunk_ends = basecall_helper.chunk_read(normed_signal, args.chunk_size, args.overlap)
        with torch.no_grad():
            out = model(torch.tensor(chunks, device=device))
            trans, _, _ = flipflop_make_trans(out)
            _, _, chunk_best_paths = flipflop_viterbi(trans)
            best_path = basecall_helper.stitch_paths(chunk_best_paths, chunk_starts, chunk_ends, stride)
            basecall = path_to_str(best_path.cpu().numpy(), alphabet=args.alphabet)
            print(">{}\n{}".format(read_id, basecall))
