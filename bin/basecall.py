#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import sys
import time
import torch

from ont_fast5_api import fast5_interface

from taiyaki import basecall_helpers, fast5utils, layers
from taiyaki.cmdargs import AutoBool, FileAbsent, FileExists, NonNegative, Positive
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.constants import DEFAULT_ALPHABET
from taiyaki.cupy_extensions.flipflop import flipflop_make_trans, flipflop_viterbi
from taiyaki.flipflopfings import extract_mod_weights, nstate_flipflop, path_to_str
from taiyaki.helpers import (guess_model_stride, load_model, open_file_or_stdout,
                             Progress)
from taiyaki.maths import med_mad
from taiyaki.prepare_mapping_funcs import get_per_read_params_dict_from_tsv
from taiyaki.signal import Signal


STITCH_BEFORE_VITERBI = False


parser = argparse.ArgumentParser(
    description="Basecall reads using a taiyaki model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_common_command_args(parser, 'alphabet device input_folder input_strand_list limit output quiet recursive version'.split())

parser.add_argument("--chunk_size", type=Positive(int),
                    default=basecall_helpers._DEFAULT_CHUNK_SIZE,
                    help="Size of signal chunks sent to GPU")
parser.add_argument("--modified_base_output", action=FileAbsent, default=None,
                    help="Output filename for modified base output.")
parser.add_argument("--overlap", type=NonNegative(int),
                    default=basecall_helpers._DEFAULT_OVERLAP,
                    help="Overlap between signal chunks sent to GPU")
parser.add_argument('--reverse', default=False, action=AutoBool,
                    help='Reverse sequences in output')
parser.add_argument('--scaling', action=FileExists, default=None,
                    help='Per-read scaling params')
parser.add_argument("model", action=FileExists,
                    help="Model checkpoint file to use for basecalling")


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
            return sig.current

    except Exception as e:
        sys.stderr.write('Unable to obtain signal for {} from {}.\n{}\n'.format(
            read_id, read_filename, repr(e)))
        return None

def process_read(
        read_filename, read_id, model, chunk_size, overlap, read_params,
        n_can_state, stride, alphabet, is_cat_mod, mods_fp):
    signal = get_signal(read_filename, read_id)
    if signal is None:
        return None, 0

    if read_params is None:
        normed_signal = med_mad_norm(signal)
    else:
        normed_signal = (signal - read_params['shift']) / read_params['scale']

    chunks, chunk_starts, chunk_ends = basecall_helpers.chunk_read(
        normed_signal, chunk_size, overlap)
    with torch.no_grad():
        device = next(model.parameters()).device
        out = model(torch.tensor(chunks, device=device))

        if STITCH_BEFORE_VITERBI:
            out = basecall_helpers.stitch_chunks(
                out, chunk_starts, chunk_ends, stride)
            trans, _, _ = flipflop_make_trans(
                out.unsqueeze(1)[:,:,:n_can_state])
            _, _, best_path = flipflop_viterbi(trans)
        else:
            trans, _, _ = flipflop_make_trans(out[:,:,:n_can_state])
            _, _, chunk_best_paths = flipflop_viterbi(trans)
            best_path = basecall_helpers.stitch_chunks(
                chunk_best_paths, chunk_starts, chunk_ends, stride,
                path_stitching=is_cat_mod)

        if is_cat_mod and mods_fp is not None:
            # output modified base weights for each base call
            if STITCH_BEFORE_VITERBI:
                mod_weights = out[:,n_can_state:]
            else:
                mod_weights = basecall_helpers.stitch_chunks(
                    out[:,:,n_can_state:], chunk_starts, chunk_ends, stride)
            mods_scores = extract_mod_weights(
                mod_weights.detach().cpu().numpy(),
                best_path.detach().cpu().numpy(),
                model.sublayers[-1].can_nmods)
            mods_fp.create_dataset(
                'Reads/' + read_id, data=mods_scores,
                compression="gzip")

        basecall = path_to_str(
            best_path.cpu().numpy(), alphabet=alphabet)

    return basecall, len(signal)


def main():
    args = parser.parse_args()

    assert args.device != 'cpu', "Flipflop basecalling in taiyaki requires a GPU and for cupy to be installed"
    device = torch.device(args.device)
    # TODO convert to logging
    sys.stderr.write("* Loading model.\n")
    model = load_model(args.model).to(device)
    is_cat_mod = isinstance(model.sublayers[-1], layers.GlobalNormFlipFlopCatMod)
    do_output_mods = args.modified_base_output is not None
    if do_output_mods and not is_cat_mod:
        sys.stderr.write(
            "Cannot output modified bases from canonical base only model.")
        sys.exit()
    n_can_states = nstate_flipflop(model.sublayers[-1].nbase)
    stride = guess_model_stride(model)
    chunk_size = args.chunk_size * stride
    chunk_overlap = args.overlap * stride

    sys.stderr.write("* Initializing reads file search.\n")
    fast5_reads = list(fast5utils.iterate_fast5_reads(args.input_folder,
                                                      limit=args.limit,
                                                      strand_list=args.input_strand_list,
                                                      recursive=args.recursive))
    sys.stderr.write("* Found {} reads.\n".format(len(fast5_reads)))

    if args.scaling is not None:
        sys.stderr.write("* Loading read scaling parameters from {}.\n".format(args.scaling))
        all_read_params = get_per_read_params_dict_from_tsv(args.scaling)
        input_read_ids = frozenset(rec[1] for rec in fast5_reads)
        scaling_read_ids = frozenset(all_read_params.keys())
        sys.stderr.write("* {} / {} reads have scaling information.\n".format(len(input_read_ids & scaling_read_ids), len(input_read_ids)))
        fast5_reads = [rec for rec in fast5_reads if rec[1] in scaling_read_ids]
    else:
        all_read_params = {}

    mods_fp = None
    if do_output_mods:
        mods_fp = h5py.File(args.modified_base_output)
        mods_fp.create_group('Reads')
        mod_long_names = model.sublayers[-1].ordered_mod_long_names
        sys.stderr.write("* Preparing modified base output: {}.\n".format(
            ', '.join(map(str, mod_long_names))))
        mods_fp.create_dataset(
            'mod_long_names', data=np.array(mod_long_names, dtype='S'),
            dtype=h5py.special_dtype(vlen=str))

    sys.stderr.write("* Calling reads.\n")
    nbase, ncalled, nread, nsample = 0, 0, 0, 0
    t0 = time.time()
    progress = Progress(quiet=args.quiet)
    try:
        with open_file_or_stdout(args.output) as fh:
            for read_filename, read_id in fast5_reads:
                read_params = all_read_params[read_id] if read_id in all_read_params else None
                basecall, read_nsample = process_read(
                    read_filename, read_id, model, chunk_size, chunk_overlap,
                    read_params, n_can_states, stride, args.alphabet,
                    is_cat_mod, mods_fp)
                if basecall is not None:
                    fh.write(">{}\n{}\n".format(read_id, basecall[::-1] if args.reverse else basecall))
                    nbase += len(basecall)
                    ncalled += 1
                nread += 1
                nsample += read_nsample
                progress.step()
    finally:
        if mods_fp is not None:
            mods_fp.close()
    total_time = time.time() - t0

    sys.stderr.write("* Called {} reads in {:.2f}s\n".format(nread, int(total_time)))
    sys.stderr.write("* {:7.2f} kbase / s\n".format(nbase / total_time / 1000.0))
    sys.stderr.write("* {:7.2f} ksample / s\n".format(nsample / total_time / 1000.0))
    sys.stderr.write("* {} reads failed.\n".format(nread - ncalled))
    return


if __name__ == '__main__':
    main()
