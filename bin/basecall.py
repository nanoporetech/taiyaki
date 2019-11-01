#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import sys
import time
import torch

from ont_fast5_api import fast5_interface

from taiyaki import basecall_helpers, fast5utils, helpers, layers, qscores
from taiyaki.cmdargs import AutoBool, FileAbsent, FileExists, NonNegative, Positive
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.decode import flipflop_make_trans, flipflop_viterbi
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

add_common_command_args(parser, """alphabet device input_folder
                        input_strand_list limit output quiet
                        recursive version""".split())

parser.add_argument("--chunk_size", type=Positive(int), metavar="blocks",
                    default=basecall_helpers._DEFAULT_CHUNK_SIZE,
                    help="Size of signal chunks sent to GPU is chunk_size * model stride")
parser.add_argument('--fastq', default=False, action=AutoBool,
                     help='Write output in fastq format (default is fasta)')
parser.add_argument("--max_concurrent_chunks", type=Positive(int),
                    default=128, help="Maximum number of chunks to call at "
                    "once. Lower values will consume less (GPU) RAM.")
parser.add_argument("--modified_base_output", action=FileAbsent, default=None, metavar="mod_basecalls.hdf5",
                    help="Output filename for modified base output.")
parser.add_argument("--overlap", type=NonNegative(int), metavar="blocks",
                    default=basecall_helpers._DEFAULT_OVERLAP,
                    help="Overlap between signal chunks sent to GPU")
parser.add_argument("--qscore_offset", type=float,
                    default=0.0,
                    help="Offset to apply to q scores in fastq (after scale)")
parser.add_argument("--qscore_scale", type=float,
                    default=1.0,
                    help="Scaling factor to apply to q scores in fastq")
parser.add_argument('--reverse', default=False, action=AutoBool,
                    help='Reverse sequences in output')
parser.add_argument('--scaling', action=FileExists, default=None,
                    help='Path to TSV containing per-read scaling params')
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
        n_can_state, stride, alphabet, is_cat_mod, mods_fp,
        max_concurrent_chunks, fastq = False, qscore_scale=1.0,
        qscore_offset=0.0):
    """Basecall a read, dividing the samples into chunks before applying the
    basecalling network and then stitching them back together.

    :param read_filename: filename to load data from
    :param read_id: id used in comment line in fasta or fastq output
    :param model: pytorch basecalling network
    :param chunk_size: chunk size, measured in samples
    :param overlap: overlap between chunks, measured in samples
    :param read_params: dict of read params including 'shift' and 'scale'
    :param n_can_state: number of canonical flip-flop transitions (40 for ACGT)
    :param stride: stride of basecalling network (measured in samples)
    :param alphabet: python str containing alphabet (e.g. 'ACGT')
    :param is_cat_mod: bool. True for multi-level categorical mod-base model.
    :param mods_fp: h5py handle to hdf5 file prepared to accept mod base output
                    (not used unless is_cat_mod)
    :param max_concurrent_chunks: max number of chunks to basecall at same time
              (having this limit prevents running out of memory for long reads)
    :param fastq: generate fastq file with q scores if this is True. Otherwise
                  generate fasta.
    :param qscore_scale: qscore <-- qscore * qscore_scale + qscore_offset
                         before coding as fastq
    :param qscore_offset: see qscore_scale above
    :returns: tuple (basecall, qstring, len(signal))
              where basecall and qstring are python strings, except when
              fastq is False: in this case qstring is None.

    :note: fastq output implemented only for the case is_cat_mod=False
    """
    if is_cat_mod and fastq:
        raise Exception("fastq output not implemented for mod bases")

    signal = get_signal(read_filename, read_id)
    if signal is None:
        return None, 0

    if read_params is None:
        normed_signal = med_mad_norm(signal)
    else:
        normed_signal = (signal - read_params['shift']) / read_params['scale']

    chunks, chunk_starts, chunk_ends = basecall_helpers.chunk_read(
        normed_signal, chunk_size, overlap)

    qstring = None
    with torch.no_grad():
        device = next(model.parameters()).device
        chunks = torch.tensor(chunks, device=device)
        out = []
        for some_chunks in torch.split(chunks, max_concurrent_chunks, 1):
            out.append(model(some_chunks))
        out = torch.cat(out, 1)

        if STITCH_BEFORE_VITERBI:
            out = basecall_helpers.stitch_chunks(
                out, chunk_starts, chunk_ends, stride)
            trans = flipflop_make_trans(out.unsqueeze(1)[:,:,:n_can_state])
            _, _, best_path = flipflop_viterbi(trans)
        else:
            trans = flipflop_make_trans(out[:,:,:n_can_state])
            _, _, chunk_best_paths = flipflop_viterbi(trans)
            best_path = basecall_helpers.stitch_chunks(
                chunk_best_paths, chunk_starts, chunk_ends, stride,
                path_stitching=is_cat_mod)
            if fastq:
                chunk_errprobs = qscores.errprobs_from_trans(trans,
                                                         chunk_best_paths)
                errprobs = basecall_helpers.stitch_chunks(
                        chunk_errprobs, chunk_starts, chunk_ends, stride,
                        path_stitching=is_cat_mod)
                qstring = qscores.path_errprobs_to_qstring(errprobs, best_path,
                                                      qscore_scale,
                                                      qscore_offset)

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

        #Don't include first source state from the path in the basecall.
        #This makes our basecalls agree with Guppy's, and removes the
        #problem that there is no entry transition for the first path
        #element, so we don't know what the q score is.
        basecall = path_to_str(best_path.cpu().numpy(), alphabet=alphabet,
                               include_first_source=False)

    return basecall, qstring, len(signal)


def main():
    args = parser.parse_args()

    device = helpers.set_torch_device(args.device)
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
    startcharacter = '@' if args.fastq else '>'
    try:
        with open_file_or_stdout(args.output) as fh:
            for read_filename, read_id in fast5_reads:
                read_params = all_read_params[read_id] if read_id in all_read_params else None
                basecall, qstring, read_nsample = process_read(
                    read_filename, read_id, model, chunk_size, chunk_overlap,
                    read_params, n_can_states, stride, args.alphabet,
                    is_cat_mod, mods_fp, args.max_concurrent_chunks,
                    args.fastq, args.qscore_scale, args.qscore_offset)
                if basecall is not None:
                    fh.write("{}{}\n{}\n".format(startcharacter,
                             read_id,
                             basecall[::-1] if args.reverse else basecall))
                    nbase += len(basecall)
                    ncalled += 1
                    if args.fastq:
                        fh.write("+\n{}\n".format(
                                qstring[::-1] if args.reverse else qstring))
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
