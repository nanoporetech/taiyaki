#!/usr/bin/env python3
import argparse
from multiprocessing import Pool
import sys
import time
import torch

from ont_fast5_api import fast5_interface

from taiyaki import basecall_helpers, decodeutil, fast5utils, helpers, qscores
from taiyaki.cmdargs import (AutoBool, FileExists, NonNegative,
                             ParseToNamedTuple, Positive)
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.decode import flipflop_make_trans, flipflop_viterbi
from taiyaki.flipflopfings import nstate_flipflop, path_to_str
from taiyaki.helpers import (guess_model_stride, load_model,
                             open_file_or_stdout, Progress)
from taiyaki.maths import med_mad
from taiyaki.prepare_mapping_funcs import get_per_read_params_dict_from_tsv
from taiyaki.signal import Signal


def get_parser():
    parser = argparse.ArgumentParser(
        description="Basecall reads using a taiyaki model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_common_command_args(
        parser, """alphabet device input_folder
        input_strand_list jobs limit output quiet
        recursive version""".split())

    parser.add_argument(
        '--beam', default=None, metavar=('width', 'guided'), nargs=2,
        type=(int, bool), action=ParseToNamedTuple,
        help='Use beam search decoding')
    parser.add_argument(
        "--chunk_size", type=Positive(int), metavar="blocks",
        default=basecall_helpers._DEFAULT_CHUNK_SIZE,
        help="Size of signal chunks sent to GPU is chunk_size * model stride")
    parser.add_argument(
        '--fastq', default=False, action=AutoBool,
        help='Write output in fastq format (default is fasta)')
    parser.add_argument(
        "--max_concurrent_chunks", type=Positive(int), default=128,
        help="Maximum number of chunks to call at "
        "once. Lower values will consume less (GPU) RAM.")
    parser.add_argument(
        "--overlap", type=NonNegative(int), metavar="blocks",
        default=basecall_helpers._DEFAULT_OVERLAP,
        help="Overlap between signal chunks sent to GPU")
    parser.add_argument(
        '--posterior', default=True, action=AutoBool,
        help='Use posterior-viterbi decoding')
    parser.add_argument(
        "--qscore_offset", type=float, default=0.0,
        help="Offset to apply to q scores in fastq (after scale)")
    parser.add_argument(
        "--qscore_scale", type=float, default=1.0,
        help="Scaling factor to apply to q scores in fastq")
    parser.add_argument(
        '--reverse', default=False, action=AutoBool,
        help='Reverse sequences in output')
    parser.add_argument(
        '--scaling', action=FileExists, default=None,
        help='Path to TSV containing per-read scaling params')
    parser.add_argument(
        '--temperature', default=1.0, type=float,
        help='Scaling factor applied to network outputs before decoding')
    parser.add_argument(
        "model", action=FileExists,
        help="Model checkpoint file to use for basecalling")

    return parser


def med_mad_norm(x, dtype='f4'):
    """ Normalise a numpy array using median and MAD

    Args:
        x (:class:`ndarray`): 1D array containing values to be normalised.
        dtype (str or :class:`dtype`): dtype of returned array.

    Returns:
        :class:`ndarray`:  Array of same shape as `x` and dtype `dtype`
            contained normalised values.
    """
    med, mad = med_mad(x)
    normed_x = (x - med) / mad
    return normed_x.astype(dtype)


def get_signal(read_filename, read_id):
    """ Get raw signal from read tuple

    Args:
        read_filename (str): Name of file from which to read signal.
        read_id (str): ID of signal to read from `read_filename`

    Returns:
        class:`ndarray`: 1D array containing signal.

        If unable to read signal from file, `None` is returned.
    """
    try:
        with fast5_interface.get_fast5_file(read_filename, 'r') as f5file:
            read = f5file.get_read(read_id)
            sig = Signal(read)
            return sig.current

    except Exception as e:
        sys.stderr.write(
            'Unable to obtain signal for {} from {}.\n{}\n'.format(
                read_id, read_filename, repr(e)))
        return None


def worker_init(device, modelname, chunk_size, overlap,
                read_params, alphabet, max_concurrent_chunks,
                fastq, qscore_scale, qscore_offset, beam, posterior,
                temperature):
    global all_read_params
    global process_read_partial

    all_read_params = read_params
    device = helpers.set_torch_device(device)
    model = load_model(modelname).to(device)
    stride = guess_model_stride(model)
    chunk_size = chunk_size * stride
    overlap = overlap * stride

    n_can_base = len(alphabet)
    n_can_state = nstate_flipflop(n_can_base)

    def process_read_partial(read_filename, read_id, read_params):
        res = process_read(read_filename, read_id,
                           model, chunk_size, overlap, read_params,
                           n_can_state, stride, alphabet,
                           max_concurrent_chunks, fastq, qscore_scale,
                           qscore_offset, beam, posterior, temperature)
        return (read_id, *res)


def worker(args):
    read_filename, read_id = args
    read_params = all_read_params[
        read_id] if read_id in all_read_params else None
    return process_read_partial(read_filename, read_id, read_params)


def process_read(
        read_filename, read_id, model, chunk_size, overlap, read_params,
        n_can_state, stride, alphabet, max_concurrent_chunks,
        fastq=False, qscore_scale=1.0, qscore_offset=0.0, beam=None,
        posterior=True, temperature=1.0):
    """Basecall a read, dividing the samples into chunks before applying the
    basecalling network and then stitching them back together.

    Args:
        read_filename (str): filename to load data from.
        read_id (str): id used in comment line in fasta or fastq output.
        model (:class:`nn.Module`): Taiyaki network.
        chunk_size (int): chunk size, measured in samples.
        overlap (int): overlap between chunks, measured in samples.
        read_params (dict str -> T): reads specific scaling parameters,
            including 'shift' and 'scale'.
        n_can_state (int): number of canonical flip-flop transitions (40 for
            ACGT).
        stride (int): stride of basecalling network (measured in samples)
        alphabet (str): Alphabet (e.g. 'ACGT').
        max_concurrent_chunks (int): max number of chunks to basecall at same
            time (having this limit prevents running out of memory for long
            reads).
        fastq (bool): generate fastq file with q scores if this is True,
            otherwise generate fasta.
        qscore_scale (float): Scaling factor for Q score calibration.
        qscore_offset (float): Offset for Q score calibration.
        beam (None or NamedTuple): Use beam search decoding
        posterior (bool): Decode using posterior probability of transitions
        temperature (float): Multiplier for network output

    Returns:
        tuple of str and str and int: strings containing the called bases and
            their associated Phred-encoded quality scores, and the number of
            samples in the read (before chunking).

        When `fastq` is False, `None` is returned instead of a quality string.
    """
    signal = get_signal(read_filename, read_id)
    if signal is None:
        return None, None, 0
    if model.metadata['reverse']:
        signal = signal[::-1]

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
        trans = []
        for some_chunks in torch.split(chunks, max_concurrent_chunks, 1):
            trans.append(model(some_chunks)[:, :, :n_can_state])
        trans = torch.cat(trans, 1) * temperature

        if posterior:
            trans = (flipflop_make_trans(trans) + 1e-8).log()

        if beam is not None:
            trans = basecall_helpers.stitch_chunks(trans, chunk_starts,
                                                   chunk_ends, stride)
            best_path, score = decodeutil.beamsearch(trans.cpu().numpy(),
                                                     beam_width=beam.width,
                                                     guided=beam.guided)
        else:
            _, _, chunk_best_paths = flipflop_viterbi(trans)
            best_path = basecall_helpers.stitch_chunks(
                chunk_best_paths, chunk_starts, chunk_ends,
                stride).cpu().numpy()

        if fastq:
            chunk_errprobs = qscores.errprobs_from_trans(trans,
                                                         chunk_best_paths)
            errprobs = basecall_helpers.stitch_chunks(
                chunk_errprobs, chunk_starts, chunk_ends, stride)
            qstring = qscores.path_errprobs_to_qstring(errprobs, best_path,
                                                       qscore_scale,
                                                       qscore_offset)

    # This makes our basecalls agree with Guppy's, and removes the
    # problem that there is no entry transition for the first path
    # element, so we don't know what the q score is.
    basecall = path_to_str(best_path, alphabet=alphabet,
                           include_first_source=False)

    return basecall, qstring, len(signal)


def main():
    args = get_parser().parse_args()

    # TODO convert to logging

    sys.stderr.write("* Initializing reads file search.\n")
    fast5_reads = fast5utils.iterate_fast5_reads(
        args.input_folder, limit=args.limit,
        strand_list=args.input_strand_list, recursive=args.recursive)

    if args.scaling is not None:
        sys.stderr.write(
            "* Loading read scaling parameters from {}.\n".format(
                args.scaling))
        all_read_params = get_per_read_params_dict_from_tsv(args.scaling)
        input_read_ids = frozenset(rec[1] for rec in fast5_reads)
        scaling_read_ids = frozenset(all_read_params.keys())
        sys.stderr.write("* {} / {} reads have scaling information.\n".format(
            len(input_read_ids & scaling_read_ids), len(input_read_ids)))
        fast5_reads = [rec for rec in fast5_reads if rec[
            1] in scaling_read_ids]
    else:
        all_read_params = {}

    sys.stderr.write("* Calling reads.\n")
    nbase, ncalled, nread, nsample = 0, 0, 0, 0
    t0 = time.time()
    progress = Progress(quiet=args.quiet)
    startcharacter = '@' if args.fastq else '>'
    initargs = [args.device, args.model, args.chunk_size, args.overlap,
                all_read_params, args.alphabet,
                args.max_concurrent_chunks, args.fastq, args.qscore_scale,
                args.qscore_offset, args.beam, args.posterior,
                args.temperature]
    pool = Pool(args.jobs, initializer=worker_init, initargs=initargs)
    with open_file_or_stdout(args.output) as fh:
        for read_id, basecall, qstring, read_nsample in \
                pool.imap_unordered(worker, fast5_reads):
            if basecall is not None and len(basecall) > 0:
                fh.write("{}{}\n{}\n".format(
                    startcharacter, read_id,
                    basecall[::-1] if args.reverse else basecall))
                nbase += len(basecall)
                ncalled += 1
                if args.fastq:
                    fh.write("+\n{}\n".format(
                        qstring[::-1] if args.reverse else qstring))

            nread += 1
            nsample += read_nsample
            progress.step()
    total_time = time.time() - t0

    sys.stderr.write(
        "* Called {} reads in {:.2f}s\n".format(nread, int(total_time)))
    sys.stderr.write(
        "* {:7.2f} kbase / s\n".format(nbase / total_time / 1000.0))
    sys.stderr.write(
        "* {:7.2f} ksample / s\n".format(nsample / total_time / 1000.0))
    sys.stderr.write("* {} reads failed.\n".format(nread - ncalled))
    return


if __name__ == '__main__':
    main()
