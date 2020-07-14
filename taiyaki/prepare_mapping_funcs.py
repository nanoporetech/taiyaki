from collections import defaultdict
import enum
import sys

import numpy as np
import torch

from ont_fast5_api import fast5_interface
from taiyaki import (
    flipflop_remap, helpers, mapped_signal_files, signal_mapping, signal)
from taiyaki.config import taiyaki_dtype
from taiyaki.fileio import readtsv


class RemapResult(enum.Enum):
    """Enumerates possible results from remapping a read"""
    SUCCESS = 'Success!'
    READ_ID_INFO_NOT_FOUND = 'No information for read id found in file.'
    NO_REF_FOUND = 'No fasta reference found.'
    NO_PARAMS = 'No per-read params provided.'
    NETWORK_ERROR = 'Failure applying basecall network to remap read.'
    REF_TOO_LONG = 'Reference exceeded maximum allowed read length.'


def oneread_remap(
        read_tuple, model, per_read_params_dict, alphabet_info,
        max_read_length, device='cpu', localpen=0.0):
    """ Worker function for remapping reads using flip-flop model on raw signal

    Args:
        read_tuple (tuple) : read, identified by a tuple
                                  (filepath, read_id, read reference)
        model (pytorch Module): pytorch model
        device (int or float): integer specifying which GPU to use for
                                remapping, or 'cpu' to use CPU
        per_read_params_dict (dict) : dictionary where keys are UUIDs,
                                      values are dicts containing keys
                                      trim_start trim_end shift scale
        alphabet_info (AlphabetInfo object):  for basecalling
        max_read_length (int) : Don't attempt to remap reads with references
                                longer than this
        localpen (float): Penalty for local mapping

    Returns:
        tuple :(dict,str) containing
        1. dictionary as specified in
            signal_mapping.SignalMapping.get_read_dictionary
        2. message string indicating an error if one occured
    """
    filename, read_id, read_ref = read_tuple

    if read_ref is None:
        return None, RemapResult.NO_REF_FOUND

    if max_read_length is not None and len(read_ref) > max_read_length:
        return None, RemapResult.REF_TOO_LONG

    try:
        read_params_dict = per_read_params_dict[read_id]
    except KeyError:
        return None, RemapResult.NO_PARAMS

    try:
        with fast5_interface.get_fast5_file(filename, 'r') as f5file:
            read = f5file.get_read(read_id)
            sig = signal.Signal(read, read_params=read_params_dict)
    except Exception:
        return None, RemapResult.READ_ID_INFO_NOT_FOUND

    try:
        # Prevents torch doing its own parallelisation on top of our imap_map
        torch.set_num_threads(1)
        # Make signal into 3D tensor with shape [siglength,1,1] and move to
        # appropriate device (GPU  number or CPU)
        signalTensor = torch.tensor(
            sig.standardized_current[:, np.newaxis, np.newaxis].astype(
                taiyaki_dtype), device=device)
        # The model must live on the same device
        modelOnDevice = model.to(device)
        # Apply the network to the signal, generating transition weight matrix,
        # and put it back into a numpy array
        with torch.no_grad():
            transweights = modelOnDevice(signalTensor).cpu().numpy()
    except Exception:
        return None, RemapResult.NETWORK_ERROR

    # Extra dimensions introduced by np.newaxis above removed by np.squeeze
    can_read_ref = alphabet_info.collapse_sequence(read_ref)
    remappingscore, path = flipflop_remap.flipflop_remap(
        np.squeeze(transweights), can_read_ref,
        alphabet=alphabet_info.can_bases, localpen=localpen)
    # read_ref comes out as a bytes object, so we need to convert to str
    # localpen=0.0 does local alignment

    # flipflop_remap() establishes a mapping between the network outputs and
    # the reference.
    # What we need is a mapping between the signal and the reference.
    # To resolve this we need to know the stride of the model (how many samples
    # for each network output)
    model_stride = helpers.guess_model_stride(model)
    int_ref = signal_mapping.SignalMapping.get_integer_reference(
        read_ref, alphabet_info.alphabet)
    sig_mapping = signal_mapping.SignalMapping.from_remapping_path(
        path, int_ref, model_stride, sig)
    try:
        sig_mapping_dict = sig_mapping.get_read_dictionary()
    except signal_mapping.TaiyakiSigMapError as e:
        return None, str(e)
    return sig_mapping_dict, RemapResult.SUCCESS


def generate_output_from_results(results, output, alphabet_info, verbose=True):
    """
    Given an iterable of dictionaries, each representing the results of mapping
    a single read, output a mapped-read file.

    This version outputs to the V8 'chunk' file format (actually containing
    mapped reads, not chunks)

    Args:
        results (iterable): an iterable of read dictionaries (with mappings)
        output (str): output filename
        alphabet_info (AlphabetInfo object): alphabet
    """
    progress = helpers.Progress(quiet=not verbose)
    err_types = defaultdict(int)
    with mapped_signal_files.HDF5Writer(output, alphabet_info) as f:
        for resultdict, mesg in results:
            # filter out error messages for reporting later
            if resultdict is None:
                err_types[mesg] += 1
            else:
                progress.step()
                f.write_read(resultdict)
    sys.stderr.write('\n')

    # report errors at the end to avoid spamming stderr
    sys.stderr.write('* {} reads mapped successfully\n'.format(progress.count))
    if len(err_types) > 0:
        for result, n_errs in err_types.items():
            sys.stderr.write((
                '* {} reads failed to produce remapping results ' +
                'due to: {}\n').format(n_errs, result.value))


def get_per_read_params_dict_from_tsv(input_file):
    """Load per read parameter .tsv into a np array and parse into a dictionary

    Args:
        input_file (str): filename including path for the tsv file

    Returns:
        dict : dictionary with keys being UUIDs, values being named
        tuple('per_read_params', 'trim_start trim_end shift scale')"""
    try:
        per_read_params_array = readtsv(
            input_file, ['UUID', 'trim_start', 'trim_end', 'shift', 'scale'])
    except Exception as e:
        sys.stderr.write(
            'Failed to get per-read parameters from {}.\n{}\n'.format(
                input_file, repr(e)))
        return None

    per_read_params_dict = {}
    for row in per_read_params_array:
        try:
            per_read_params_dict[row[0]] = {
                'trim_start': row[1], 'trim_end': row[2], 'shift': row[3],
                'scale': row[4]}
        except:
            sys.stderr.write(
                "Warning: ignoring incorrect line {} in {}\n".format(
                    row, input_file))

    return per_read_params_dict
