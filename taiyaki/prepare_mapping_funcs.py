from collections import defaultdict
import enum
import numpy as np
import sys
from ont_fast5_api import fast5_interface
import torch
from taiyaki import flipflop_remap, helpers, mapping, mapped_signal_files, signal
from taiyaki.config import taiyaki_dtype
from taiyaki.fileio import readtsv


class RemapResult(enum.Enum):
    SUCCESS = 'Success!'
    READ_ID_INFO_NOT_FOUND = 'No information for read id found in file.'
    NO_REF_FOUND = 'No fasta reference found.'
    NO_PARAMS = 'No per-read params provided.'
    NETWORK_ERROR = 'Failure applying basecall network to remap read.'
    REF_TOO_LONG = 'Reference exceeded maximum allowed read length.'


def oneread_remap(read_tuple, model, per_read_params_dict,
                  alphabet_info, max_read_length, device='cpu'):
    """ Worker function for remapping reads using flip-flop model on raw signal
    :param read_tuple                 : read, identified by a tuple (filepath, read_id, read reference)
    :param model                      :pytorch model (the torch data structure, not a filename)
    :param device                     :integer specifying which GPU to use for remapping, or 'cpu' to use CPU
    :param per_read_params_dict       :dictionary where keys are UUIDs, values are dicts containing keys
                                         trim_start trim_end shift scale
    :param alphabet_info              : AlphabetInfo object for basecalling
    :param max_read_length            : Don't attempt to remap reads with references longer than this

    :returns: tuple of dictionary as specified in mapped_signal_files.Read class
              and a message string indicating an error if one occured
    """
    filename, read_id, read_ref = read_tuple

    if read_ref is None:
        return None, RemapResult.NO_REF_FOUND

    if max_read_length is not None and len(read_ref) > max_read_length:
        return None, RemapResult.REF_TOO_LONG

    try:
        with fast5_interface.get_fast5_file(filename, 'r') as f5file:
            read = f5file.get_read(read_id)
            sig = signal.Signal(read)
    except Exception:
        return None, RemapResult.READ_ID_INFO_NOT_FOUND

    try:
        read_params_dict = per_read_params_dict[read_id]
    except KeyError:
        return None, RemapResult.NO_PARAMS

    sig.set_trim_absolute(read_params_dict['trim_start'], read_params_dict['trim_end'])

    try:
        torch.set_num_threads(1)  # Prevents torch doing its own parallelisation on top of our imap_map
        # Standardise (i.e. shift/scale so that approximately mean =0, std=1)
        signalArray = (sig.current - read_params_dict['shift']) / read_params_dict['scale']
        # Make signal into 3D tensor with shape [siglength,1,1] and move to appropriate device (GPU  number or CPU)
        signalTensor = torch.tensor(signalArray[:, np.newaxis, np.newaxis].astype(taiyaki_dtype) , device=device)
        # The model must live on the same device
        modelOnDevice = model.to(device)
        # Apply the network to the signal, generating transition weight matrix, and put it back into a numpy array
        with torch.no_grad():
            transweights = modelOnDevice(signalTensor).cpu().numpy()
    except Exception:
        return None, RemapResult.NETWORK_ERROR

    # Extra dimensions introduced by np.newaxis above removed by np.squeeze
    can_read_ref = alphabet_info.collapse_sequence(read_ref)
    remappingscore, path = flipflop_remap.flipflop_remap(
        np.squeeze(transweights), can_read_ref,
        alphabet=alphabet_info.can_bases, localpen=0.0)
    # read_ref comes out as a bytes object, so we need to convert to str
    # localpen=0.0 does local alignment

    # flipflop_remap() establishes a mapping between the network outputs and the reference.
    # What we need is a mapping between the signal and the reference.
    # To resolve this we need to know the stride of the model (how many samples for each network output)
    model_stride = helpers.guess_model_stride(model)
    remapping = mapping.Mapping.from_remapping_path(
        sig, path, read_ref, model_stride)
    remapping.add_integer_reference(alphabet_info.alphabet)

    return remapping.get_read_dictionary(read_params_dict['shift'],
                                         read_params_dict['scale'],
                                         read_id), RemapResult.SUCCESS


def generate_output_from_results(results, output, alphabet_info):
    """
    Given an iterable of dictionaries, each representing the results of mapping
    a single read, output a mapped-read file.
    This version outputs to the V8 'chunk' file format (actually containing mapped reads, not chunks)

    param: results     : an iterable of read dictionaries
                         (with mappings)
    param: output      : output filename
    param: alphabet_info : taiyaki.alphabet.AlphabetInfo instance
    """
    progress = helpers.Progress()
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
    :param input_file     :   filename including path for the tsv file
    :returns:                 dictionary with keys being UUIDs, values being named
                              tuple('per_read_params', 'trim_start trim_end shift scale')"""
    try:
        per_read_params_array = readtsv(input_file, ['UUID', 'trim_start', 'trim_end', 'shift', 'scale'])
    except Exception as e:
        sys.stderr.write('Failed to get per-read parameters from {}.\n{}\n'.format(input_file, repr(e)))
        return None

    per_read_params_dict = {}
    for row in per_read_params_array:
        per_read_params_dict[row[0]] = {'trim_start': row[1],
                                        'trim_end': row[2],
                                        'shift': row[3],
                                        'scale': row[4]}
    return per_read_params_dict
