import numpy as np
import sys
from ont_fast5_api import fast5_interface
import torch
from taiyaki import flipflop_remap, helpers, mapping, mapped_signal_files, signal
from taiyaki.config import taiyaki_dtype
from taiyaki.fileio import readtsv


def oneread_remap(read_tuple, references, model, device, per_read_params_dict,
                  alphabet, collapse_alphabet):
    """ Worker function for remapping reads using flip-flop model on raw signal
    :param read_tuple                 : read, identified by a tuple (filepath, read_id)
    :param references                 :dict mapping fast5 filenames to reference strings
    :param model                      :pytorch model (the torch data structure, not a filename)
    :param device                     :integer specifying which GPU to use for remapping, or 'cpu' to use CPU
    :param per_read_params_dict       :dictionary where keys are UUIDs, values are dicts containing keys
                                         trim_start trim_end shift scale
    :param alphabet                   : alphabet for basecalling (passed on to mapped-read file)
    :param collapse_alphabet          : collapsed alphabet for basecalling (passed on to mapped-read file)

    :returns: dictionary as specified in mapped_signal_files.Read class
    """
    filename, read_id = read_tuple
    try:
        with fast5_interface.get_fast5_file(filename, 'r') as f5file:
            read = f5file.get_read(read_id)
            sig = signal.Signal(read)
    except Exception as e:
        # We want any single failure in the batch of reads to not disrupt other reads being processed.
        sys.stderr.write('No read information on read {} found in file {}.\n{}\n'.format(read_id, filename, repr(e)))
        return None

    if read_id in references:
        read_ref = references[read_id].decode("utf-8")
    else:
        sys.stderr.write('No fasta reference found for {}.\n'.format(read_id))
        return None

    if read_id in per_read_params_dict:
        read_params_dict = per_read_params_dict[read_id]
    else:
        return None

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
    except Exception as e:
        sys.stderr.write("Failure applying basecall network to remap read {}.\n{}\n".format(sig.read_id, repr(e)))
        return None

    # Extra dimensions introduced by np.newaxis above removed by np.squeeze
    remappingscore, path = flipflop_remap.flipflop_remap(
        np.squeeze(transweights), read_ref, localpen=0.0)
    # read_ref comes out as a bytes object, so we need to convert to str
    # localpen=0.0 does local alignment

    # flipflop_remap() establishes a mapping between the network outputs and the reference.
    # What we need is a mapping between the signal and the reference.
    # To resolve this we need to know the stride of the model (how many samples for each network output)
    model_stride = helpers.guess_model_stride(model, device=device)
    remapping = mapping.Mapping.from_remapping_path(sig, path, read_ref, model_stride)

    return remapping.get_read_dictionary(read_params_dict['shift'], read_params_dict['scale'], read_id,
                                      alphabet=alphabet, collapse_alphabet=collapse_alphabet)



def generate_output_from_results(results, args):
    """
    Given an iterable of dictionaries, each representing the results of mapping
    a single read, output a mapped-read file.
    This version outputs to the V7 'chunk' file format (actually containing mapped reads, not chunks)

    param: results     : an iterable of read dictionaries
                         (with mappings)
    param: args        : command line args object
    """
    progress = helpers.Progress()

    # filter removes None and False and 0; filter(None,  is same as filter(o:o,
    read_ids = []
    with mapped_signal_files.HDF5(args.output, "w") as f:
        f.write_version_number()
        for readnumber, resultdict in enumerate(filter(None, results)):
            progress.step()
            read_id = resultdict['read_id']
            read_ids.append(read_id)
            f.write_read(read_id, mapped_signal_files.Read(resultdict))


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
        per_read_params_dict[row[0]] = {'trim_start': row[1], 'trim_end': row[2], 'shift': row[3], 'scale': row[4]}
    return per_read_params_dict
