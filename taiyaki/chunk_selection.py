# Functions to select and filter chunks for training.
# Data structures are based on the read dictionary defined in mapped_signal_files.py
from collections import defaultdict
import os
import numpy as np
from taiyaki.maths import med_mad


def get_mean_dwell(chunkdict, TINY=0.00000001):
    """Calculate mean dwell from the dict of data returned by
    mapped_signal_files.Read.get_chunk_with_sample_length()
    or by mapped_signal_files.Read.get_chunk_with_sequence_length().
    TINY is added to the denominator to avoid overflow in the case
    of zero sequence length"""
    if not ('current' in chunkdict and 'sequence' in chunkdict):
        return None
    return len(chunkdict['current']) / (len(chunkdict['sequence']) + TINY)


def chunk_filter(chunkdict, args, filter_parameters):
    """Given a chunk dict as returned by mapped_signal_files.Read._get_chunk(),
    apply filtering conditions, returning "pass" if everything OK
    or a string describing reason for failure if not.

    param : chunkdict        : a dictionary as returned by mapped_signal_files.get_chunk()
    param : args             : command-line args object used to determine filter limits
    param : filter_parameters: tuple median(mean_dwell), mad(mean_dwell) from sampled
                               reads, used to determine filter centre values

    If filter_parameters is None, then don't filter according to dwell,
    but still reject reads which haven't produced a chunk at all because
    they're not long enough or end in a slip.
    """
    if chunkdict is None:  # Not possible to get a chunk
        return "nochunk"
    if 'rejected' in chunkdict:
        # The chunkdict contains a reason why it should be rejected. Return this.
        return chunkdict['rejected']

    if filter_parameters is not None:
        mean_dwell = get_mean_dwell(chunkdict)
        if mean_dwell is None:
            return 'data_missing'
        median_meandwell, mad_meandwell = filter_parameters
        mean_dwell_dev_from_median = abs(mean_dwell - median_meandwell)
        if mean_dwell_dev_from_median > args.filter_mean_dwell * mad_meandwell:
            return 'meandwell'
        if chunkdict['max_dwell'] > args.filter_max_dwell * median_meandwell:
            return 'maxdwell'
    return 'pass'


def sample_chunks(read_data, number_to_sample, chunk_len, args, chunkfunc,
                  fraction_of_fails_allowed=0.5,
                  filter_parameters=None, log=None,
                  chunk_log=None,
                  log_accepted_chunks=False,
                  chunk_len_means_sequence_len=False):
    """Sample <number_to_sample> chunks from a list of read_data, returning
    a tuple (chunklist, rejection_dict), where chunklist contains the
    results of applying <chunkfunc> to each chunk that is not rejected.

    rejection_dict is a dictionary with keys describing the reasons for
    rejection and values being the number rejected for that reason. E.g.
    {'pass':3,'meandwell':3, 'maxdwell':4}.

    param: read_data        : a list of Read objects as defined in mapped_signal_files.py
    param: number_to_sample : target number of data elements to return, each from
                              a sampled chunk. If number_to_sample is 0 or None
                              then get the same number of chunks as the number
                              of read_data items supplied.
    param: chunk_len        : desired length of chunk in samples, or length
                              of sequence in bases if chunk_len_means_sequence_len
    param: args             : command-line args object from argparse. Used to
                              pass the filter command-line arguments.
    param: chunkfunc        : the function to be applied to the chunkdict to get a single
                              data item in the list returned
    param: fraction_of_fails_allowed : Visit a maximum of
                             (number_to_sample / fraction_of_fails_allowed) reads
                             before stopping.
    param: filter_parameters: a tuple (median_meandwell, mad_meandwell) which
                              determines the filter used. If None, then no filtering.
    param: log              : log object used to report if not enough chunks
                              passing the tests can be found.
    param: chunk_log        : ChunkLog object used to record rejected chunks
                              (accepted chunks will be recorded along with their
                              loss scores after the training step)
    param: log_accepted_chunks : If this is True, then we log all chunks.
                                 If it's false, then we log only rejected ones.
                                 During training we use log_accepted_chunks=False
                                 because the accepted chunks are logged later
                                 when their loss has been calculated.
    param: chunk_len_means_sequence_len : if this is False (the default) then
                             chunk_len determines the length in samples of the
                             chunk, and we use mapped_signal_files.get_chunk_with_sample_length().
                             If this is True, then chunk_len determines the length
                             in bases of the sequence in the chunk, and we use
                             mapped_signal_files.get_chunk_with_sequence_length()
    """
    nreads = len(read_data)
    if number_to_sample is None or number_to_sample == 0:
        number_to_sample_used = nreads
    else:
        number_to_sample_used = number_to_sample
    maximum_attempts_allowed = int(number_to_sample_used / fraction_of_fails_allowed)
    chunklist = []
    count_dict = defaultdict(lambda: 0)  # Will contain counts of numbers of rejects and passes
    attempts = 0
    while(count_dict['pass'] < number_to_sample_used and attempts < maximum_attempts_allowed):
        attempts += 1
        read_number = np.random.randint(nreads)
        read = read_data[read_number]
        if chunk_len_means_sequence_len:
            chunkdict = read.get_chunk_with_sequence_length(chunk_len)
        else:
            chunkdict = read.get_chunk_with_sample_length(chunk_len)
        passfail_str = chunk_filter(chunkdict, args, filter_parameters)
        count_dict[passfail_str] += 1
        if passfail_str == 'pass':
            chunklist.append(chunkfunc(chunkdict))
        if log_accepted_chunks or passfail_str != 'pass':
            if chunk_log is not None:
                chunk_log.write_chunk(-1, chunkdict, passfail_str)

    if count_dict['pass'] < number_to_sample_used and log is not None:
        log.write('* Warning: only {} chunks passed tests after {} attempts.\n'.format(count_dict['pass'], attempts))
        log.write('* Summary:')
        for k, v in count_dict.items():
            log.write(' {}:{}'.format(k, v))
        log.write('\n')

    return chunklist, count_dict


def sample_filter_parameters(read_data, number_to_sample, chunk_len, args,
                             log=None, chunk_log=None,
                             chunk_len_means_sequence_len = False):
    """Sample number_to_sample reads from read_data, calculate median and MAD
    of mean dwell. Note the MAD has an adjustment factor so that it would give the
    same result as the std for a normal distribution.

    See docstring for sample_chunks() for the parameters.
    """
    meandwells, _ = sample_chunks(read_data, number_to_sample, chunk_len, args, get_mean_dwell,
                                  log=log, chunk_log=chunk_log, log_accepted_chunks=True,
                                  chunk_len_means_sequence_len=chunk_len_means_sequence_len)
    return med_mad(meandwells)


def assemble_batch(read_data, batch_size, chunk_len, filter_parameters, args, log,
                   chunk_log=None, chunk_len_means_sequence_len=False):
    """Assemble a batch of data by repeatedly choosing a random read and location
    in that read, continuing until we have found batch_size chunks that pass the
    tests.

    Returns tuple (chunklist, rejection_dict)

    where chunklist is a list of dicts, each with entries
        (signal_chunk, sequence_chunk, start_sample, read_id).
        signal_chunks and sequence_chunks are np arrays.
    and rejection_dict is a dictionary with keys describing the reasons for
        rejection and values being the number rejected for that reason. E.g.
        {'pass':3,'meandwell':3, 'maxdwell':4}.

    If we can't find enough chunks after the allowed number of attempts ,then
    return the short batch, but output a message to the log.

    See docstring for sample_chunks for parameters.
    """
    return sample_chunks(read_data, batch_size, chunk_len, chunkfunc=lambda x: x,
                         filter_parameters=filter_parameters, log=log,
                         chunk_log=chunk_log, args=args,
                         chunk_len_means_sequence_len=chunk_len_means_sequence_len)


class ChunkLog:
    """Handles saving of chunk metadata to file"""

    def __init__(self, outputdirectory, outputfilename="chunklog.tsv"):
        """Open and write header line"""
        filepath = os.path.join(outputdirectory, outputfilename)
        self.dumpfile = open(filepath, "w")
        self.dumpfile.write(
            "iteration\t read_id\t start_sample\t chunk_len_samples\t chunk_len_bases\t max_dwell\t status\t loss\n")

    def write_chunk(self, iteration, chunk_dict, status, lossvalue=None, loss_not_calculated=-1.0):
        """Write a single line of data to the chunk log, using -1 to indicate missing data.
        param iteration  : the training iteration (measured in batches, or -1 if not used in training)
        param chunk_dict : chunk dictionary
        param status     : string for reject/accept status (e.g. 'pass', 'meandwell')
        param lossvalue  : loss if available (not calculated for rejected chunks)
        param loss_not_calculated : value to store in the log file in the loss column
                                    for chunks where loss has not been calculated
        """
        format_string = ("{}\t" * 6) + "{}\n"
        if lossvalue is None:
            lossvalue_written = loss_not_calculated
        else:
            lossvalue_written = lossvalue
        if chunk_dict is None:
            self.dumpfile.write(format_string.format(iteration, '--------', -1, -1, -1, status, lossvalue_written))
        else:
            # Some elements of dict may be missing if chunk construction failed
            self.dumpfile.write('{}\t{}\t'.format(iteration, chunk_dict['read_id']))
            if 'start_sample' in chunk_dict:
                self.dumpfile.write('{}\t'.format(chunk_dict['start_sample']))
            else:
                self.dumpfile.write('-1\t')
            for k in ['current', 'sequence']:
                if k in chunk_dict:
                    self.dumpfile.write('{}\t'.format(len(chunk_dict[k])))
                else:
                    self.dumpfile.write('-1\t')
            if 'max_dwell' in chunk_dict:
                self.dumpfile.write('{}\t'.format(chunk_dict['max_dwell']))
            else:
                self.dumpfile.write('-1\t')
            self.dumpfile.write("{}\t{}\n".format(status, lossvalue_written))

    def write_batch(self, iteration, chunk_batch, lossvector):
        """Write information about a single batch to the log.
        All these chunks will have been accepted, so their status is 'pass'"""
        for chunk_dict, lossvalue in zip(chunk_batch, lossvector):
            self.write_chunk(iteration, chunk_dict, "pass", lossvalue)
