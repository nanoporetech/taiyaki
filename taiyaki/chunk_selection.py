# Functions to select and filter chunks for training.
# Data structures are based on the read dictionary defined in mapped_signal_files.py
from collections import defaultdict, namedtuple
import os
import numpy as np
from taiyaki.maths import med_mad


class FILTER_PARAMETERS(namedtuple(
        'FILTER_PARAMETERS', (
            'filter_mean_dwell', 'filter_max_dwell',
            'median_meandwell', 'mad_meandwell'))):
    """ Parameters to filter signal chunk selections

    param: filter_mean_dwell : Number of deviations from median to reject read
    param: filter_max_dwell  : Multiple of median dwell to reject read based
        based on maximum dwell.
    param: median_meandwell  : Drop chunks with mean dwell more than radius
        deviations from the median
    param: mad_meandwell     : Drop chunks with max dwell more than multiple
        of median
    """


def get_mean_dwell(chunkdict, TINY=0.00000001):
    """Calculate mean dwell from the dict of data returned by
    mapped_signal_files.Read.get_chunk_with_sample_length()
    or by mapped_signal_files.Read.get_chunk_with_sequence_length().
    TINY is added to the denominator to avoid overflow in the case
    of zero sequence length"""
    return len(chunkdict['current']) / (len(chunkdict['sequence']) + TINY)


def chunk_filter(chunkdict, filter_params):
    """Given a chunk dict as returned by mapped_signal_files.Read._get_chunk(),
    apply filtering conditions, returning "pass" if everything OK
    or a string describing reason for failure if not.

    param: chunkdict         : a dictionary as returned by mapped_signal_files.get_chunk()
    param: filter_params : taiyaki.chunk_selection.FILTER_PARAMETERS namedtuple

    If filter_params.median_mean_dwell or filter_params.mad_dwell are
    None, then don't filter according to dwell, but still reject reads which
    haven't produced a chunk at all because they're not long enough or end in
    a slip.
    """
    if chunkdict is None:
        # Not possible to get a chunk
        return "nochunk"
    if 'rejected' in chunkdict:
        # The chunkdict contains a reason why it should be rejected. Return this.
        return chunkdict['rejected']

    if (filter_params.median_meandwell is None or
        filter_params.mad_meandwell is None):
        #  Short-circuit no filtering
        return 'pass'

    mean_dwell = get_mean_dwell(chunkdict)
    mean_dwell_dev_from_median = abs(
        mean_dwell - filter_params.median_meandwell)
    if (mean_dwell_dev_from_median >
        filter_params.filter_mean_dwell * filter_params.mad_meandwell):
        #  Failed mean dwell filter
        return 'meandwell'

    if (chunkdict['max_dwell'] >
        filter_params.filter_max_dwell * filter_params.median_meandwell):
        #  Failed maximum dwell filter
        return 'maxdwell'
    return 'pass'


def sample_chunks(read_data, number_to_sample, chunk_len, filter_params,
                  fraction_of_fails_allowed=0.5,
                  chunk_len_means_sequence_len=False):
    """Sample <number_to_sample> chunks from a list of read_data, returning
    a tuple (chunklist, rejection_dict)

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
    param: fraction_of_fails_allowed : Visit a maximum of
                             (number_to_sample / fraction_of_fails_allowed) reads
                             before stopping.
    param: filter_params    : taiyaki.chunk_selection.FILTER_PARAMETERS namedtuple
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
        passfail_str = chunk_filter(chunkdict, filter_params)
        count_dict[passfail_str] += 1
        if passfail_str == 'pass':
            chunklist.append(chunkdict)

    return chunklist, count_dict


def sample_filter_parameters(read_data, number_to_sample, chunk_len,
                             filter_mean_dwell, filter_max_dwell,
                             chunk_len_means_sequence_len = False):
    """Sample number_to_sample reads from read_data, calculate median and MAD
    of mean dwell. Note the MAD has an adjustment factor so that it would give the
    same result as the std for a normal distribution.

    See docstring for sample_chunks() for the parameters.
    """
    no_filter_params = FILTER_PARAMETERS(
        filter_mean_dwell=filter_mean_dwell, filter_max_dwell=filter_max_dwell,
        median_meandwell=None, mad_meandwell=None)
    chunks, _ = sample_chunks(read_data, number_to_sample, chunk_len,
                              no_filter_params,
                              chunk_len_means_sequence_len=chunk_len_means_sequence_len)
    meandwells = [get_mean_dwell(chunk) for chunk in chunks]
    median_meandwell, mad_meandwell = med_mad(meandwells)
    return FILTER_PARAMETERS(
        filter_mean_dwell=filter_mean_dwell, filter_max_dwell=filter_max_dwell,
        median_meandwell=median_meandwell, mad_meandwell=mad_meandwell)


def assemble_batch(read_data, batch_size, chunk_len, filter_params,
                   chunk_len_means_sequence_len=False):
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
    return sample_chunks(read_data, batch_size, chunk_len,
                         filter_params=filter_params,
                         chunk_len_means_sequence_len=chunk_len_means_sequence_len)
