# Functions to select and filter chunks for training.
# Data structures are based on the read dictionary defined in
# signal_mapping.SignalMapping
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


def sample_chunks(read_data, number_to_sample, chunk_len, filter_params,
                  fraction_of_fails_allowed=0.5,
                  chunk_len_means_sequence_len=False,
                  standardize=True, select_strands_randomly=True,
                  first_strand_index=0):
    """Sample <number_to_sample> chunks from a list of read_data, returning
    a tuple (chunklist, rejection_dict).

    Random read sampling continues until <number_to_sample> chunks that pass
    filter_params have been sampled or <fraction_of_fails_allowed> has been
    reached.

    rejection_dict is a dictionary with keys describing the reasons for
    rejection and values being the number rejected for that reason. E.g.
    {'pass':3,'meandwell':3, 'maxdwell':4}.

    param: read_data: list of signal_mapping.SignalMapping objects
    param: number_to_sample: target number of data elements to return, each
        from a sampled chunk. If number_to_sample is 0 or None then get the
        same number of chunks as the number of read_data items supplied.
    param: chunk_len: desired length of chunk in samples, or length of
        sequence in bases if chunk_len_means_sequence_len
    param: fraction_of_fails_allowed: Visit a maximum of
        (number_to_sample / fraction_of_fails_allowed) reads before stopping.
    param: filter_params: taiyaki.chunk_selection.FILTER_PARAMETERS namedtuple
    param: chunk_len_means_sequence_len: if this is False (the default) then
        chunk_len determines the length in samples of the chunk, and we
        use signal_mapping.SignalMapping.get_chunk_with_sample_length().
        If this is True, then chunk_len determines the length in bases of the
        sequence in the chunk, and we use
        signal_mapping.SignalMapping.get_chunk_with_sequence_length()
    :param standardize: Return standardized currents, otherwise unscaled
    param: select_strands_randomly : Choose a random read at each iteration.
        When select_strands_randomly=False, iterate through reads as found
        in read_data.
    param: first_strand_index : When select_strands_randomly=False, begin
        selecting strands at this index.
    """
    nreads = len(read_data)
    if number_to_sample is None or number_to_sample == 0:
        number_to_sample_used = nreads
    else:
        number_to_sample_used = number_to_sample
    maximum_attempts_allowed = int(
        number_to_sample_used / fraction_of_fails_allowed)
    chunks = []
    # Will contain counts of numbers of rejects and passes
    rejection_reasons = defaultdict(lambda: 0)
    attempts = 0
    while(len(chunks) < number_to_sample_used and
          attempts < maximum_attempts_allowed):
        read_number = np.random.randint(nreads) if select_strands_randomly else \
            (first_strand_index + attempts) % nreads
        attempts += 1
        read = read_data[read_number]
        if chunk_len_means_sequence_len:
            chunk = read.get_chunk_with_sequence_length(
                chunk_len, standardize=standardize)
        else:
            chunk = read.get_chunk_with_sample_length(
                chunk_len, standardize=standardize)
        chunk.apply_filters(filter_params)
        rejection_reasons[chunk.reject_reason] += 1
        if chunk.accepted:
            chunks.append(chunk)

    return chunks, rejection_reasons


def sample_filter_parameters(read_data, number_to_sample, chunk_len,
                             filter_mean_dwell, filter_max_dwell,
                             chunk_len_means_sequence_len=False):
    """Sample number_to_sample reads from read_data, calculate median and MAD
    of mean dwell. Note the MAD has an adjustment factor so that it would give the
    same result as the std for a normal distribution.

    See FILTER_PARAMETERS docstring for details.
    """
    no_filter_params = FILTER_PARAMETERS(
        filter_mean_dwell=filter_mean_dwell, filter_max_dwell=filter_max_dwell,
        median_meandwell=None, mad_meandwell=None)
    chunks, _ = sample_chunks(read_data, number_to_sample, chunk_len,
                              no_filter_params,
                              chunk_len_means_sequence_len=chunk_len_means_sequence_len)
    meandwells = [chunk.mean_dwell for chunk in chunks]
    median_meandwell, mad_meandwell = med_mad(meandwells)
    return FILTER_PARAMETERS(
        filter_mean_dwell=filter_mean_dwell, filter_max_dwell=filter_max_dwell,
        median_meandwell=median_meandwell, mad_meandwell=mad_meandwell)
