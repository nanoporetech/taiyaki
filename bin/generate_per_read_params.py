#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import sys

from ont_fast5_api import fast5_interface
from taiyaki.cmdargs import NonNegative
from taiyaki.common_cmdargs import add_common_command_args
import taiyaki.fast5utils as fast5utils
from taiyaki.helpers import open_file_or_stdout
from taiyaki.iterators import imap_mp
from taiyaki.maths import med_mad
from taiyaki.signal import Signal


def get_parser():
    parser = argparse.ArgumentParser()

    add_common_command_args(
        parser, ('input_folder input_strand_list limit output ' +
                 'recursive version jobs').split())

    parser.add_argument(
        '--trim', default=(200, 50), nargs=2, type=NonNegative(int),
        metavar=('beginning', 'end'),
        help='Number of samples to trim off start and end')

    return parser


def one_read_shift_scale(read_tuple):
    """  Read signal from fast5 and perform medmad scaling

    Args:
        read_tuple (tuple of str and str): A filename and the read_id to read
            from it.

    Returns:
        tuple of str and float and float: read_id of the read and the
            calculated shift and scale parameters.

        If the signal is unable to be read from the file, the read_id is not
        present for example, then (None, , None, None) is returned.

        When a signal is read, but has zero length, the shift and scale
        returned are `np.NaN`
    """
    read_filename, read_id = read_tuple

    try:
        with fast5_interface.get_fast5_file(read_filename, 'r') as f5file:
            read = f5file.get_read(read_id)
            sig = Signal(read)

    except Exception as e:
        sys.stderr.write(
            'Unable to obtain signal for {} from {}.\n{}\n'.format(
                read_id, read_filename, repr(e)))
        return (None, None, None)

    else:
        signal = sig.current

        if len(signal) > 0:
            shift, scale = med_mad(signal)
        else:
            shift, scale = np.NaN, np.NaN
            # Note - if signal trimmed by ub, it could be of length zero by
            # this point for short reads
            # These are taken out later in the existing code, in the new code
            # we'll take out ub trimming

        return (read_id, shift, scale)


def main():
    args = get_parser().parse_args()

    trim_start, trim_end = args.trim

    fast5_reads = fast5utils.iterate_fast5_reads(
        args.input_folder, limit=args.limit,
        strand_list=args.input_strand_list, recursive=args.recursive)

    with open_file_or_stdout(args.output) as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        # UUID is 32hexdigits and four dashes eg.
        # '43f6a05c-0856-4edc-8cd2-4866d9d60eaa'
        writer.writerow(['UUID', 'trim_start', 'trim_end', 'shift', 'scale'])

        results = imap_mp(one_read_shift_scale, fast5_reads, threads=args.jobs)

        for result in results:
            if all(result):
                read_id, shift, scale = result
                writer.writerow([read_id, trim_start, trim_end, shift, scale])


if __name__ == '__main__':
    main()
