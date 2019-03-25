#!/usr/bin/env python3
import argparse
import csv
from functools import partial
import numpy as np
import os
import sys

from ont_fast5_api import fast5_interface
from taiyaki.cmdargs import Maybe, NonNegative, Positive
import taiyaki.common_cmdargs as common_cmdargs
import taiyaki.fast5utils as fast5utils
from taiyaki.iterators import imap_mp
from taiyaki.maths import med_mad
from taiyaki.signal import Signal

parser = argparse.ArgumentParser()

common_cmdargs.add_common_command_args(parser, 'input_folder input_strand_list limit overwrite recursive version jobs'.split())

parser.add_argument('--trim', default=(200, 50), nargs=2, type=NonNegative(int),
                    metavar=('beginning', 'end'), help='Number of samples to trim off start and end')

parser.add_argument('output', help='Output .tsv file')


def one_read_shift_scale(read_tuple):

    read_filename, read_id = read_tuple

    try:
        with fast5_interface.get_fast5_file(read_filename, 'r') as f5file:
            read = f5file.get_read(read_id)
            sig = Signal(read)

    except Exception as e:
        sys.stderr.write('Unable to obtain signal for {} from {}.\n{}\n'.format(
            read_id, read_filename, repr(e)))
        return (None, None, None)

    else:
        signal = sig.current

        if len(signal) > 0:
            shift, scale = med_mad(signal)
        else:
            shift, scale = np.NaN, np.NaN
            # note - if signal trimmed by ub, it could be of length zero by this point for short reads
            # These are taken out later in the existing code, in the new code we'll take out ub trimming

        return (read_id, shift, scale)


if __name__ == '__main__':

    args = parser.parse_args()

    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)

    fast5_reads = fast5utils.iterate_fast5_reads(
        args.input_folder, limit=args.limit, strand_list=args.input_strand_list,
        recursive=args.recursive)
    trim_start, trim_end = args.trim

    with open(args.output, 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        # UUID is 32hexdigits and four dashes eg. '43f6a05c-0856-4edc-8cd2-4866d9d60eaa'
        writer.writerow(['UUID', 'trim_start', 'trim_end', 'shift', 'scale'])

        results = imap_mp(one_read_shift_scale, fast5_reads, threads=args.jobs)

        for result in results:
            if all(result):
                read_id, shift, scale = result
                writer.writerow([read_id, trim_start, trim_end, shift, scale])

