#!/usr/bin/env python3
# Combine mapped-read files in HDF5 format into a single file

import argparse
import numpy as np
import sys
from taiyaki import alphabet, mapped_signal_files

parser = argparse.ArgumentParser(
    description='Split a mapped signal file into two groups. Intended for ' +
    'train test splitting.')
parser.add_argument('output_base', help='Output filenames base')
parser.add_argument('input', help='Input file')
parser.add_argument(
    '--number_of_test_reads', type=int,
    help='Number of reads to include in test file.')
parser.add_argument(
    '--fraction_of_test_reads', type=float,
    help='Fraction of reads to include in test file.')

# To convert to any new mapped read format (e.g. mapped_signal_files.SQL)
# we should be able to just change MAPPED_SIGNAL_WRITER amd READER to equal the
# new classes.
MAPPED_SIGNAL_WRITER = mapped_signal_files.HDF5Writer
MAPPED_SIGNAL_READER = mapped_signal_files.HDF5Reader

def write_train_test_reads(hin, read_ids, num_test_reads, out_base):
    alphabet_info = hin.get_alphabet_information()
    np.random.shuffle(read_ids)

    with MAPPED_SIGNAL_WRITER(out_base + '.test.hdf5', alphabet_info) as hout:
        for read_id in read_ids[:num_test_reads]:
            read = hin.get_read(read_id)
            read['read_id'] = read_id
            hout.write_read(read)

    with MAPPED_SIGNAL_WRITER(out_base + '.train.hdf5', alphabet_info) as hout:
        for read_id in read_ids[num_test_reads:]:
            read = hin.get_read(read_id)
            read['read_id'] = read_id
            hout.write_read(read)

    return

def main():
    args = parser.parse_args()

    hin = MAPPED_SIGNAL_READER(args.input)
    read_ids = hin.get_read_ids()
    if args.number_of_test_reads is None:
        if args.fraction_of_test_reads is None:
            sys.stderr.write(
                'Must provide either --number_of_test_reads or ' +
                '--fraction_of_test_reads.\n')
            sys.exit(1)
        else:
            num_test_reads = int(
                len(read_ids) * args.args.fraction_of_test_reads)
    else:
        if args.fraction_of_test_reads is None:
            num_test_reads = args.number_of_test_reads
        else:
            sys.stderr.write(
                'Both --number_of_test_reads and --fraction_of_test_reads ' +
                'provided. Only provide one.\n')
            sys.exit(1)

    sys.stderr.write(
        "Copying {} reads to test file and {} reads to train file.\n".format(
            num_test_reads, len(read_ids) - num_test_reads))
    write_train_test_reads(hin, read_ids, num_test_reads, args.output_base)

    return

if __name__ == '__main__':
    main()
