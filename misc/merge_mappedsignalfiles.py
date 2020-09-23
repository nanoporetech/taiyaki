#!/usr/bin/env python3
# Combine mapped-read files in HDF5 format into a single file

import argparse
import numpy as np
import sys

from taiyaki import alphabet, mapped_signal_files
from taiyaki.cmdargs import AutoBool, Maybe, NonNegative


# To convert to any new mapped read format (e.g. mapped_signal_files.SQL)
# we should be able to just change MAPPED_SIGNAL_WRITER amd READER to equal the
# new classes.
MAPPED_SIGNAL_WRITER = mapped_signal_files.HDF5Writer
MAPPED_SIGNAL_READER = mapped_signal_files.HDF5Reader


def get_parser():
    parser = argparse.ArgumentParser(
        description='Combine HDF5 mapped-signal files into a single file. ' +
        'Checks that alphabets are compatible.')
    parser.add_argument('output', help='Output filename')

    parser.add_argument(
        '--input', required=True, nargs=2, action='append',
        metavar=('mapped_signal_file', 'num_reads'),
        help='Mapped signal filename and the number of reads to merge from ' +
        'this file. Specify "None" to merge all reads from a file.')
    parser.add_argument(
        '--load_in_mem', action=AutoBool, default=False,
        help='Load each input file into memory before processing. ' +
        'Potentially large increase in speed but also increased memory usage')
    parser.add_argument(
        '--seed', type=Maybe(NonNegative(int)), default=None,
        help='Seed for randomly selected reads when limits are set ' +
        '(default random seed)')
    parser.add_argument(
        '--allow_mod_merge', action='store_true',
        help='Allow merging of data sets with different modified bases. ' +
        'While alphabets may differ, incompatible alphabets are not allowed ' +
        '(e.g. same single letter code used for different canonical bases).')

    return parser


def none_or_int(num):
    return None if num == 'None' else int(num)


def check_version(file_handle, filename):
    """Check to make sure version agrees with the file format version in
    this edition of Taiyaki. If not, throw an exception."""
    if mapped_signal_files._version != file_handle.version:
        raise Exception(
            ("File version of mapped signal file ({}, version {}) does " +
             "not match this version of Taiyaki (file version {})").format(
                 filename, file_handle.version, mapped_signal_files._version))
    return


def validate_and_merge_alphabets(in_fns):
    """ Validate that all alphabets are compatible. Alphabets can be
    incompatible if:
      1) Same mod_base corresponds to different can_base
      2) Same mod_base has different mod_long_names
      3) Same mod_long_name has different mod_bases

    Return the merge_alphabet_info object

    Also check file versions so this this doesn't short circuit a longer run.
    """
    all_alphabets = []
    for in_fn in in_fns:
        with MAPPED_SIGNAL_READER(in_fn) as hin:
            all_alphabets.append(hin.get_alphabet_information())
            check_version(hin, in_fn)

    can_bases = all_alphabets[0].can_bases
    if not all((file_alphabet.can_bases == can_bases
                for file_alphabet in all_alphabets)):
        sys.stderr.write(
            "All canonical alphabets must be the same for " +
            "--allow_mod_merge. Got: {}\n".format(
                ', '.join(set(fa.can_bases for fa in all_alphabets))))
        sys.exit(1)

    all_mods, mod_long_names, mod_fns = {}, {}, {}
    for in_fn, file_alphabet in zip(in_fns, all_alphabets):
        for mod_base in file_alphabet.mod_bases:
            can_base = mod_base.translate(file_alphabet.translation_table)
            mod_long_name = file_alphabet.mod_name_conv[mod_base]
            if mod_base in all_mods:
                # if this mod base has been seen assert that all other
                # attributes agree
                if all_mods[mod_base] != (can_base, mod_long_name):
                    sys.stderr.write((
                        'Incompatible modified bases encountered:\n\t' +
                        '{}={} (alt to {}) from {}\n\t' +
                        '{}={} (alt to {}) from {}\n').format(
                            mod_base, mod_long_name, can_base, in_fn,
                            mod_base, all_mods[mod_base][1],
                            all_mods[mod_base][0], mod_fns[mod_base]))
                    sys.exit(1)
            else:
                # if the mod_base has not been seen before, the long name must
                # also be unique
                if mod_long_name in mod_long_names:
                    sys.stderr.write((
                        'Incompatible modified bases encountered:\n\t' +
                        '{}={} (alt to {}) from {}\n\t' +
                        '{}={} (alt to {}) from {}\n').format(
                            mod_base, mod_long_name, can_base, in_fn,
                            mod_long_names[mod_long_name], mod_long_name,
                            all_mods[mod_long_names[mod_long_name]][0],
                            mod_fns[mod_long_names[mod_long_name]]))
                    sys.exit(1)
                all_mods[mod_base] = (can_base, mod_long_name)
                mod_long_names[mod_long_name] = mod_base
                mod_fns[mod_base] = in_fn

    all_mods = [(mod_nase, can_b, mln)
                for mod_nase, (can_b, mln) in all_mods.items()]
    merge_alphabet = can_bases + ''.join(list(zip(*all_mods))[0])
    merge_collapse_alphabet = can_bases + ''.join(list(zip(*all_mods))[1])
    merge_mod_long_names = list(zip(*all_mods))[2]
    return alphabet.AlphabetInfo(
        merge_alphabet, merge_collapse_alphabet, merge_mod_long_names,
        do_reorder=True)


def assert_all_alphabets_equal(in_fns):
    """ Check that all alphabets are the same in order to perform simple merge.

    Return the alphabet_info object common to all files.

    Also check file versions so this this doesn't short circuit a longer run.
    """
    with MAPPED_SIGNAL_READER(in_fns[0]) as hin:
        #  Copy alphabet and modification information from first file
        merge_alphabet_info = hin.get_alphabet_information()
        check_version(hin, in_fns[0])

    for in_fn in in_fns[1:]:
        with MAPPED_SIGNAL_READER(in_fn) as hin:
            file_alph_info = hin.get_alphabet_information()
            check_version(hin, in_fn)
        # assert that each alphabet equals the first one
        if not merge_alphabet_info.equals(file_alph_info):
            sys.stderr.write(
                'Alphabet info in {} differs from that in {}\n'.format(
                    in_fn, in_fns[0]))
            sys.exit(1)

    return merge_alphabet_info


def create_alphabet_conversion(hin, merge_alphabet_info):
    file_alphabet_info = hin.get_alphabet_information()
    file_alphabet_conv = np.zeros(file_alphabet_info.nbase, dtype=np.int16) - 1
    for file_base_code, file_base in enumerate(file_alphabet_info.alphabet):
        file_alphabet_conv[file_base_code] = \
            merge_alphabet_info.alphabet.index(file_base)
    return file_alphabet_conv


def convert_reference(read, file_alphabet_conv):
    read.Reference = file_alphabet_conv[read.Reference]
    return read


def add_file_reads(
        hin, hout, input_fn, allow_mod_merge, merge_alphabet_info, input_limit,
        reads_written):
    if allow_mod_merge:
        # create integer alphabet conversion array
        file_alphabet_conv = create_alphabet_conversion(
            hin, merge_alphabet_info)

    start_reads_written = len(reads_written)
    read_ids = hin.get_read_ids()
    if input_limit is not None:
        np.random.shuffle(read_ids)
    for read_id in read_ids:
        if read_id in reads_written:
            sys.stderr.write((
                "* Read {} already present: not copying from " +
                "{}.\n").format(read_id, input_fn))
            continue

        read = hin.get_read(read_id)
        if allow_mod_merge:
            read = convert_reference(read, file_alphabet_conv)
        hout.write_read(read.get_read_dictionary())
        reads_written.add(read_id)
        # check if reads limit has been achieved
        if input_limit is not None and \
           len(reads_written) - start_reads_written >= input_limit:
            break
    sys.stderr.write("Copied {} reads from {}.\n".format(
        len(reads_written) - start_reads_written, input_fn))

    return reads_written


def main():
    args = get_parser().parse_args()
    # convert input limits to integers or None since we can't use argparse
    # types and action='append'
    input_fns, input_limits = [], []
    for input_fn, input_limit in args.input:
        input_fns.append(input_fn)
        input_limits.append(none_or_int(input_limit))

    if args.allow_mod_merge:
        merge_alphabet_info = validate_and_merge_alphabets(input_fns)
        sys.stderr.write('Merged alphabet contains: {}\n'.format(
            str(merge_alphabet_info)))
    else:
        merge_alphabet_info = assert_all_alphabets_equal(input_fns)

    if args.seed is not None:
        np.random.seed(args.seed)
    reads_written = set()
    sys.stderr.write("Writing reads to {}\n".format(args.output))
    with MAPPED_SIGNAL_WRITER(args.output, merge_alphabet_info) as hout:
        for input_fn, input_limit in zip(input_fns, input_limits):
            with MAPPED_SIGNAL_READER(input_fn,
                                      load_in_mem=args.load_in_mem) as hin:
                reads_written = add_file_reads(
                    hin, hout, input_fn, args.allow_mod_merge,
                    merge_alphabet_info, input_limit, reads_written)
    sys.stderr.write("Copied {} reads in total.\n".format(len(reads_written)))


if __name__ == '__main__':
    main()
