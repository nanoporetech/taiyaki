#!/usr/bin/env python3
# Combine mapped-read files in HDF5 format into a single file

import argparse
import numpy as np
import sys
from taiyaki import alphabet, mapped_signal_files

parser = argparse.ArgumentParser(
    description='Combine HDF5 mapped-signal files into a single file. Checks ' +
    'that alphabets are compatible.')
parser.add_argument('output', help='Output filename')
parser.add_argument('input', nargs='+', help='One or more input files')
parser.add_argument(
    '--limit', type=int, default=None,
    help='Max number of reads to include in total (default no max)')
parser.add_argument(
    '--per_file_limit', type=int, default=None,
    help='Max number of reads to include from a single file (default no max)')
parser.add_argument(
    '--allow_mod_merge', action='store_true',
    help='Allow merging of data sets with different modified bases. While ' +
    'alphabets may differ, note that incompatible alphabets are not allowed ' +
    '(e.g. same single letter code corresponding to different canonical base).')


# To convert to any new mapped read format (e.g. mapped_signal_files.SQL)
# we should be able to just change MAPPED_SIGNAL_WRITER amd READER to equal the
# new classes.
MAPPED_SIGNAL_WRITER = mapped_signal_files.HDF5Writer
MAPPED_SIGNAL_READER = mapped_signal_files.HDF5Reader


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

    all_mods = [(mod_nase, can_base, mod_long_name)
                for mod_nase, (can_base, mod_long_name) in all_mods.items()]
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
        file_alphabet_conv[file_base_code] = merge_alphabet_info.alphabet.index(
            file_base)
    return file_alphabet_conv

def convert_reference(read, file_alphabet_conv):
    read.Reference = file_alphabet_conv[read.Reference]
    return read

def add_file_reads(
        hin, hout, infile, allow_mod_merge, merge_alphabet_info,
        global_limit, per_file_limit, reads_written):
    file_num_reads_added = 0
    if allow_mod_merge:
        # create integer alphabet conversion array
        file_alphabet_conv = create_alphabet_conversion(
            hin, merge_alphabet_info)

    for read_id in hin.get_read_ids():
        if read_id in reads_written:
            sys.stderr.write((
                "* Read {} already present: not copying from " +
                "{}.\n").format(read_id, infile))
            continue

        read = hin.get_read(read_id)
        if allow_mod_merge:
            read = convert_reference(read, file_alphabet_conv)
        hout.write_read(read.get_read_dictionary())
        reads_written.add(read_id)
        file_num_reads_added += 1
        # check if either global or per-file reads limit has been achieved
        if ((global_limit is not None and len(reads_written) >= global_limit) or
            (per_file_limit is not None and
             file_num_reads_added >= per_file_limit)):
            break

    return file_num_reads_added, reads_written

def main():
    args = parser.parse_args()
    if args.allow_mod_merge:
        merge_alphabet_info = validate_and_merge_alphabets(args.input)
        sys.stderr.write('Merged alphabet contains: {}\n'.format(
            str(merge_alphabet_info)))
    else:
        merge_alphabet_info = assert_all_alphabets_equal(args.input)

    reads_written = set()
    sys.stderr.write("Writing reads to {}\n".format(args.output))
    with  MAPPED_SIGNAL_WRITER(args.output, merge_alphabet_info) as hout:
        for infile in args.input:
            with MAPPED_SIGNAL_READER(infile) as hin:
                file_num_reads_added, reads_written = add_file_reads(
                    hin, hout, infile, args.allow_mod_merge,
                    merge_alphabet_info, args.limit, args.per_file_limit,
                    reads_written)
            sys.stderr.write("Copied {} reads from {}.\n".format(
                file_num_reads_added, infile))
            if args.limit is not None and len(reads_written) >= args.limit:
                break
    sys.stderr.write("Copied {} reads in total.\n".format(len(reads_written)))

    return

if __name__ == '__main__':
    main()
