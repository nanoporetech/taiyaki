#!/usr/bin/env python3
# Combine mapped-read files in HDF5 format into a single file

import argparse
from taiyaki import mapped_signal_files
from taiyaki.cmdargs import Positive
from taiyaki.common_cmdargs import add_common_command_args

parser = argparse.ArgumentParser(
    description='Combine HDF5 mapped-read files into a single file')

add_common_command_args(['version'])
parser.add_argument('output', help='Output filename')
parser.add_argument('input', nargs='+', help='One or more input files')

#To convert to any new mapped read format (e.g. mapped_signal_files.SQL)
#we should be able to just change MAPPED_READ_CLASS to equal the new class.
MAPPED_READ_CLASS = mapped_signal_files.HDF5


def main():
    args = parser.parse_args()

    with MAPPED_READ_CLASS(args.inputs[0], "r") as hin:
        #  Copy alphabet and modification information from first file
        in_alphabet, in_collapse_alphabet, in_mod_long_names \
            = hin.get_alphabet_information()
        args.alphabet = in_alphabet
        args.collapse_alphabet = in_collapse_alphabet
        args.mod_long_names = in_mod_long_names

    reads_written = set()
    print("Writing reads to ", args.output)
    with  MAPPED_READ_CLASS(args.output, "w") as hout:
        hout.write_version_number(args.version)
        hout.write_alphabet_information(
            args.alphabet, args.collapse_alphabet, args.mod_long_names)
        # TODO include logic to merge data sets with different alphabets
        # requires specifying exactly how this merge would occur
        for infile in args.inputs:
            copied_from_this_file = 0
            with MAPPED_READ_CLASS(infile, "r") as hin:
                in_version = hin.get_version_number()
                if in_version != args.version:
                    raise Exception("Version number of files should be {} but version number of {} is {}".format(args.version, infile, in_version))
                in_alphabet, in_collapse_alphabet, in_mod_long_names \
                    = hin.get_alphabet_information()
                if in_alphabet != args.alphabet:
                    raise Exception(
                        "Alphabet should be {} but alphabet in {} is {}".format(
                            args.alphabet, infile, in_alphabet))
                if in_collapse_alphabet != args.collapse_alphabet:
                    raise Exception(
                        ("Collapes alphabet should be {} but collapse " +
                         "alphabet in {} is {}").format(
                            args.collapse_alphabet, infile,
                             in_collapse_alphabet))
                if (len(in_mod_long_names) != len(args.mod_long_names) or
                    any(in_mod_long_names[mod_i] != args.mod_long_names[mod_i]
                        for mod_i range(len(in_mod_long_names)))):
                    raise Exception(
                        ("Modified base long names should be {} but modified " +
                         "base long names in {} is {}").format(
                             args.mod_long_names, infile, in_mod_long_names))
                for read_id in hin.get_read_ids():
                    if read_id in reads_written:
                        print("* Read",read_id,"already present: not copying from",infile)
                    else:
                        hout.write_read(read_id, hin.get_read(read_id))
                        reads_written.add(read_id)
                        copied_from_this_file += 1
            print("Copied",copied_from_this_file,"reads from",infile)
    print("Copied",len(reads_written),"reads in total")


if __name__ == '__main__':
    main()
