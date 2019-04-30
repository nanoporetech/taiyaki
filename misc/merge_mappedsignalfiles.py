#!/usr/bin/env python3
# Combine mapped-read files in HDF5 format into a single file

import argparse
from taiyaki import alphabet, mapped_signal_files

parser = argparse.ArgumentParser(
    description='Combine HDF5 mapped-signal files into a single file. Checks to make sure alphabets agree.')

parser.add_argument('output', help='Output filename')
parser.add_argument('input', nargs='+', help='One or more input files')

#To convert to any new mapped read format (e.g. mapped_signal_files.SQL)
#we should be able to just change MAPPED_READ_WRITER amd READER to equal the new classes.
MAPPED_READ_WRITER = mapped_signal_files.HDF5Writer
MAPPED_READ_READER = mapped_signal_files.HDF5Reader


def check_version(file_handle, filename):
    """Check to make sure version agrees with the file format version in
    this edition of Taiyaki. If not, throw an exception."""
    if mapped_signal_files._version != file_handle.version:
        raise Exception(("File version of mapped signal file ({}, version {}) does " +
                         "not match this version of Taiyaki (file version {})").format(
                         filename, file_handle.version, mapped_signal_files._version))



def main():
    args = parser.parse_args()
    with MAPPED_READ_READER(args.input[0]) as hin:
        check_version(hin, args.input[0])
        #  Copy alphabet and modification information from first file
        alph_info = alphabet.AlphabetInfo(*hin.get_alphabet_information())
    reads_written = set()
    print("Writing reads to ", args.output)
    with  MAPPED_READ_WRITER(args.output, alph_info) as hout:
        for infile in args.input:
            copied_from_this_file = 0
            with MAPPED_READ_READER(infile) as hin:
                check_version(hin, args.input[0])
                in_alph_info = alphabet.AlphabetInfo(*hin.get_alphabet_information())
                if not alph_info.equals(in_alph_info):
                    raise Exception("Alphabet info in {} differs from that in {}".format(
                                    infile,args.input[0]))
                for read_id in hin.get_read_ids():
                    if read_id in reads_written:
                        print("* Read",read_id,"already present: not copying from",infile)
                    else:
                        readObject = hin.get_read(read_id)
                        readObject['read_id']=read_id
                        hout.write_read(readObject)
                        reads_written.add(read_id)
                        copied_from_this_file += 1
            print("Copied",copied_from_this_file,"reads from",infile)
    print("Copied",len(reads_written),"reads in total")

if __name__ == '__main__':
    main()
