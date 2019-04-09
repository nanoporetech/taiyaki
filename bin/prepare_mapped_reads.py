#!/usr/bin/env python
import argparse
from taiyaki.iterators import imap_mp
import os
import sys
from taiyaki.cmdargs import FileExists
import taiyaki.common_cmdargs as common_cmdargs
from taiyaki import fast5utils, helpers, prepare_mapping_funcs, variables


program_description = "Prepare data for model training and save to hdf5 file by remapping with flip-flop model"
parser = argparse.ArgumentParser(description=program_description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


common_cmdargs.add_common_command_args(parser, 'device input_folder input_strand_list jobs limit overwrite recursive version'.split())
default_alphabet_str = variables.DEFAULT_ALPHABET.decode("utf-8")
parser.add_argument('--alphabet', default=default_alphabet_str,
                    help='Alphabet for basecalling. Defaults to ' + default_alphabet_str)
parser.add_argument('--collapse_alphabet', default=default_alphabet_str,
                    help='Collapsed alphabet for basecalling. Defaults to ' + default_alphabet_str)
parser.add_argument('input_per_read_params', action=FileExists,
                    help='Input per read parameter .tsv file')
parser.add_argument('output', help='Output HDF5 file')
parser.add_argument('model', action=FileExists, help='Taiyaki model file')
parser.add_argument('references', action=FileExists,
                    help='Single fasta file containing references for each read')


def main():
    """Main function to process mapping for each read using functions in prepare_mapping_funcs"""
    args = parser.parse_args()
    print("Running prepare_mapping using flip-flop remapping")

    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)

    # Make an iterator that yields all the reads we're interested in.
    fast5_reads = fast5utils.iterate_fast5_reads(
        args.input_folder, limit=args.limit, strand_list=args.input_strand_list,
        recursive=args.recursive)

    # Set up arguments (kwargs) for the worker function for each read
    kwargs = helpers.get_kwargs(args, ['alphabet', 'collapse_alphabet', 'device'])
    kwargs['per_read_params_dict'] = prepare_mapping_funcs.get_per_read_params_dict_from_tsv(args.input_per_read_params)
    kwargs['references'] = helpers.fasta_file_to_dict(args.references)
    kwargs['model'] = helpers.load_model(args.model)
    workerFunction = prepare_mapping_funcs.oneread_remap  # remaps a single read using flip-flip network

    results = imap_mp(workerFunction, fast5_reads, threads=args.jobs,
                      fix_kwargs=kwargs, unordered=True)

    # results is an iterable of dicts
    # each dict is a set of return values from a single read
    prepare_mapping_funcs.generate_output_from_results(results, args)


if __name__ == '__main__':
    main()
