#!/usr/bin/env python
import argparse
from taiyaki.iterators import imap_mp
import numpy as np
import os
import sys
from taiyaki.cmdargs import FileExists, Maybe
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki import alphabet, fast5utils, helpers, prepare_mapping_funcs


program_description = "Prepare data for model training and save to hdf5 file by remapping with flip-flop model"
parser = argparse.ArgumentParser(description=program_description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_common_command_args(parser, 'alphabet input_folder input_strand_list jobs limit overwrite recursive version'.split())

parser.add_argument('--max_read_length', metavar='bases', default=None, type=Maybe(int),
                    help='Don\'t attempt remapping for reads longer than this')
parser.add_argument('--mod', nargs=3, metavar=('base', 'canonical', 'name'),
                    default=[], action='append',
                    help='Modified base description')
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

    # Create alphabet and check for consistency
    modified_bases = [elt[0] for elt in args.mod]
    canonical_bases = [elt[1] for elt in args.mod]
    for b in modified_bases:
        assert len(b) == 1, "Modified bases must be a single character, got {}".format(b)
        assert b not in args.alphabet, "Modified base must not be a canonical base, got {}".format(b)
    for b in canonical_bases:
        assert len(b) == 1, "Canonical coding for modified bases must be a single character, got {}".format(b)
        assert b in args.alphabet, "Canonical coding for modified base must be a canonical base, got {}".format(b)
    full_alphabet = args.alphabet + ''.join(modified_bases)
    flat_alphabet = args.alphabet + ''.join(canonical_bases)
    modification_names = [elt[2] for elt in args.mod]

    alphabet_info = alphabet.AlphabetInfo(full_alphabet, flat_alphabet,
                                          modification_names, do_reorder=True)

    print("Converting references to labels using {}".format(str(alphabet_info)))

    # Make an iterator that yields all the reads we're interested in.
    fast5_reads = fast5utils.iterate_fast5_reads(
        args.input_folder, limit=args.limit, strand_list=args.input_strand_list,
        recursive=args.recursive)

    # Set up arguments (kwargs) for the worker function for each read
    kwargs = {}
    kwargs['per_read_params_dict'] = prepare_mapping_funcs.get_per_read_params_dict_from_tsv(
        args.input_per_read_params)
    kwargs['model'] = helpers.load_model(args.model)
    kwargs['alphabet_info'] = alphabet_info
    kwargs['max_read_length'] = args.max_read_length

    # remaps a single read using flip-flip network
    workerFunction = prepare_mapping_funcs.oneread_remap

    def iter_jobs():
        references = helpers.fasta_file_to_dict(args.references, alphabet=full_alphabet)
        for fn, read_id in fast5_reads:
            yield fn, read_id, references.get(read_id, None)

    if args.limit is not None:
        chunksize = args.limit // (2 * args.jobs)
        chunksize = int(np.clip(chunksize, 1, 50))
    else:
        chunksize = 50

    results = imap_mp(workerFunction, iter_jobs(), threads=args.jobs,
                      fix_kwargs=kwargs, unordered=True, chunksize=chunksize)

    # results is an iterable of dicts
    # each dict is a set of return values from a single read
    prepare_mapping_funcs.generate_output_from_results(
        results, args.output, alphabet_info)


if __name__ == '__main__':
    main()
