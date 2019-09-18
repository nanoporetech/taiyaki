#!/usr/bin/env python3
# Convert the alphabet attributes contained within a mapped signal file.

import re
import sys
import argparse
from taiyaki import alphabet, mapped_signal_files

def get_parser():
    parser = argparse.ArgumentParser(
        description='Convert (or print) alphabet attributes contained within ' +
        'a mapped signal file. Attributes will be adjusted inplace. Note ' +
        'that association of modified bases to canonical bases cannot be ' +
        'converted with this script.')
    parser.add_argument('input', help='Mapped signal file.')
    parser.add_argument(
        '--print_only', action='store_true',
        help='Only print the alphabet information contained within this file.')
    parser.add_argument(
        '--can_base_convert', nargs=2,
        default=[], action='append',
        help='Current and new values for conversion of single letter ' +
        'canonical base.')
    parser.add_argument(
        '--mod_base_convert', nargs=2,
        default=[], action='append',
        help='Current and new values for conversion of single letter ' +
        'modified base.')
    parser.add_argument(
        '--mod_long_name_convert', nargs=2,
        default=[], action='append',
        help='Current and new values for conversion of modified base ' +
        'long names.')
    return parser

def main():
    args = get_parser().parse_args()
    with mapped_signal_files.HDF5Reader(args.input) as hin:
        alphabet_info = hin.get_alphabet_information()
    sys.stderr.write('File, "{}", currently contains alphabet: {}\n'.format(
        args.Input, str(alphabet_info)))
    if args.print_only:
        sys.exit()

    new_alphabet_bases, new_collapse_bases = {}, {}
    for curr_can_base, new_can_base in args.can_base_convert:
        assert len(curr_can_base) == 1, (
            'Single letter codes must be a single character. Got {}'.format(
            curr_can_base))
        assert len(new_can_base) == 1, (
            'Single letter codes must be a single character. Got {}'.format(
            new_can_base))
        if curr_can_base not in alphabet_info.can_bases_set:
            sys.stderr.write((
                'Specified current canonical base ({}) not found in ' +
                'file.').format(curr_can_base))
            sys.exit(1)
        new_alphabet_bases[
            alphabet_info.alphabet.index(curr_can_base)] = new_can_base
        for m in re.finditer(curr_can_base, alphabet_info.collapse_alphabet):
            new_collapse_bases[m.start()] = new_can_base
    new_collapse_alphabet = None
    if len(new_collapse_bases) > 0:
        new_collapse_alphabet = ''.join(
            new_collapse_bases[idx] if idx in new_collapse_bases else b
            for idx, b in enumerate(alphabet_info.collapse_alphabet))

    for curr_mod_base, new_mod_base in args.mod_base_convert:
        assert len(curr_mod_base) == 1, (
            'Single letter codes must be a single character. Got {}'.format(
            curr_mod_base))
        assert len(new_mod_base) == 1, (
            'Single letter codes must be a single character. Got {}'.format(
            new_mod_base))
        if curr_mod_base not in alphabet_info.mod_bases_set:
            sys.stderr.write((
                'Specified current modified base ({}) not found in ' +
                'file.').format(curr_mod_base))
            sys.exit(1)
        new_alphabet_bases[
            alphabet_info.alphabet.index(curr_mod_base)] = new_mod_base
    new_alphabet = None
    if len(new_alphabet_bases) > 0:
        new_alphabet = ''.join(
            new_alphabet_bases[idx] if idx in new_alphabet_bases else b
            for idx, b in enumerate(alphabet_info.alphabet))

    new_mlns = {}
    for curr_mln, new_mln in args.mod_long_name_convert:
        assert re.search('\n', new_mln) is None, (
            'Modified base long name ({}) includes an invalid  newline ' +
            'character.').format(curr_mln)
        assert curr_mln in alphabet_info.mod_long_names, (
            'Specified current modified base long name ({}) not found in ' +
            'file.').format(curr_mln)
        new_mlns[curr_mln] = new_mln
    new_mod_long_names = None
    if len(new_mlns) > 0:
        new_mod_long_names = [
            new_mlns[curr_mln] if curr_mln in new_mlns else curr_mln
            for curr_mln in alphabet_info.mod_long_names]

    if all(alphabet_attr is None for alphabet_attr in (
            new_collapse_alphabet, new_alphabet, new_mod_long_names)):
        sys.stderr.write('No new alphabet information provided.\n')
        sys.exit(1)
    with h5py.File(args.Input, 'r+') as msf:
        if new_alphabet is not None:
            sys.stderr.write('Converting alphabet from "{}" to "{}".\n'.format(
                alphabet_info.alphabet, new_alphabet))
            msf.attrs['alphabet'] = new_alphabet
        if new_collapse_alphabet is not None:
            sys.stderr.write(
                'Converting collapse alphabet from "{}" to "{}".\n'.format(
                    alphabet_info.collapse_alphabet, new_collapse_alphabet))
            msf.attrs['collapse_alphabet'] = new_collapse_alphabet
        if new_mod_base is not None:
            sys.stderr.write((
                'Converting modified base long names from "{}" to ' +
                '"{}".\n').format(
                    '", "'.join(alphabet_info.mod_long_names),
                    '", "'.join(new_collapse_alphabet)))
            msf.attrs['mod_long_names'] = '\n'.join(new_mod_long_names)

    return

if __name__ == '__main__':
    main()
