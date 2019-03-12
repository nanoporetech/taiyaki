#!/usr/bin/env python3
import argparse
import numpy as np

from taiyaki.cmdargs import (AutoBool, FileExists, Positive)
from taiyaki.fileio import readtsv
from taiyaki.helpers import fasta_file_to_dict

parser = argparse.ArgumentParser()
parser.add_argument('--refbackground', default=False, action=AutoBool,
                    help='Get background from references')
parser.add_argument('--down', metavar='bases', type=Positive(int),
                    default=15, help='number of bases down stream')
parser.add_argument('--up', metavar='bases', type=Positive(int),
                    default=15, help='number of bases up stream')
parser.add_argument('references', action=FileExists,
                    help='Fasta file containing references')
parser.add_argument('coordinates', action=FileExists,
                    help='coordinates file')

bases = {b: i for i, b in enumerate('ACGT')}

if __name__ == '__main__':
    args = parser.parse_args()
    args.up += 1

    refdict = fasta_file_to_dict(args.references)
    coordinates = readtsv(args.coordinates)

    background_counts = np.zeros(len(bases), dtype=float)
    if args.refbackground:
        for ref in refdict.values():
            refstr = ref.decode('ascii')
            background_counts += [refstr.count(b) for b in bases.keys()]

    frags = []
    for coord in coordinates:
        readname, pos = coord['filename'], coord['pos']
        readname = readname.decode('ascii')
        if pos < args.down:
            continue
        if not readname in refdict:
            continue
        ref = refdict[readname]
        if pos + args.up > len(ref):
            continue

        frag = ref[pos - args.down : pos + args.up].decode('ascii')
        states = [bases[b] for b in frag]
        frags.append([np.array(states)])

    if len(frags) == 0:
        print("No reads")

    frag_array = np.concatenate(frags).transpose()
    count_array = []

    for pos_array in frag_array:
        counts = np.bincount(pos_array)
        count_array.append([counts])
        if not args.refbackground:
            background_counts += counts

    background_counts /= sum(background_counts)

    position_counts = np.concatenate(count_array) / len(frags)
    relative_abundence = position_counts / background_counts

    for pos, logodds in zip(range(-args.down, args.up), np.log(relative_abundence)):
        print(pos, logodds)
