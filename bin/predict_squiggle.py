#!/usr/bin/env python3
import argparse
from Bio import SeqIO
import numpy as np
import torch

from taiyaki import helpers, squiggle_match
from taiyaki.cmdargs import display_version_and_exit, FileExists, Positive
from taiyaki.version import __version__


parser = argparse.ArgumentParser(
    description='Predict squiggle from sequence',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--version', nargs=0, action=display_version_and_exit,
                    metavar=__version__, help='Display version information')
parser.add_argument('model', action=FileExists, help='Model file')
parser.add_argument('input', action=FileExists, help='Fasta file')


if __name__ == '__main__':
    args = parser.parse_args()

    predict_squiggle = helpers.load_model(args.model)

    for seq in SeqIO.parse(args.input, 'fasta'):
        seqstr = str(seq.seq).encode('ascii')
        embedded_seq_numpy = np.expand_dims(squiggle_match.embed_sequence(seqstr), axis=1)
        embedded_seq_torch = torch.tensor(embedded_seq_numpy, dtype=torch.float32)

        with torch.no_grad():
            squiggle = np.squeeze(predict_squiggle(embedded_seq_torch).cpu().numpy(), axis=1)

        print('base', 'current', 'sd', 'dwell', sep='\t')
        for base, (mean, logsd, dwell) in zip(seq.seq, squiggle):
            print(base, mean, np.exp(logsd), np.exp(-dwell), sep='\t')
