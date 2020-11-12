#!/usr/bin/env python3
import argparse
from Bio import SeqIO
import numpy as np
import torch

from taiyaki import helpers, squiggle_match
from taiyaki.cmdargs import FileExists
from taiyaki.common_cmdargs import add_common_command_args


def get_parser():
    """Get argparser object.

    Returns:
        argparse.ArgumentParser : the argparser object
    """
    parser = argparse.ArgumentParser(
        description='Predict squiggle from sequence',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_common_command_args(parser, "output version".split())

    parser.add_argument(
        'model', action=FileExists, help='Model file')
    parser.add_argument(
        'input', action=FileExists, help='Fasta file')

    return parser


def main():
    args = get_parser().parse_args()

    predict_squiggle = helpers.load_model(args.model)

    with helpers.open_file_or_stdout(args.output) as fh:
        for seq in SeqIO.parse(args.input, 'fasta'):
            seqstr = str(seq.seq)
            embedded_seq_numpy = np.expand_dims(
                squiggle_match.embed_sequence(seqstr), axis=1)
            embedded_seq_torch = torch.tensor(
                embedded_seq_numpy, dtype=torch.float32)

            with torch.no_grad():
                squiggle = np.squeeze(predict_squiggle(
                    embedded_seq_torch).cpu().numpy(), axis=1)

            fh.write('base\tcurrent\tsd\tdwell\n')
            for base, (mean, logsd, dwell) in zip(seq.seq, squiggle):
                fh.write('{}\t{}\t{}\t{}\n'.format(
                    base, mean, np.exp(logsd), np.exp(-dwell)))


if __name__ == '__main__':
    main()
