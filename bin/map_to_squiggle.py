#!/usr/bin/env python3
import argparse

from taiyaki import fast5utils, helpers, squiggle_match
from taiyaki.cmdargs import (FileExists, Maybe, NonNegative, proportion)
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.iterators import imap_mp


parser = argparse.ArgumentParser(
    description='Map sequence to current trace using squiggle predictor model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


add_common_command_args(parser, "limit jobs output recursive version".split())

parser.add_argument('--back_prob', default=1e-15, metavar='probability',
                    type=proportion, help='Probability of backwards move')
parser.add_argument('--input_strand_list', default=None, action=FileExists,
                    help='Strand summary file containing subset')
parser.add_argument('--localpen', default=None, type=Maybe(NonNegative(float)),
                    help='Penalty for staying in start and end states, or None to disable them')
parser.add_argument('--minscore', default=None, type=Maybe(NonNegative(float)),
                    help='Minimum score for matching')
parser.add_argument('--trim', default=(200, 10), nargs=2, type=NonNegative(int),
                    metavar=('beginning', 'end'), help='Number of samples to trim off start and end')
parser.add_argument('model', action=FileExists, help='Model file')
parser.add_argument('references', action=FileExists, help='Fasta file')
parser.add_argument('read_dir', action=FileExists,
                    help='Directory for fast5 reads')


def main():
    args = parser.parse_args()

    worker_kwarg_names = ['back_prob', 'localpen', 'minscore', 'trim']

    model = helpers.load_model(args.model)

    fast5_reads = fast5utils.iterate_fast5_reads(args.read_dir,
                                                 limit=args.limit,
                                                 strand_list=args.input_strand_list,
                                                 recursive=args.recursive)

    with helpers.open_file_or_stdout(args.output) as fh:
        for res in imap_mp(squiggle_match.worker, fast5_reads, threads=args.jobs,
                           fix_kwargs=helpers.get_kwargs(
                               args, worker_kwarg_names),
                           unordered=True, init=squiggle_match.init_worker,
                           initargs=[model, args.references]):
            if res is None:
                continue
            read_id, sig, score, path, squiggle, bases = res
            bases = bases.decode('ascii')
            fh.write('#{} {}\n'.format(read_id, score))
            for i, (s, p) in enumerate(zip(sig, path)):
                fh.write('{}\t{}\t{}\t{}\t{}\t{}\t{}i\n'.format(read_id, i, s, p,
                                                                bases[p],
                                                                squiggle[p, 0],
                                                                squiggle[p, 1],
                                                                squiggle[p, 2]))


if __name__ == '__main__':
    main()
