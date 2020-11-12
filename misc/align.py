#!/usr/bin/env python3
import argparse
import os
import pysam
import subprocess
import sys
import traceback

from taiyaki.cmdargs import AutoBool, proportion
from assess_alignment import (
    main as assess_main, get_parser as assess_get_parser)


def get_parser():
    """Get argparser object.

    Returns:
        :argparse:`ArgumentParser` : the argparser object
    """
    parser = argparse.ArgumentParser(
        description='Align reads to reference. Use assess_alignment.py to ' +
        'obtain accuracy metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO: add several named commonly used values for bwa_mem_args
    parser.add_argument(
        '--bwa_mem_args', metavar='args',
        default='-k14 -W20 -r10 -t 16 -A 1 -B 2 -O 2 -E 1',
        help="Command line arguments to pass to bwa mem")

    assess_grp = parser.add_argument_group('Alignment Assessment Arguments')
    assess_grp.add_argument(
        '--coverage', metavar='proportion', default=0.6,
        type=proportion, help='Minimum coverage')
    assess_grp.add_argument(
        '--data_name', default=None,
        help="Data name. If not set file name is used.")
    assess_grp.add_argument(
        '--figure_format', default="png", help="Figure file format.")
    assess_grp.add_argument(
        '--fill', default=True, action=AutoBool,
        help='Fill basecall quality histogram with color')
    assess_grp.add_argument(
        '--show_median', default=False, action=AutoBool,
        help='Show median in a histogram plot')
    assess_grp.add_argument(
        '--reference', default=None,
        help="Reference sequence to align against")

    parser.add_argument(
        'files', metavar='input', nargs='+',
        help="One or more files containing query sequences")

    return parser


STRAND = {0: '+', 16: '-'}

QUANTILES = [5, 25, 50, 75, 95]


def call_bwa_mem(fin, fout, genome, clargs=''):
    """Call bwa aligner using the subprocess module.

    Args:
        fin (str): input sequence filename
        fout (str): filename for the output sam file
        genome (str): path to reference to align against
        clargs (str): optional cmd line arguments to pass to bwa as a string

    Returns:
        str: stdout of bwa command

    Raises:
        :subprocess:`CalledProcessError`: subprocess err. message from bwa call
    """
    command_line = "bwa mem {} {} {} > {}".format(clargs, genome, fin, fout)
    try:
        output = subprocess.check_output(command_line,
                                         stderr=subprocess.STDOUT,
                                         shell=True,
                                         universal_newlines=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            "Error calling bwa, exit code {}\n".format(e.returncode))
        sys.stderr.write(e.output + "\n")
        raise
    return output


def main():
    args = get_parser().parse_args()

    exit_code = 0
    for fn in args.files:
        try:
            pysam.AlignmentFile(fn, 'r')
            align_fn = fn
        except ValueError:
            sys.stdout.write(
                'Input file does not appear to be a SAM/BAM file. Alignment ' +
                'will be performed.\n')
            align_fn = '{}.sam'.format(os.path.splitext(fn)[0])

            # align sequences to reference
            sys.stdout.write("Aligning {}...\n".format(fn))
            try:
                bwa_output = call_bwa_mem(
                    fn, align_fn, args.reference, args.bwa_mem_args)
                sys.stdout.write(bwa_output)
            except Exception:
                sys.stderr.write(
                    "{}: something went wrong, skipping\n\n".format(fn))
                sys.stderr.write(
                    "Traceback:\n\n{}\n\n".format(traceback.format_exc()))
                exit_code = 1

        assess_args = [
            align_fn, '--coverage', str(args.coverage), '--data_name',
            str(args.data_name), '--figure_format', str(args.figure_format)]
        if args.show_median:
            assess_args.append('--show_median')
        assess_main(assess_get_parser().parse_args(assess_args))

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
