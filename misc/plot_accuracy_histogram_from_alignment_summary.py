#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np

from taiyaki.fileio import readtsv
from taiyaki.cmdargs import FileExists, Positive

if True:
    #  Protect in block to prevent autopep8 refactoring
    import matplotlib
    matplotlib.use('Agg')


def get_parser():
    parser = argparse.ArgumentParser(
        description='Plot an accuracy histogram from a combined read file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'combined_read_file', action=FileExists,
        help='Combined read file to get data from')
    parser.add_argument(
        '--bins', default=100, type=Positive(int),
        help='Number of bins for histogram')
    parser.add_argument(
        '--title', default='', help='Figure title')
    parser.add_argument(
        '--output_name', default='basecaller_histogram.png',
        help='Output file name')

    return parser


def main():
    args = get_parser().parse_args()

    AccVals = readtsv(args.combined_read_file)['alignment_accuracy']

    fig, ax = plt.subplots()

    escaped_title = bytes(args.title, "utf-8").decode("unicode_escape")
    ax.set_title(escaped_title, fontsize=7)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Reads')

    ax.minorticks_on()
    ax.grid(which='major', linestyle=':')
    ax.grid(which='minor', linestyle=':')

    plt.hist(np.array(AccVals[AccVals >= 0]), bins=args.bins)

    plt.tight_layout()

    plt.savefig(args.output_name)


if __name__ == "__main__":
    main()
