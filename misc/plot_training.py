#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from taiyaki.cmdargs import Positive
from taiyaki import fileio
from taiyaki.constants import BATCH_LOG_FILENAME, VAL_LOG_FILENAME

if True:
    #  Protect in block to prevent autopep8 refactoring
    import matplotlib
    matplotlib.use('Agg')


def get_parser():
    """Get argparser object.

    Returns:
        :argparse:`ArgumentParser` : the argparser object
    """
    parser = argparse.ArgumentParser(
        description='Plot graphs of training loss',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--mav', default=None, type=int,
        help='Moving average window applied to batchlog loss.' +
        'e.g --mav 10 visually separates loss curves')
    parser.add_argument(
        '--upper_y_limit', default=None, type=Positive(float),
        help='Upper limit of plot y(loss) axis')
    parser.add_argument(
        '--lower_y_limit', default=None, type=Positive(float),
        help='Lower limit of plot y(loss) axis')
    parser.add_argument(
        '--upper_x_limit', default=None, type=Positive(float),
        help='Upper limit of plot x(iterations) axis')
    parser.add_argument(
        '--lower_x_limit', default=None, type=Positive(float),
        help='Lower limit of plot x(iterations) axis')

    parser.add_argument(
        'output', help='Output png file')
    parser.add_argument(
        'input_directories', nargs='+',
        help='One or more directories containing {} and {} files'.format(
            BATCH_LOG_FILENAME, VAL_LOG_FILENAME))

    return parser


def moving_average(a, n=3):
    """ Generate moving average.
    Args:
        a (:np:`ndarray`) : 1D input array
        n (int, optional) : square window length

    Returns:
        :np:`ndarray` : 1D output array

    Note: If length of a is less than n, and for elements earlier than the nth,
        average as many points as available.
    """
    x = np.cumsum(a, dtype=float)
    m = len(x)
    if m > n:
        x[n:] = x[n:] - x[:-n]
        x[n:] = x[n:] / n
    x[:n] = x[:n] / np.arange(1, min(n, m) + 1)
    return x


def main():
    args = get_parser().parse_args()

    batchdata = {}
    valdata = {}
    for td in args.input_directories:
        batchdata[td] = fileio.readtsv(os.path.join(td, BATCH_LOG_FILENAME))
        valdata[td] = fileio.readtsv(os.path.join(td, VAL_LOG_FILENAME))
        if args.mav is not None:
            batchdata[td]['loss'] = moving_average(
                batchdata[td]['loss'], args.mav)

    # Plot validation and training loss
    plt.figure(figsize=(6, 4.8))
    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for td, colour in zip(args.input_directories, colour_cycle):
        label = os.path.basename(os.path.normpath(td))
        plt.plot(batchdata[td]['iter'], batchdata[td]['loss'],
                 color=colour, label=label + ' (training)', alpha=0.5,
                 linewidth=0.5)
        if len(valdata[td]['iter']) == 0:
            print(('No validtion log data for {}. The first validation run ' +
                   'has likely not completed.').format(td))
            continue
        plt.plot(valdata[td]['iter'], valdata[td]['loss'],
                 color=colour, label=label + ' (validation)', linewidth=0.5)

    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    if args.upper_y_limit is not None:
        plt.ylim(top=args.upper_y_limit)
    if args.lower_y_limit is not None:
        plt.ylim(bottom=args.lower_y_limit)
    if args.upper_x_limit is not None:
        plt.xlim(right=args.upper_x_limit)
    if args.lower_x_limit is not None:
        plt.xlim(left=args.lower_x_limit)
    leg = plt.legend(loc='upper right')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)

    if args.mav is not None:
        plt.title('Moving average window = {} iterations'.format(args.mav))
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
