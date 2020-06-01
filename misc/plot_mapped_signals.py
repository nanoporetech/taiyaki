#!/usr/bin/env python3
import argparse
import matplotlib as mpl
mpl.use('Agg')  # So we don't need an x server
import matplotlib.pyplot as plt
import numpy as np
import sys
from taiyaki.cmdargs import Positive
from taiyaki import mapped_signal_files

parser = argparse.ArgumentParser(
    description='Plot graphs of reference-to-signal maps from mapped signal files. Also dump one-line summary of each read to stdout',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('mapped_read_files',  nargs='+',
                    help='Inputs: one or more mapped read files')

parser.add_argument('--output', help='Output PNG filename. Default: only output per-read summaries.')
parser.add_argument('--maxlegendsize', type=Positive(int), default=10,
                    help='Maximum number of reads to list in the legend.')


parser.add_argument('--nreads', type=Positive(int), default=10,
                    help='Max number of reads to read from each file. Not used if read_ids are given')
parser.add_argument('--read_ids',  nargs='+', default=[],
                    help='One or more read_ids. If not present, plots the first NREADS in each file')

parser.add_argument('--xmin', default = None, type = float,
                    help='Minimum x for plot')
parser.add_argument('--xmax', default = None, type = float,
                    help='Maximum x for plot')
parser.add_argument('--ymin', default = None, type = float,
                    help='Minimum x for plot')
parser.add_argument('--ymax', default = None, type = float,
                    help='Maximum x for plot')
parser.add_argument('--line_transparency', type = float, default = 1.0,
                    help='Transparency value for lines. Default: %(default)f')
parser.add_argument('--zero_signal_start', action = 'store_true',
                    help='Start signal locations at zero. Default: start at ' +
                    'assigned position within entire read.')
parser.add_argument('--quiet', action='store_true',
                    help='Do not display status messages.')


def main():
    args = parser.parse_args()
    if args.output is not None:
        plt.figure(figsize=(12, 10))
    reads_sofar = 0
    for nfile, mapped_read_file in enumerate(args.mapped_read_files):
        with mapped_signal_files.HDF5Reader(mapped_read_file) as h5:
            all_read_ids = h5.get_read_ids()
            if len(args.read_ids) > 0:
                read_ids = args.read_ids
            else:
                read_ids = all_read_ids[:args.nreads]
                if not args.quiet:
                    sys.stderr.write(
                        "Reading first {} read ids in file {}\n".format(
                            args.nreads, mapped_read_file))
            for nread, read_id in enumerate(read_ids):
                r = h5.get_read(read_id)
                mapping = r.Ref_to_signal
                f = mapping >= 0
                if sum(f) == 0:
                    continue
                if args.zero_signal_start:
                    mapping[f] -= mapping[f][0]
                maplen = len(mapping)
                read_info_text = (
                    'file {} read {}:{} reflen:{}, daclen:{}').format(
                        nfile, nread, read_id, maplen - 1, len(r.Dacs))
                if not args.quiet:
                    sys.stdout.write(read_info_text + '\n')

                if args.output is not None:
                    label = (read_info_text
                             if reads_sofar < args.maxlegendsize
                             else None)
                    x, y = np.arange(maplen)[f], mapping[f]
                    if args.xmin is not None:
                        xf = x >= args.xmin
                        x, y = x[xf], y[xf]
                    if args.xmax is not None:
                        xf = x <= args.xmax
                        x, y = x[xf], y[xf]
                    if args.ymin is not None:
                        yf = y >= args.ymin
                        x, y = x[yf], y[yf]
                    if args.ymax is not None:
                        yf = y <= args.ymax
                        x, y = x[yf], y[yf]
                    plt.plot(x, y, label=label,
                             linestyle = 'dashed' if nfile == 1 else 'solid',
                             alpha=args.line_transparency)
                reads_sofar += 1

    if args.output is not None:
        plt.grid()
        plt.xlabel('Reference location')
        plt.ylabel('Signal location')
        plt.legend(loc='upper left', framealpha=0.3)
        plt.tight_layout()
        if not args.quiet:
            sys.stderr.write("Saving plot to {}\n".format(args.output))
        plt.savefig(args.output)



if __name__=="__main__":
    main()
