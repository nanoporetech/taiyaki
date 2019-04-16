#!/usr/bin/env python3
import argparse
import matplotlib as mpl
mpl.use('Agg')  # So we don't need an x server
import matplotlib.pyplot as plt
import numpy as np
import sys
from taiyaki.cmdargs import AutoBool, Positive
from taiyaki import mapped_signal_files

parser = argparse.ArgumentParser(
    description='Plot graphs of reference-to-signal maps from mapped signal files.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('output', help='Output: a png file')
parser.add_argument('mapped_read_files',  nargs='+',
                    help='Inputs: one or more mapped read files')
parser.add_argument('--nreads', type=Positive(int), default=10,
                    help='Number of reads to plot. Not used if read_ids are given')
parser.add_argument('--read_ids',  nargs='+', default=[],
                    help='One or more read_ids. If not present, plots the first NREADS in each file')

parser.add_argument('--xmin', default = None, type = float,
                    help='Minimum x for plot')
parser.add_argument('--xmax', default = None, type = float,
                    help='Maximum x for plot')


if __name__=="__main__":
    args = parser.parse_args()
    plt.figure(figsize=(12, 10))
    for nfile, mapped_read_file in enumerate(args.mapped_read_files):
        sys.stderr.write("Opening {}\n".format(mapped_read_file))
        with mapped_signal_files.HDF5(mapped_read_file, "r") as h5:
            all_read_ids = h5.get_read_ids()
            sys.stderr.write("First ten read_ids in file:\n")
            for read_id in all_read_ids[:10]:
                sys.stderr.write("    {}\n".format(read_id))
            if len(args.read_ids) > 0:
                read_ids = args.read_ids
            else:
                read_ids = all_read_ids[:args.nreads]
                sys.stderr.write("Plotting first {} read ids in file\n".format(args.nreads))
            for nread, read_id in enumerate(read_ids):
                sys.stderr.write("Opening read id {}\n".format(read_id))
                r = h5.get_read(read_id)
                mapping = r['Ref_to_signal']
                f = mapping >= 0
                maplen = len(mapping)
                label = 'file '+str(nfile)+' read '+str(nread) + ":" + read_id + " reflen:" + str(maplen - 1) + ", daclen:" + str(len(r['Dacs']))
                x,y = np.arange(maplen)[f], mapping[f]
                if args.xmin is not None:
                    xf = (x >= args.xmin)
                    x,y = x[xf],y[xf]
                if args.xmax is not None:
                    xf = (x <= args.xmax)
                    x,y = x[xf],y[xf]
                plt.plot(x, y, label=label, linestyle = 'dashed' if nfile==1 else 'solid')

    plt.grid()
    plt.xlabel('Reference location')
    plt.ylabel('Signal location')
    if len(read_ids) < 15:
        plt.legend(loc='upper left', framealpha=0.3)
    plt.tight_layout()
    sys.stderr.write("Saving plot to {}\n".format(args.output))
    plt.savefig(args.output)
