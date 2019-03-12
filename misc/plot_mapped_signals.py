#!/usr/bin/env python3
import argparse
import matplotlib as mpl
mpl.use('Agg')  # So we don't need an x server
import matplotlib.pyplot as plt
import numpy as np
from taiyaki.cmdargs import Positive
from taiyaki import mapped_signal_files

parser = argparse.ArgumentParser(
    description='Plot graphs of training mapped reads.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('mapped_read_file', help='Input: a mapped read file')
parser.add_argument('output', help='Output: a png file')
parser.add_argument('--nreads', type=Positive(int), default=10,
                    help='Number of reads to plot. Not used if read_ids are given')
parser.add_argument('--read_ids',  nargs='+', default=[],
                    help='One or more read_ids. If not present, plots the first NREADS in the file')

if __name__=="__main__":
    args = parser.parse_args()
    print("Opening ", args.mapped_read_file)
    with mapped_signal_files.HDF5(args.mapped_read_file, "r") as h5:
        all_read_ids = h5.get_read_ids()
        print("First ten read_ids in file:")
        for read_id in all_read_ids[:10]:
            print("    ", read_id)
        if len(args.read_ids) > 0:
            read_ids = args.read_ids
        else:
            read_ids = all_read_ids[:args.nreads]
            print("Plotting first ", args.nreads, "read ids in file")
        plt.figure(figsize=(12, 10))
        for nread, read_id in enumerate(read_ids):
            print("Opening read id ",read_id)
            r = h5.get_read(read_id)
            mapping = r['Ref_to_signal']
            f = mapping >= 0
            maplen = len(mapping)
            label = str(nread) + ":" + read_id + " reflen:" + str(maplen - 1) + ", daclen:" + str(len(r['Dacs']))
            plt.plot(np.arange(maplen)[f], mapping[f], label=label)

    plt.grid()
    plt.xlabel('Reference location')
    plt.ylabel('Signal location')
    if len(read_ids) < 15:
        plt.legend(loc='upper left', framealpha=0.3)
    plt.tight_layout()
    print("Saving plot to", args.output)
    plt.savefig(args.output)
