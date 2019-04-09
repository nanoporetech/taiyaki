#!/usr/bin/env python3
import matplotlib as mpl
mpl.use('Agg')  # So we don't need an x server
import matplotlib.pyplot as plt
import numpy as np
import sys
from taiyaki import fileio

def main():
    print("Plots summary of chunk log.")
    print("Usage:")
    print("plot_chunk_log.py <chunk_log_file> <output_file>")
    if len(sys.argv) < 3:
        print("ERROR: Needs command line arguments!")
    else:
        chunk_log_file = sys.argv[1]
        plotfile = sys.argv[2]
        t = fileio.readtsv(chunk_log_file)

        plt.figure(figsize=(16, 12))

        plt.subplot(2, 2, 1)
        plt.title('Mean dwells of chunks sampled to get filter params')
        f = (t['iteration'] == -1) & (t['status'] == 'pass')
        bases = t['chunk_len_bases'][f]
        samples = t['chunk_len_samples'][f]
        filter_sample_length = len(bases)
        meandwells = samples / (bases + 0.0001)
        plt.hist(meandwells, bins=100, log=True)
        plt.grid()

        # Remove the part that refers to the sampling for filter params
        t = t[filter_sample_length:]

        plt.subplot(2, 2, 2)
        plt.title('Lengths of accepted and rejected chunks')
        status_choices = np.unique(t['status'])
        # Need to do 'pass' first - otherwise it overwhelms everything
        status_choices = list(status_choices[status_choices != 'pass'])
        status_choices = ['pass'] + status_choices
        for status in status_choices:
            filt = (t['status'] == status)
            bases = t['chunk_len_bases'][filt]
            samples = t['chunk_len_samples'][filt]
            print("Status", status, "number of chunks=", len(bases))
            plt.scatter(bases, samples, label=status, s=4)

        plt.grid()
        plt.ylabel('Length in bases')
        plt.xlabel('Length in samples')
        plt.legend(loc='upper left', framealpha=0.3)

        for nplot, scale in enumerate('log linear'.split()):
            plt.subplot(2, 2, nplot + 3, xscale=scale, yscale=scale)
            plt.title('Max and mean dwells')
            status_choices = np.unique(t['status'])
            # Need to do 'pass' first - otherwise it overwhelms everything
            status_choices = list(status_choices[status_choices != 'pass'])
            status_choices = ['pass'] + status_choices
            for status in status_choices:
                filt = (t['status'] == status)
                bases = t['chunk_len_bases'][filt]
                samples = t['chunk_len_samples'][filt]
                count = len(bases)
                meandwells = samples / (bases + 0.0001)
                maxdwells = t['max_dwell'][filt]
                plt.scatter(meandwells, maxdwells, label=status+' ('+str(count)+')', s=4, alpha=0.5)

            plt.grid()
            plt.xlabel('Mean dwell')
            plt.ylabel('Max dwell')
            plt.legend(loc='lower right', framealpha=0.3)

        plt.savefig(plotfile)


if __name__ == '__main__':
    main()
