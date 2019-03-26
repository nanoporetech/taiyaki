#!/usr/bin/env python3
import argparse
import matplotlib as mpl
mpl.use('Agg')  # So we don't need an x server
import matplotlib.pyplot as plt
import numpy as np
import os
from taiyaki.cmdargs import Positive

parser = argparse.ArgumentParser(
    description='Plot graphs of training loss',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('output', help='Output png file')
parser.add_argument('input_directories',  nargs='+',
                    help='One or more directories containing files called model.log')
parser.add_argument('--mav', default=None, 
                    type=int,
                    help='Moving average window applied to loss.' +
                         'e.g --mav 10 makes loss curves easier to separate visually')
parser.add_argument('--upper_y_limit', default=None, 
                    type=Positive(float), help='Upper limit of plot y(loss) axis')
parser.add_argument('--lower_y_limit', default=None, 
                    type=Positive(float), help='Lower limit of plot y(loss) axis')

def ewma(x,timescale):
    """Exponentially weighted moving average of numpy vector (or list)
    with given timescale, so that  averaging goes like exp(-t/timescale).
    Useful to make separate loss plots more visible."""
    a = np.exp(-1.0 / timescale)
    y = np.zeros(len(x), dtype=np.float64)
    y[0] = x[0]
    for t in range(1,len(x)):
        y[t] = (1-a) * x[t] + a * y[t-1]
    return y

if __name__=="__main__":
    args = parser.parse_args()
    plt.figure()
    for training_directory in args.input_directories:
        blocklist = []
        losslist = []
        filepath = training_directory + "/model.log"
        print("Opening", filepath)
        with open(filepath, "r") as f:
            for line in f:
                # The * removes error messges in the log
                if line.startswith('.') and not ('*' in line):
                    splitline = line.split()
                    try:
                        # This try...except only needed in the case where training stops after
                        # some dots and before the numbers are written to the file
                        blocklist.append(int(splitline[1]))
                        losslist.append(float(splitline[2]))
                    except:
                        break
        #The label for the legend is the name of the directory (without its full path)
        if args.mav is not None:
            losslist = ewma(losslist, args.mav)
        plt.plot(blocklist, losslist, label = os.path.basename(training_directory))
    plt.grid()
    plt.xlabel('Iteration blocks (each block = 50 iterations)')
    plt.ylabel('Loss')
    if args.upper_y_limit is not None:
        plt.ylim(top=args.upper_y_limit)
    if args.lower_y_limit is not None:
        plt.ylim(bottom=args.lower_y_limit)
    plt.legend(loc='upper right')
    if args.mav is not None:
        plt.title('Moving average window = {} blocks'.format(args.mav))
    plt.tight_layout()
    print("Saving plot to", args.output)
    plt.savefig(args.output)
