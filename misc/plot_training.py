#!/usr/bin/env python3
import argparse
import matplotlib as mpl
mpl.use('Agg')  # So we don't need an x server
import matplotlib.pyplot as plt
import numpy as np
import os
from taiyaki.cmdargs import Positive
from taiyaki import fileio
from taiyaki.constants import DOTROWLENGTH

parser = argparse.ArgumentParser(
    description='Plot graphs of training loss',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('output', help='Output png file')
parser.add_argument('input_directories',  nargs='+',
                    help='One or more directories containing files called model.log and batch.log')
parser.add_argument('--mav', default=None,
                    type=int,
                    help='Moving average window applied to batchlog loss.' +
                         'e.g --mav 10 makes loss curves easier to separate visually')
parser.add_argument('--upper_y_limit', default=None,
                    type=Positive(float), help='Upper limit of plot y(loss) axis')
parser.add_argument('--lower_y_limit', default=None,
                    type=Positive(float), help='Lower limit of plot y(loss) axis')
parser.add_argument('--upper_x_limit', default=None,
                    type=Positive(float), help='Upper limit of plot x(iterations) axis')
parser.add_argument('--lower_x_limit', default=None,
                    type=Positive(float), help='Lower limit of plot x(iterations) axis')

def moving_average(a, n=3) :
    """Moving average with square window length n.
    If length of a is less than n, and for elements earlier
    than the nth, average as many points as available."""
    x = np.cumsum(a, dtype=float)
    m=len(x)
    if m>n:
        x[n:] = x[n:] - x[:-n]
        x[n:] = x[n:]/n
    x[:n] = x[:n]/np.arange(1,min(n,m)+1)
    return x


def read_training_log(filepath):
    polkas,train_loss,val_loss,lr = [],[],[],[]
    with open(filepath, "r") as f:
        for line in f:
            if not ('*' in line):
                splitline = line.split()
                try:
                    # This try...except is needed in the case where training stops after
                    # some dots and before the numbers are written to the file
                    polkas.append(int(splitline[1]))
                    train_loss.append(float(splitline[2]))
                    val_loss.append(float(splitline[3]))
                    lr.append(float(line.split('lr=')[1].split()[0]))
                except:
                    break
    return {'t':DOTROWLENGTH*np.array(polkas),
            'training_loss':np.array(train_loss),
            'validation_loss':np.array(val_loss),
            'learning_rate':np.array(lr)}

def read_batch_log(filepath):
    t = fileio.readtsv(filepath)
    return {'t':np.arange(len(t)),
            'training_loss':t['loss'],
            'gradientnorm':t['gradientnorm'],
            'gradientcap':t['gradientcap']}
       
def main():
    args = parser.parse_args()
   
    logdata = {}
    batchdata = {}
    for td in args.input_directories:
        logdata[td] = read_training_log(os.path.join(td,'model.log'))
        batchdata[td] = read_batch_log(os.path.join(td,'batch.log'))
        if args.mav is not None:
            batchdata[td]['training_loss'] = moving_average(batchdata[td]['training_loss'], args.mav)
       
    #Plot validation and training loss
    plt.figure(figsize=(5,4))
    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for td,colour in zip(args.input_directories,colour_cycle):
        label = os.path.basename(td)
        plt.plot(batchdata[td]['t'], batchdata[td]['training_loss'],
                 color=colour, label=label+' (training)')
        if len(logdata[td]['t'])==0:
            print("No log data for {} - perhaps <{} iterations complete?".format(td,DOTROWLENGTH))
            continue
        plt.scatter(logdata[td]['t'],logdata[td]['validation_loss'],
                    marker = '+', color=colour, label=label+' (validation)')
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
    plt.legend(loc='upper right')
    if args.mav is not None:
        plt.title('Moving average window = {} iterations'.format(args.mav))
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()
   
if __name__=="__main__":
    main()
