#!/usr/bin/env python3
# Take a directory or a strand list and make N separate files,
# which together list all the strands.

import argparse
import os

from taiyaki.cmdargs import Positive

parser = argparse.ArgumentParser(
    description='Split a strand list into a number of smaller strand lists, or alternatively do the same thing starting with a directory containing fast5s.')
parser.add_argument('--maxlistsize', default=10000, type=Positive(int),
                    help='Maximum size for a strand list')

parser.add_argument('--outputbase', default=10000,
                    help='Strand lists will be saved as <outputbase>_000.txt etc. If outputbase not present then the input will be used as the base name.')


parser.add_argument('input', help='either a strand list file or a directory name')

strandlist_header = "filename"


def main():
    args = parser.parse_args()
    # If we can read strands from it, then it's a strand list
    try:
        strands = []
        with open(args.input, "r") as f:
            for nline, line in enumerate(f):
                cleanedline = line.rstrip()
                if nline < 10:
                    print(cleanedline)
                if cleanedline.endswith('fast5'):  # First line is often 'filename'
                    strands.append(cleanedline)
        print("Read", len(strands), "files from strand list")
    except:
        strands = os.listdir(args.input)
        print("Read", len(strands), "files from directory")
        for fi in strands:
            if not (fi.endswith('fast5')):
                raise Exception("Not all files in directory are fast5 files")

    filebase = args.outputbase
    if filebase is None:
        filebase = args.input
    nfiles = (len(strands) + args.maxlistsize - 1) // args.maxlistsize
    for filenumber in range(nfiles):
        fname = filebase + str(filenumber).zfill(3)
        with open(fname, "w") as f:
            f.write(strandlist_header + "\n")
            startnum = filenumber * args.maxlistsize
            endnum = min(len(strands), (filenumber + 1) * args.maxlistsize)
            for nstrand in range(startnum, endnum):
                f.write(strands[nstrand] + "\n")


if __name__ == '__main__':
    main()
