#!/usr/bin/env python3
# Check that a HDF5 file c given keys
# Return failure condition if they are not present.

import argparse
import h5py

parser = argparse.ArgumentParser(
    description='Check that given keys exist in an HDF5 file')

parser.add_argument('input', help='HDF5 file')
parser.add_argument("keys", nargs="+", help="Keys to check")


def main():
    args = parser.parse_args()
    with h5py.File(args.input, 'r') as h5:
        for key in args.keys:
            assert key in h5
            print("Key ", key, "present in", args.input)

    print("All keys present")


if __name__ == "__main__":
    main()
