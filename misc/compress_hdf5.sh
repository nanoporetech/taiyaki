#!/usr/bin/env bash

if [ "$1" == "" ]
then
	echo "Compress data sets within a HDF5 file"
	echo "Usage: compress_hdf5.sh file.hdf5"
	exit 1
fi
INFILE=$1

TMPFILE=`mktemp -p .`
h5repack -f SHUF -f GZIP=1 ${INFILE} ${TMPFILE} && mv ${TMPFILE} ${INFILE}
