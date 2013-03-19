#! /bin/bash

if [ ! $# == 2 ]; then
    echo "Usage: $0 [RTree index ] [result file]"
    exit
fi

loc=/data2/ablimit/Data/spatialdata

make -f Makefile

echo "started partitioning"
#./partitioner $1 | ./mergetreepartition.py  | sort -nk1 > $2
./partitioner $1 | sort -nk2 > $2

echo "done."

