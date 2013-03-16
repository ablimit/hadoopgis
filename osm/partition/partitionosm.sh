#! /bin/bash

if [ ! $# == 2 ]; then
    echo "Usage: $0 [RTree index ] [result file]"
    exit
fi

loc=/data2/ablimit/Data/spatialdata

make -f Makefile

echo "started partitioning"
./parter < $1 | ./mergetreepartition.py  | sort -nk1 > $2

echo "done."

