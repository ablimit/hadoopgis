#! /bin/bash

tempf=/tmp/zahide.gnu

for t in 1 2 3 4 5
do
    for p in quadtree binarysplit
    do
        if [ -e testdata/test${t}.${p}.part.txt ]
        then
            python drawSinglePartition.py pics/test${t}.${p}.png testdata/test${t}.${p}.part.txt testdata/test${t}.obj.txt >${tempf}
            gnuplot ${tempf}
        fi
    done
done

