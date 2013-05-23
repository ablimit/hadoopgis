#! /bin/bash

#if [ ! $# < 5 ]; then
#    echo "Usage:  [number of buckets] [input path] [output path] [I | II] [Grid Size]"
#    exit 0
#fi

buckets=$1
inPath=$2
outPath=$3
samplerate=$4
show=$5

java -Xmx2048M -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.RVrkHistSTHist $1 $2 $3 sthist $4 $5

