#! /bin/bash

if [ $# -lt 6 ]; then
    echo "Usage: $0 [-Xss4m -Xmx3500M] [number of buckets] [input path] [output path] [I | II] [Grid Size]"
    exit 0
fi

jvm=$1
buckets=$2
inPath=$3
outPath=$4
histotype=$5
grid=$6
show=$7

#-Xss4m -Xmx3500M  
java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.MinSkew ${buckets} ${inPath} ${outPath} ${histotype} ${grid} ${show}

