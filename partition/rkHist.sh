#! /bin/bash

if [ $# -lt 5 ]; then
    echo "Usage: $0 [-Xss4m -Xmx3500M] [number of buckets] [input path] [output path] [alpha]"
    exit 0
fi

jvm=$1
buckets=$2
inPath=$3
outPath=$4
alpha=$5
show=$6

java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.RVrkHistSTHist ${buckets} ${inPath} ${outPath} rkhist ${alpha} ${show}

