#! /bin/bash

if [ $# -lt 4 ] ;
then
    echo "Usage: $0 [-Xss4m -Xmx3500M] [number of buckets] [input path] [output path] [show]"
    exit 0
fi

# echo "number of param $#"

jvm=$1
buckets=$2
inPath=$3
outPath=$4
show=$5

java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.RTree ${buckets} ${inPath} ${outPath} ${show}

