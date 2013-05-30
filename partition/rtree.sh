#! /bin/bash


buckets=$1
inPath=$2
outPath=$3
show=$4

java -Xss4m -Xmx80000M -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.RTree $1 $2 $3 $4

