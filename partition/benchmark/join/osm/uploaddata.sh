#! /bin/bash

#if [ ! $# == 1 ]; then
#    echo "Usage: $0 [root hdfs output dir]"
#    exit 0
#fi

hmkdir="sudo -u hdfs hdfs dfs -mkdir"
hput="sudo -u hdfs hdfs dfs -put"

indir=/scratch/aaji/osm/partition

for method in minskew rkHist rv rtree minskewrefine
do
    ${hmkdir} -p /user/aaji/partition/osm/${method}

    for batch in oc100k oc200k oc300k oc400k oc500k oc50k

    do 
	echo "${method} -- ${batch}"
	zcat  ${indir}/${batch}/osm.${method}.txt.gz | ${hput} - /user/aaji/partition/osm/${method}/osm.${batch}.txt
    done
done
