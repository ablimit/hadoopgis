#! /bin/bash

if [ ! $# == 1 ]; then
    echo "Usage: $0 [root hdfs output dir]"
    exit 0
fi

make -f Makefile

export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce

hdfsoutdir=$1
hdfsinputdir="-input /user/aaji/osm/smalltile"

START=$(date +%s)
for i in 1 2 3 4 5 6 7 8 9 10
do
    sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -mapper "duplicator ${i}" -reducer org.apache.hadoop.mapred.lib.IndentityReducer -file duplicator ${input} -output ${hdfsoutdir}/x{i} -numReduceTasks 8 -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="osm_duplication"  -jobconf mapred.task.timeout=36000000


    END=$(date +%s)
    DIFF=$(( $END - $START ))
    echo "Factor_$i,${DIFF}" >> dup.log

    # sudo -u hdfs hdfs dfs -copyToLocal /user/aaji/joinout ${OUTDIR}/mjoin_${1}_${reducecount}
done

