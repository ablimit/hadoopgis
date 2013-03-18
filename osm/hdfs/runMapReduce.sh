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
for i in 3 4 5 6 7 8 9 10
do
    reducecount=`expr ${i} \\* 10`
    # sudo -u hdfs hdfs dfs -rm -r ${hdfsoutdir}/x${i}
    sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.child.java.opts=-Xmx2048M -mapper "replicator ${i}" -file replicator ${hdfsinputdir} -output ${hdfsoutdir}/x${i} -numReduceTasks ${reducecount} -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="osm_duplication"  -jobconf mapred.task.timeout=36000000


    # -reducer org.apache.hadoop.mapred.lib.IndentityReducer 
    END=$(date +%s)
    DIFF=$(( $END - $START ))
    echo "Factor_$i,${DIFF}" >> dup.log

    # sudo -u hdfs hdfs dfs -copyToLocal /user/aaji/joinout ${OUTDIR}/mjoin_${1}_${reducecount}
done

