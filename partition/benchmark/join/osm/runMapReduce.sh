#! /bin/bash

#if [ ! $# == 1 ]; then
#    echo "Usage: $0 [root hdfs output dir]"
#    exit 0
#fi
make -f Makefile

export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce

log=osm.join.log
outpath=/scratch/aaji/osm

# factor=2
hdfsoutdir=/user/aaji/osmjoinout

date >> ${log}
echo "" >> ${log}

# for round in 1
for size in 50000 100000 200000 300000 400000 500000
do
    mark=${size%000}
    batch=oc${mark}k
    outdir=${outpath}/partition/oc${mark}k

    for method in minskew rv rtree minskewrefine
    do

	# if some file exists ; so this has to be done on node35
	if [ -e ${outdir}/osm.${method}.txt.gz ] ; then

	    optinput="-input /user/aaji/partition/osm/${method}/osm.${batch}.txt"

	    for reducecount in 180 140 40 
	    do
		sudo -u hdfs hdfs dfs -rm -r ${hdfsoutdir}
		sudo -u hdfs hdfs dfs -rm -r /user/aaji/dedupout

		START=$(date +%s)

		sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.child.java.opts=-Xmx4096M -mapper mapper -reducer reducer -file mapper -file reducer ${optinput} -output ${hdfsoutdir} -numReduceTasks ${reducecount} -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="oj_${batch}_${method}_${reducecount}"  -jobconf mapred.task.timeout=36000000

		END=$(date +%s)
		DIFF=$(( $END - $START ))
		echo "Join,${batch},${method},${reducecount},${DIFF}" >> ${log}

		START=$(date +%s)
		sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.child.java.opts=-Xmx4096M -mapper 'cat - ' -file /bin/cat -reducer /usr/bin/uniq -input ${hdfsoutdir} -output /user/aaji/dedupout -numReduceTasks ${reducecount} -verbose -jobconf mapred.job.name="dedup_osmjoin_${reducecount}"  -jobconf mapred.task.timeout=36000000

		END=$(date +%s)
		DIFF=$(( $END - $START ))
		echo "dedup,${batch},${method},${reducecount},${DIFF}" >> ${log}
	    done
	fi
    done
done
# done


# make clean

