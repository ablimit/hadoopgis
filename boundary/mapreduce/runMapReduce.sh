#! /bin/bash

#if [ ! $# == 1 ]; then
#    echo "Usage: $0 [root hdfs output dir]"
#    exit 0
#fi

make -f Makefile

export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce

# factor=2
hdfsoutdir=/user/aaji/boundarytestout
hdfsindir=/user/aaji/paisboundary

date >> dup.log
echo "" >> dup.log

for round in 1
do
    for factor in 2 3 4 5 6 7 8 9 
    # for factor in 2
    do
	for reducecount in 10 20 40 60 80 100 140 200
	# for reducecount in 100
	do
	    sudo -u hdfs hdfs dfs -rm -r ${hdfsoutdir}
	    sudo -u hdfs hdfs dfs -rm -r /user/aaji/dedupout

	    START=$(date +%s)

	    sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.child.java.opts=-Xmx4096M -mapper mapper -reducer reducer -file mapper -file reducer -input ${hdfsindir}/rep${factor}/algo1 -input ${hdfsindir}/rep${factor}/algo2 -output ${hdfsoutdir} -numReduceTasks ${reducecount} -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="dup_join_${reducecount}"  -jobconf mapred.task.timeout=36000000

	    END=$(date +%s)
	    DIFF=$(( $END - $START ))
	    echo "DupStep,${round},${reducecount},${DIFF}" >> dup.log

	    START=$(date +%s)
	    sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.child.java.opts=-Xmx4096M -mapper 'cat - ' -reducer 'sort -u -k1,2 ' /usr/bin/uniq -input ${hdfsoutdir} -output /user/aaji/dedupout -numReduceTasks 1 -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="dedup_join_${reducecount}"  -jobconf mapred.task.timeout=36000000

	    END=$(date +%s)
	    DIFF=$(( $END - $START ))
	    echo "deDupStep,${round},${reducecount},${DIFF}" >> dup.log
	done
    done
done

make clean

