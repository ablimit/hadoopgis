#! /bin/bash

#if [ ! $# == 1 ]; then
#    echo "Usage: $0 [log_id]"
#    exit 0
#fi

make -f Makefile

reco=$(date +%F-%k-%M)
# reco=$1
export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce
sudo -u hdfs hdfs dfs -rm -r /user/aaji/osmjoinout

for j in 1 2 3 4 5 6 7 8 9 10
do
    date >> osm.${reco}.log
    echo "round ${j}"
    for reducecount in 200 150 100 80 60 40 20
    do
	START=$(date +%s)

	sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -mapper mapper -reducer reducer -file mapper -file reducer -input /user/aaji/osm -output /user/aaji/osmjoinout -numReduceTasks ${reducecount} -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="osm_join_${reducecount}"  -jobconf mapred.task.timeout=36000000


	END=$(date +%s)
	DIFF=$(( $END - $START ))
	echo "${reducecount},${DIFF}" >> osm.${reco}.log

	# sudo -u hdfs hdfs dfs -copyToLocal /user/aaji/joinout ${OUTDIR}/mjoin_${1}_${reducecount}
	sudo -u hdfs hdfs dfs -rm -r /user/aaji/osmjoinout
    done
    echo -e "\n\n" >>osm.${reco}.log
done

