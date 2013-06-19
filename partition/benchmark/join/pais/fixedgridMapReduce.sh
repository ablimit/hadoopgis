#! /bin/bash

logfile=fixedgridjoin.log

# reco=$(date +%F-%k-%M)

# make -f Makefile

export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce
sudo -u hdfs hdfs dfs -rm -r /user/aaji/paisjoinout

date >> ${logfile}

for batch in 256 512 768 1024 2048 4096 8192 16384
do
    optinput="-input /user/aaji/partition/pais/fixedgrid/grid${batch}"
    
    for reducecount in 190 140 100 80 60 40 20
    do
	#reducecount=`expr ${maxmap} \\* 5`
	START=$(date +%s)


	sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.child.java.opts=-Xmx4096M -mapper fgmapper -reducer reducer -file fgmapper -file reducer ${optinput} -output /user/aaji/paisjoinout -numReduceTasks ${reducecount} -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="pj_grid_${batch}_${reducecount}" -jobconf mapred.task.timeout=36000000


	END=$(date +%s)
	DIFF=$(( $END - $START ))
	echo "${batch},${reducecount},${DIFF}" >> ${logfile}

	sudo -u hdfs hdfs dfs -rm -r /user/aaji/paisjoinout
    done
done

