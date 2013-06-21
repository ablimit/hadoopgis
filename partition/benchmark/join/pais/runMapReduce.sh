#! /bin/bash

logfile=join.log

# reco=$(date +%F-%k-%M)

make -f Makefile

hdfsoutdir=/user/aaji/paisjoinout

export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce

date >> ${logfile}

for batch in oc2500 oc5000 oc10000 oc15000 oc20000 oc25000 oc30000 oc50000
do
    for method in minskew rkHist rv rtree
    do
        for reducecount in 140 40
        do
            optinput="-input /user/aaji/partition/pais/${method}/${batch}"
            
	    sudo -u hdfs hdfs dfs -rm -r ${hdfsoutdir}
	    sudo -u hdfs hdfs dfs -rm -r /user/aaji/dedupout

	    START=$(date +%s)

            sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.child.java.opts=-Xmx4096M -mapper mapper -reducer reducer -file mapper -file reducer ${optinput} -output ${hdfsoutdir} -numReduceTasks ${reducecount} -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="pj_${batch}_${method}_${reducecount}" -jobconf mapred.task.timeout=36000000

            echo "join,${batch},${method},${reducecount},${DIFF}" >> ${logfile}

	    START=$(date +%s)
	    sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.child.java.opts=-Xmx4096M -mapper 'cat - ' -file /bin/cat -reducer /usr/bin/uniq -input ${hdfsoutdir} -output /user/aaji/dedupout -numReduceTasks ${reducecount} -verbose -jobconf mapred.job.name="de_pj_${batch}_${method}_${reducecount}"  -jobconf mapred.task.timeout=36000000

	    END=$(date +%s)
	    DIFF=$(( $END - $START ))
            echo "dedup,${batch},${method},${reducecount},${DIFF}" >> ${logfile}
            
        done
    done
done

# make clean

