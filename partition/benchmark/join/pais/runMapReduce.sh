#! /bin/bash

logfile=join.log

# reco=$(date +%F-%k-%M)

# make -f Makefile

export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce
sudo -u hdfs hdfs dfs -rm -r /user/aaji/paisjoinout

date >> ${logfile}

for batch in oc2500 oc5000 oc10000 oc15000 oc20000 oc25000 oc30000 oc50000
do
    for method in minskew rkHist rv rtree
    do
        for reducecount in 190 140 100 80 60 40 20
        do
            #reducecount=`expr ${maxmap} \\* 5`
            START=$(date +%s)

            optinput="-input /user/aaji/partition/pais/${method}/${batch}"
            
            sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.child.java.opts=-Xmx4096M -mapper mapper -reducer reducer -file mapper -file reducer ${optinput} -output /user/aaji/paisjoinout -numReduceTasks ${reducecount} -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="pj_${method}_${batch}_${reducecount}" -jobconf mapred.task.timeout=36000000


            END=$(date +%s)
            DIFF=$(( $END - $START ))
            echo "${batch},${method},${reducecount},${DIFF}" >> ${logfile}

            # sudo -u hdfs hdfs dfs -copyToLocal /user/aaji/joinout ${OUTDIR}/mjoin_${1}_${reducecount}
            sudo -u hdfs hdfs dfs -rm -r /user/aaji/paisjoinout
        done
    done
done

