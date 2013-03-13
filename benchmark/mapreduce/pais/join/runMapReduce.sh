#! /bin/bash

if [ ! $# == 1 ]; then
    echo "Usage: $0 [ joincardinality ]"
    exit 0
fi

optinput=""
#OUTDIR=/data2/ablimit/hadooplog

logfile=join.log
card=0

case "$1" in
    2)
    optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2"
    card=2
    ;;

    3)
    optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2 -input /user/aaji/mjoin/algo3"
    card=3
    ;;

    4)
    optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2 -input /user/aaji/mjoin/algo3 -input /user/aaji/mjoin/algo4"
    card=4
    ;;
    5)
    optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2 -input /user/aaji/mjoin/algo3 -input /user/aaji/mjoin/algo4 -input /user/aaji/mjoin/algo5"
    card=5
    ;;
    6)
    optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2 -input /user/aaji/mjoin/algo3 -input /user/aaji/mjoin/algo4 -input /user/aaji/mjoin/algo5 -input /user/aaji/mjoin/algo6"
    card=6
    ;;
    *)
    echo "Usage: $0 [ joincardinality=[2|3|4|5|6] ]"
    exit 0

esac

# echo $optinput 
# echo "good"

# reco=$(date +%F-%k-%M)
# reco=$2

make -f Makefile

export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce
sudo -u hdfs hdfs dfs -rm -r /user/aaji/paisjoinout

date >> ${logfile}

for j in 1 2 3
do
    echo "round ${j}"
    # for reducecount in 200 150 100 80 60 40 20
    for maxmap in 4 2 1
    do
	reducecount=`expr ${maxmap} \\* 8`
	START=$(date +%s)

	sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.tasktracker.map.tasks.maximum=${maxmap} -D mapred.tasktracker.reduce.tasks.maximum=${maxmap} -mapper mapper -reducer reducer -file mapper -file reducer ${optinput} -output /user/aaji/paisjoinout -numReduceTasks ${reducecount} -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="pais_join_${reducecount}" -jobconf mapred.task.timeout=36000000


	END=$(date +%s)
	DIFF=$(( $END - $START ))
	echo "${card},${reducecount},${DIFF}" >> ${logfile}

	# sudo -u hdfs hdfs dfs -copyToLocal /user/aaji/joinout ${OUTDIR}/mjoin_${1}_${reducecount}
	sudo -u hdfs hdfs dfs -rm -r /user/aaji/paisjoinout
    done
    echo "" >> ${logfile}
done


make clean
