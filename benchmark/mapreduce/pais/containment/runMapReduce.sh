#! /bin/bash

if [ ! $# == 1 ]; then
    echo "Usage: $0 [ input_size ]"
    exit 0
fi

logfile=cont.log

case "$1" in
    1)
	optinput="-input /user/aaji/mjoin/algo1"
	;;

    2)
	optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2"
	;;

    3)
	optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2 -input /user/aaji/mjoin/algo3"
	;;

    4)
	optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2 -input /user/aaji/mjoin/algo3 -input /user/aaji/mjoin/algo4"
	;;
    5)
	optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2 -input /user/aaji/mjoin/algo3 -input /user/aaji/mjoin/algo4 -input /user/aaji/mjoin/algo5"
	;;
    6)
	optinput="-input /user/aaji/mjoin/algo1 -input /user/aaji/mjoin/algo2 -input /user/aaji/mjoin/algo3 -input /user/aaji/mjoin/algo4 -input /user/aaji/mjoin/algo5 -input /user/aaji/mjoin/algo6"
	;;
    *)
	echo "Usage: $0 [ input_size=[2|3|4|5|6] ]"
	exit 0

esac


make q3
make qc

export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce
sudo -u hdfs hdfs dfs -rm -r /user/aaji/paiscontout

date >> ${logfile}

for query in q3 qc
do
    for j in 1 2 3
    do
	for maxmap in 20 10 6 4 2 1
	do
	    echo "round ${j}"
	    reducecount=`expr ${maxmap} \\* 5`
	    START=$(date +%s)

	    sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.reduce.tasks=0 -D mapred.tasktracker.map.tasks.maximum=${maxmap} -mapper ${query} -file ${query} ${optinput} -output /user/aaji/paiscontout -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="pais_cont_$1_${query}_${reducecount}"  -jobconf mapred.task.timeout=36000000


	    END=$(date +%s)
	    DIFF=$(( $END - $START ))
	    echo "${query},$1,${reducecount},${DIFF}" >> ${logfile}

	    # sudo -u hdfs hdfs dfs -copyToLocal /user/aaji/joinout ${OUTDIR}/mjoin_${1}_${reducecount}
	    sudo -u hdfs hdfs dfs -rm -r /user/aaji/paiscontout
	done

    done
    echo "" >>${logfile}
done

make clean



