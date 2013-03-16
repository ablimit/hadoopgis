#! /bin/bash


logfile=aggr.log

make q2
make aggr

export HADOOP_HOME=/usr/lib/hadoop-0.20-mapreduce

hdfsoutdir=/user/aaji/osmaggrout

optinput="-input /user/aaji/osm/smalltile/planet.dat.1 "

sudo -u hdfs hdfs dfs -rm -r ${hdfsoutdir}

date >> ${logfile}

for query in q2
do
    for j in 1 2 3
    do
	echo "round ${j}"
	for maxmap in 1 2 4
	do
	    expres=`expr ${maxmap} \\* 8`
	    START=$(date +%s)

	    sudo -u hdfs hadoop jar ${HADOOP_HOME}/contrib/streaming/hadoop-streaming-*.jar -D mapred.reduce.tasks=1 -D mapred.tasktracker.map.tasks.maximum=${maxmap} -mapper ${query} -file ${query} -reducer aggr -file aggr ${optinput} -output ${hdfsoutdir}  -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="osm_aggr_${query}_${expres}"  -jobconf mapred.task.timeout=36000000


	    END=$(date +%s)
	    DIFF=$(( $END - $START ))
	    echo "${query},${expres},${DIFF}" >> ${logfile}

	    # sudo -u hdfs hdfs dfs -copyToLocal /user/aaji/joinout ${OUTDIR}/mjoin_${1}_${reducecount}
	    sudo -u hdfs hdfs dfs -rm -r ${hdfsoutdir}
	done

    done
    echo "" >>${logfile}
done

make clean

