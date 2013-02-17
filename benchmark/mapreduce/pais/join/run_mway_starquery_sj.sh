#! /bin/bash

if [ ! $# == 2 ]; then
    echo "Usage: $0 [ joincardinality ] [log_id]"
    exit 0
fi

optinput=""
OUTDIR=/data2/ablimit/hadooplog

case "$1" in
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
    echo "Usage: $0 [ joincardinality=[2|3|4|5|6] ]"
    exit 0

esac

# echo $optinput 
# echo "good"

# reco=$(date +%F-%k-%M)
reco=$2

sudo -u hdfs hdfs dfs -rm -r /user/aaji/joinout

    # 100 80 60 40 20 
for reducecount in 200 
do
    START=$(date +%s)

    sudo -u hdfs hadoop jar hadoop-streaming-2.0.0-mr1-cdh4.0.0.jar -mapper mapper -reducer reducer -file mapper -file reducer ${optinput} -output /user/aaji/joinout -numReduceTasks ${reducecount} -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="j$1_ht_star_${reducecount}"  -jobconf mapred.task.timeout=36000000


    END=$(date +%s)
    DIFF=$(( $END - $START ))
    echo "$1,${reducecount},${DIFF}" >> star.${reco}.log

    # sudo -u hdfs hdfs dfs -copyToLocal /user/aaji/joinout ${OUTDIR}/mjoin_${1}_${reducecount}
    sudo -u hdfs hdfs dfs -rm -r /user/aaji/joinout

done

