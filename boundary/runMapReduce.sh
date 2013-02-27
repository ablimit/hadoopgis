#! /bin/bash

	if [ ! $# == 3 ]; then
    echo "Usage: [ joincardinality ] [ namelogfile] [sourcesuffix] "
    exit 0
fi
optinput=""
OUTDIR=/data2/hoang
SUFFIX=$3

case "$1" in
    2)
    optinput="-input /user/hvo8/mjoin/algo1b${SUFFIX} -input /user/hvo8/mjoin/algo2b${SUFFIX}"
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
mappername=mapperStarDup
reducername=reducerStarDup
outputHdfs=/user/hvo8/outputDup

nameRemover=mapperRemove
finaloutputDir=/user/hvo8/outputFinal

reco=$2

sudo -u hdfs hdfs dfs -rm -r ${outputHdfs}
sudo -u hdfs hdfs dfs -rm -r ${finaloutputDir}

for reducecount in 200
#for reducecount in 20 60 80 100 140 200
do

    START=$(date +%s)

    sudo -u hdfs hadoop jar hadoop-streaming-2.0.0-mr1-cdh4.0.0.jar -mapper ${mappername} -reducer ${reducername} -file ${mappername} -file ${reducername} ${optinput} -output ${outputHdfs} -numReduceTasks ${reducecount}  -verbose  -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="j$1_pbsm_str_${reducecount}"  -jobconf mapred.task.timeout=36000000

    END=$(date +%s)
    START2=$(date +%s)   
 
    sudo -u hdfs hadoop jar hadoop-streaming-2.0.0-mr1-cdh4.0.0.jar -mapper ${nameRemover} -reducer ${reducername} -file ${nameRemover} -file ${reducername} -input ${outputHdfs} -output ${finaloutputDir} -numReduceTasks 0  -verbose  -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH -jobconf mapred.job.name="j$1_pbsm_str_${reducecount}" -jobconf mapred.max.maps.per.node=1  -jobconf mapred.task.timeout=36000000
    END2=$(date +%s)

    DIFF=$(( $END - $START ))
    DIFF2=$(( $END2 - $START2 ))

    echo "$1,${reducecount},phase1:${DIFF}" >> pbsm.star.${reco}.dup.${SUFFIX}.log

    echo "$1,${reducecount},phase2:${DIFF2}" >> pbsm.star.${reco}.dup.${SUFFIX}.log
    sudo -u hdfs hdfs dfs -rm -r ${outputHdfs}
   sudo -u hdfs hdfs dfs -rm -r ${finaloutputDir}
done

