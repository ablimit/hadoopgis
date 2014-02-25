#!/usr/bin/env bash

# hdfs dfs -rm -R /user/aaji/temp/sort ;

redprog=slc
for n in 50000 100000 200000 500000 860000 ;  
do 
  NUMOFLINES=$(wc -l < "splitpoints.${n}.dat")
  redcount=$((NUMOFLINES + 1))

  cp splitpoints.${n}.dat splitpoints.dat
  # echo ${redcount} ;
  for c in 864 4322 8644 17288 # 43220 86441 172882 # 432206 864412
  do
    START=$(date +%s)
    # $HADOOP_HOME/bin/hadoop jar $HADOOP_HOME/contrib/streaming/hadoop-streaming-1.0.3.jar \
    $HADOOP_COMMON_HOME/bin/hadoop jar $HADOOP_COMMON_HOME/contrib/streaming/hadoop-streaming.jar \
    -D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
    -D map.output.key.field.separator=, \
    -D mapred.text.key.comparator.options=-k1,2n \
    -D mapred.reduce.tasks=${redcount} \
    -libjars corner-0.0.1.jar \
    -files splitpoints.dat,centerMapper.py,${redprog} \
    -partitioner CornerPartitioner \
    -input s3://aaji/data/mbb/osm.mbb.norm.filter.dat \
    -output s3://aaji/scratch/pout/partition/osm/${redprog}/n${n}/c${c} \
    -mapper centerMapper.py \
    -reducer "${redprog} 0 ${c}" \
    -cmdenv LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib

    rc=$?
    if [ $rc -eq 0 ];then
      END=$(date +%s)
      DIFF=$(( $END - $START ))
      echo "${redprog},${n},${c},${DIFF}" >> osm.log
    else
      echo "${redprog},${n},${c},0" >> osm.log
    fi
  done
done

for redprog in bos str
do
  for n in 50000 100000 200000 500000 860000 ;  
  do 
    NUMOFLINES=$(wc -l < "splitpoints.${n}.dat")
    redcount=$((NUMOFLINES + 1))
    # echo ${redcount} ;
    cp splitpoints.${n}.dat splitpoints.dat

    for c in 864 4322 8644 17288 # 43220 86441 172882 # 432206 864412
    do
      START=$(date +%s)
      $HADOOP_COMMON_HOME/bin/hadoop jar $HADOOP_COMMON_HOME/contrib/streaming/hadoop-streaming.jar \
      -D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
      -D map.output.key.field.separator=, \
      -D mapred.text.key.comparator.options=-k1,2n \
      -libjars corner-0.0.1.jar \
      -files splitpoints.dat,centerMapper.py,${redprog} \
      -D mapred.reduce.tasks=${redcount} \
      -input /user/aaji/osm/osm.mbb.norm.filter.dat\
      -output /user/aaji/temp/sort \
      -mapper centerMapper.py \
      -reducer "${redprog} ${c}" \
      -partitioner CornerPartitioner \
      -cmdenv LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib

      rc=$?
      if [ $rc -eq 0 ];then
        END=$(date +%s)
        DIFF=$(( $END - $START ))
        echo "${redprog},${n},${c},${DIFF}" >> osm.log
      else
        echo "${redprog},${n},${c},0" >> osm.log
      fi
    done
  done
done

