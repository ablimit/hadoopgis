#!/usr/bin/env bash

# hdfs dfs -rm -R /user/aaji/temp/sort ;

redprog=slc
for n in 50000 100000 200000 500000 860000 ;  
do 
  NUMOFLINES=$(wc -l < "splitpoints.${n}.dat")
  redcount=$((NUMOFLINES + 1))
  # echo ${redcount} ;
  for c in 864 4322 8644 17288 # 43220 86441 172882 # 432206 864412
  do
    $HADOOP_COMMON_HOME/bin/hadoop jar $HADOOP_COMMON_HOME/share/hadoop/tools/lib/hadoop-streaming-2.2.0.jar \
      -D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
      -D map.output.key.field.separator=, \
      -D mapred.text.key.comparator.options=-k1,2n \
      -libjars corner-0.0.1.jar \
      -files splitpoints.${n}.dat,centerMapper.py,${redprog} \
      -D mapred.reduce.tasks=${redcount} \
      -input s3://aaji/data/mbb/osm.mbb.norm.filter.dat \
      -output s3://aaji/scratch/pout/partition/osm/${redprog}/n${n}/c${c}
      -mapper centerMapper.py \
      -reducer "${redprog} 0 ${c}" \
      -partitioner CornerPartitioner
  done
done

exit 0;

for redprog in bos str
do
  for n in 50000 100000 200000 500000 860000 ;  
  do 
    NUMOFLINES=$(wc -l < "splitpoints.${n}.dat")
    redcount=$((NUMOFLINES + 1))
    # echo ${redcount} ;
    
    for c in 864 4322 8644 17288 # 43220 86441 172882 # 432206 864412
    do
      $HADOOP_COMMON_HOME/bin/hadoop jar $HADOOP_COMMON_HOME/share/hadoop/tools/lib/hadoop-streaming-2.2.0.jar \
        -D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
        -D map.output.key.field.separator=, \
        -D mapred.text.key.comparator.options=-k1,2n \
        -libjars corner-0.0.1.jar \
        -files splitpoints.${n}.dat,centerMapper.py,${redprog} \
        -D mapred.reduce.tasks=${redcount} \
        -input /user/aaji/osm/osm.mbb.norm.filter.dat\
        -output /user/aaji/temp/sort \
        -mapper centerMapper.py \
        -reducer "${redprog} ${c}" \
        -partitioner CornerPartitioner
    done
  done
done

