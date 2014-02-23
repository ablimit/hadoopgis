#! /bin/env bash

hdfs dfs -rm -R /user/aaji/temp/sort ;

$HADOOP_COMMON_HOME/bin/hadoop jar $HADOOP_COMMON_HOME/share/hadoop/tools/lib/hadoop-streaming-2.2.0.jar \
  -D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
  -D map.output.key.field.separator=, \
  -D mapred.text.key.comparator.options=-k1,2n \
  -libjars corner-0.0.1.jar \
  -files splitpoints.dat,centerMapper.py,slicer \
  -D mapred.reduce.tasks=865 \
  -input /user/aaji/osm/osm.mbb.norm.filter.dat\
  -output /user/aaji/temp/sort \
  -mapper centerMapper.py \
  -reducer 'slicer 0 10000'\
  -partitioner CornerPartitioner \
  -cmdenv LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aaji/softs/lib

#  -mapper org.apache.hadoop.mapred.lib.IdentityMapper \
# -reducer org.apache.hadoop.mapred.lib.IdentityReducer \

