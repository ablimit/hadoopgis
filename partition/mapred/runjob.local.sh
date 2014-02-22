#! /bin/env bash

hdfs dfs -rm -R /user/aaji/temp/sort ;

$HADOOP_COMMON_HOME/bin/hadoop jar $HADOOP_COMMON_HOME/share/hadoop/tools/lib/hadoop-streaming-2.2.0.jar \
  -D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
  -D map.output.key.field.separator=, \
  -D mapred.text.key.comparator.options=-k1,2n \
  -libjars corner-0.0.1.jar \
  -files splitpoints.dat,centerMapper.py \
  -D mapred.reduce.tasks=100 \
  -input /user/aaji/osm/osm.mbb.norm.filter.dat\
  -output /user/aaji/temp/sort \
  -mapper centerMapper.py \
  -reducer org.apache.hadoop.mapred.lib.IdentityReducer \
  -partitioner CornerPartitioner

#  -mapper org.apache.hadoop.mapred.lib.IdentityMapper \

