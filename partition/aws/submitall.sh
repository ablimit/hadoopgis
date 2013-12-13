#! /bin/bash

jobid="j-7E8BSU9IRXYU"

elastic-mapreduce -j j-7E8BSU9IRXYU --wait-for-steps

for c in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
  for algo in rp rt fg
  do
    echo "[${c}] [${algo}]"
    /usr/local/emrcli/elastic-mapreduce --jobflow ${jobid} --stream --mapper 's3://aaji/scratch/awsjoin/tagmapper.py osm.geom.dat osm.geom.2.dat' --reducer "s3://aaji/scratch/deps/bins/resque st_intersects 1 1" --input "s3://aaji/data/partitions/osm/${algo}/c${c}" --output s3://aaji/temp/${algo}c${c} --jobconf mapred.reduce.tasks=160 
  done

  algo=st
  echo "[${c}] [${algo}]"
  /usr/local/emrcli/elastic-mapreduce --jobflow ${jobid} --stream --mapper 's3://aaji/scratch/awsjoin/tagmapper.py osm.geom.dat osm.geom.2.dat' --reducer "s3://aaji/scratch/deps/bins/resque st_intersects 1 1" --input "s3://aaji/data/partitions/osm/${algo}/c${c}" --output s3://aaji/temp/${algo}c${c} --jobconf mapred.reduce.tasks=160 

done

