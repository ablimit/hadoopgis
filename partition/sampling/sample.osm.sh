#! /bin/bash

#for f in 01 05 10 15 20 25 
for f in 0001 0005 001 0015 0020 0025
do
  echo "sampling ${f}%"
  cat  /data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.norm.filter.dat | awk "BEGIN {srand()} !/^$/ { if (rand() <= .${f}) print \$0}" > /data2/ablimit/Data/spatialdata/bakup/data/partition/sampdata/osm/osm.sample.${f}.dat
done

