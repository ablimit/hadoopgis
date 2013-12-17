#! /bin/bash

# osm

opath=/scratch/data/partition/osm
dpath=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm

# > remap.log

for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
  for algo in rp rt fg
  do
    echo "[${k}] [${algo}]"
    if [ -e ${opath}/${algo}/c${k}/osm.geom.dat.gz ] ;
    then 
      zcat ${opath}/${algo}/c${k}/osm.geom.dat.gz | ./sample.py 0.5 > ${dpath}/${algo}/c${k}/osm.geom.2.dat
    fi
  done

  algo=st
  echo "[${k}] [${algo}]"
  if [ ! -e ${opath}/${algo}/x/c${k}/osm.geom.dat.gz ] ;
  then
    zcat ${opath}/${algo}/x/c${k}/osm.geom.dat.gz | ./sample.py 0.5 > ${dpath}/${algo}/x/c${k}/osm.geom.2.dat
  fi
done

