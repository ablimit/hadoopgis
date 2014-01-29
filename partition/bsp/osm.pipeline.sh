#! /bin/bash

ipath=/data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.norm.filter.dat
opath=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm/bsp
tempPath=/dev/shm/osm/bsp
mkdir -p ${tempPath}

prog=./serial/bsp

for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
  if [ ! -e ${opath}/c${k} ] ;
  then
    mkdir -p ${opath}/c${k}
  fi
  echo "----------------------------------------"
  ${prog}  --bucket ${k} --input ${ipath} > ${opath}/c${k}/regionmbb.txt

done

