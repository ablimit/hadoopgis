#! /bin/bash

dir=/shared/data/osm
osmdata=${dir}/osm.mbb.norm.filter.dat
datapath=${dir}/partition/bsp


ipath=/data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.norm.filter.dat
opath=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm/bsp
tempPath=/dev/shm/osm/bsp
indexPath=/dev/shm
mkdir -p ${tempPath}

prog=./serial/bsp

for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do

  if [ ! -e ${opath}/c${k} ] ;
  then
    mkdir -p ${opath}/c${k}
  fi

  # ${prog}  --bucket ${k} --input ${ipath} > ${opath}/c${k}/regionmbb.txt

  if [ ! -e ${opath}/c${k}/regionmbb.txt ] ;
  then
    continue ;
  fi

  # dest=${datapath}/c${k}
  # mkdir -p ${dest}

  echo "--------- ${k} --------------"

  echo -e "---------------------------------------------"
  echo "generate pid oid mapping ...."
  ../rquery ${opath}/c${k}/regionmbb.txt ${indexPath}/spatial  > ${tempPath}/pidoid.txt
  rc=$?
  if [ $rc -eq 0 ];then
    echo ""
  else
    echo -e "\nERROR: rqueryfailed."
    exit $rc ;
  fi

  echo -e "\n---------------------------------------------"
  echo "remapping objects"
  python ../mappartition.py ${tempPath}/pidoid.txt < ${ipath} > ${opath}/c${k}/osm.part

  rm -f ${tempPath}/pidoid.txt

done

