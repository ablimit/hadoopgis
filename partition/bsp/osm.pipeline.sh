#! /bin/bash

dir=/shared/data/osm
osmdata=${dir}/osm.mbb.norm.filter.dat
regionpath=/data2/ablimit/Data/spatialdata/bakup/data/partition/samp/osm
datapath=${dir}/partition/samp/osm


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
  # ${prog}  --bucket ${k} --input ${ipath} > ${opath}/c${k}/regionmbb.txt
  
  src=${regionpath}/${algo}/f${f}/c${k}
  dest=${datapath}/${algo}/f${f}/c${k}

  if [ ! -e ${src}/regionmbb.txt ] ;
  then
    continue ;
  fi

  mkdir -p ${dest}

  echo "0.${f} --------- ${k} -------------- ${algo}"

  echo -e "---------------------------------------------"
  echo "generate pid oid mapping ...."
  ../rquery ${src}/regionmbb.txt ${tempPath}/spatial  > ${tempPath}/pidoid.txt
  rc=$?
  if [ $rc -eq 0 ];then
    echo ""
  else
    echo -e "\nERROR: rqueryfailed."
    exit $rc ;
  fi

  echo -e "\n---------------------------------------------"
  echo "remapping objects"
  python ../mappartition.py ${tempPath}/pidoid.txt < ${osmdata} > ${dest}/osm.part

  rm -f ${tempPath}/pidoid.txt

done

