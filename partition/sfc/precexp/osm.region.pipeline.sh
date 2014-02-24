#! /bin/bash

ipath=/home/aaji/proj/data/osm/osm.mbb.norm.filter.dat
prog=./hc
param="864 4322 8644 17288 43220 86441 172882 432206 864412 4322062"

for p in 4 8 10 12 14 16 18 20 25 30
do
  echo "--------------------------------"
  echo ${dest}

  ${prog}  --prec ${p} --bucket ${param} --input ${ipath} # > ${dest}/regionmbb.txt
  rc=$?
  if [ ! $rc -eq 0 ];then
    echo -e "\nERROR: partition generation failed."
    exit $rc ;
  fi

  for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
  do
    dest=/home/aaji/proj/data/prec/p${p}/c${k}
    mkdir -p ${dest}
    mv c${k}.txt ${dest}/regionmbb.txt
  done
done

