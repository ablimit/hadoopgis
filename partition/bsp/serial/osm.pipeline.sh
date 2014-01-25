#! /bin/bash

ipath=/shared/data/osm/osm.mbb.norm.filter.dat
prog=./bsp

for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
    ${prog}  --bucket ${k} --input ${ipath}
    
    rc=$?
    if [ ! $rc -eq 0 ];then
      echo -e "\nERROR: partition generation failed."
      exit $rc ;
    fi
done

