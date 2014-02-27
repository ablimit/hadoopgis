#! /usr/bin/env bash

for i in 6 12 25 50
do
  #for d in 5 10 20 50 86
  for d in 50
  do
    echo -n "${i} " > levels.${i}.dat
    cat osm.${i}node.csv | grep ,${d}0000, | cut -d, -f1,4 | python gendata.py >> levels.${i}.dat
    #cat ../osm.csv | grep ,${d}0000, | grep slc  | cut -d, -f3,4 | sort -t, -nk1  > slc.dat
    #cat ../osm.csv | grep ,${d}0000, | grep bos  | cut -d, -f3,4 | sort -t, -nk1 | cut -d, -f2 > bos.dat
    #cat ../osm.csv | grep ,${d}0000, | grep str  | cut -d, -f3,4 | sort -t, -nk1 | cut -d, -f2  > str.dat
    # paste -d " " slc.dat bos.dat str.dat > d${d}.dat
  done
done

