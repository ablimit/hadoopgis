#! /usr/bin/env bash

for d in 5 10 20 50 86
do
  echo -n "${d}0000 " >> levels.dat
  cat ../osm.csv | grep ,${d}0000, | cut -d, -f1,4 | python gendata.py >> levels.dat
  #cat ../osm.csv | grep ,${d}0000, | grep slc  | cut -d, -f3,4 | sort -t, -nk1  > slc.dat
  #cat ../osm.csv | grep ,${d}0000, | grep bos  | cut -d, -f3,4 | sort -t, -nk1 | cut -d, -f2 > bos.dat
  #cat ../osm.csv | grep ,${d}0000, | grep str  | cut -d, -f3,4 | sort -t, -nk1 | cut -d, -f2  > str.dat
  # paste -d " " slc.dat bos.dat str.dat > d${d}.dat
done

