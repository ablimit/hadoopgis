#! /bin/bash

if [ ! -d pics ];
then 
  mkdir -p pics
fi

if [ ! -d meta ];
then 
  mkdir -p meta
fi

for tid in 2 3 4 5 6 7
do
  if [ -e ../testdata/test${tid}.binarysplit.part.txt ] ;
  then 
    cat ../testdata/test${tid}.binarysplit.part.txt | cut -f1,2,3,4,5 | expand  -t 1 > meta/regionmbb.${tid}.txt
    
    python ../drawSinglePartition.py ${PWD}/pics/test.${tid}.eps meta/regionmbb.${tid}.txt ../testdata/test${tid}.obj.txt > meta/plot.${tid}.gpl
    gnuplot meta/plot.${tid}.gpl
  fi
done

