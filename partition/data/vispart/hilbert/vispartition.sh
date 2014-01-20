#! /bin/bash

if [ ! -d pics ];
then 
    mkdir -p pics
fi

for tid in 2 3 4 5 6 7
do
    python ../drawSinglePartition.py ${PWD}/pics/test.${tid}.eps meta/regionmbb.${tid}.txt ../testdata/test${tid}.obj.txt > meta/plot.${tid}.gpl
    gnuplot meta/plot.${tid}.gpl
done

