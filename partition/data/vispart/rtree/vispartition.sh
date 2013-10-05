#! /bin/bash

for tid in 2 3 4 5
do
    python drawSinglePartition.py ${PWD}/pics/test.${tid}.png meta/regionmbb.${tid}.txt ../testdata/test${tid}.obj.txt > meta/plot.${tid}.gpl
    gnuplot meta/plot.${tid}.gpl
done

