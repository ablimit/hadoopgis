#! /bin/bash

for batch in oc2500 oc5000 oc10000 oc15000 oc20000 oc25000 oc30000 oc50000
do

    python getfacet.py ${batch} < ../join.log > ${batch}.csv

done

for method in minskew rkHist rv rtree
do
    grep "${method}" ../join.log | python getmethod.py > ${method}.csv 
done
