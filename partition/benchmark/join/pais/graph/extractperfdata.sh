#! /bin/bash

for batch in oc2500 oc5000 oc10000 oc15000 oc20000 oc25000 oc30000 oc50000
do

    python getfacet.py ${batch} < join.wbo.log > ${batch}.csv
    # rm ${batch}.csv

done

for method in minskew rkHist rv rtree
do
    echo "${method}"
    # grep "${method}" join.log | python getmethod.py > ${method}.csv 
done
