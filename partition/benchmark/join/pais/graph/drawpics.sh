#! /bin/bash

for batch in oc2500 oc5000 oc10000 oc15000 oc20000 oc25000 oc30000 oc50000
do
    cp template_a.gnu ${batch}.gnu
    perl -p -i -e "s/pbsm/${batch}/g" ${batch}.gnu
    gnuplot ${batch}.gnu

done

for method in minskew rkHist rv rtree
do
    cp template_b.gnu ${method}.gnu
    perl -p -i -e "s/pbsm/${method}/g" ${method}.gnu
    gnuplot ${method}.gnu
done

