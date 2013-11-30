#! /bin/bash

# osm
data=/scratch/data/osm.mbb.norm.filter.dat

opath=/scratch/data/partition/osm
for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
    for algo in rp st rt
    do
        echo -n "${algo},${k}," >> osm.eval.csv
        python evalpartition.sh ${data} < ${opath}/${algo}/c${k}/osm.part >>osm.eval.csv
    done
done

# pais
data=/data2/ablimit/Data/spatialdata/pais/mbb/oligoIII.2.norm.1.dat
opath=/scratch/data/partition/pais

for k in 20 100 200 400 1000 2000 4000 10000 20000 10000
do
    for algo in rp st rt
    do
        echo -n "${algo},${k}," >> pais.eval.csv
        python evalpartition.sh ${data} < ${path}/${algo}/c${k}/pais.part >>pais.eval.csv
    done
done
