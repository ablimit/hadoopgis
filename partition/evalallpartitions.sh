#! /bin/bash

# osm
data=/scratch/data/osm.mbb.norm.filter.dat

opath=/scratch/data/partition/osm
for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
  for algo in rp rt fg
  do
    echo "[${k}] [${algo}]"
    echo -n "${algo},${k}," >> osm.eval.csv
    python evalpartition.py 86441255 < ${opath}/${algo}/c${k}/osm.part >>osm.eval.csv
  done

  algo=st
  for dim in x y
  do
    echo "[${k}] [${algo}] [${dim}]"
    echo -n "${algo},${dim},${k}," >> osm.eval.csv
    python evalpartition.py 86441255 < ${opath}/${algo}/${dim}/c${k}/osm.part >>osm.eval.csv
  done
done

# pais
data=/data2/ablimit/Data/spatialdata/pais/mbb/oligoIII.2.norm.1.dat
opath=/scratch/data/partition/pais

for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
do
  for algo in rp rt fg
  do
    echo "[${k}] [${algo}]"
    echo -n "${algo},${k}," >> pais.eval.csv
    python evalpartition.py 2031548 < ${opath}/${algo}/c${k}/pais.part >>pais.eval.csv
  done

  algo=st
  for dim in x y
  do
    echo "[${k}] [${algo}] [${dim}]"
    echo -n "${algo},${dim},${k}," >> pais.eval.csv
    python evalpartition.py 2031548 < ${opath}/${algo}/${dim}/c${k}/pais.part >>pais.eval.csv
  done
done

