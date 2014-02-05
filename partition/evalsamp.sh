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
done
