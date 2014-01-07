#! /bin/bash

opath=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm/hc

for i in min center max
do
  algo=hc.${i}
  
  for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
  do
    echo "[${k}]"
    echo -n "${algo},${k}," >> hilbert.eval.csv
    python ../evalpartition.py 86441255 < ${opath}/${i}/c${k}/osm.part >>hilbert.eval.csv
  done
done

