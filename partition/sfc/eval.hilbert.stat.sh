#! /bin/bash

opath=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm/hc

for i in min center max
do
  continue 
  algo=hc.${i}
  
  for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
  do
    echo "[${k}]"
    echo -n "${algo},${k}," >> osm.hc.stat.csv
    python ../evalpartition.py 86441255 < ${opath}/${i}/c${k}/osm.part >>osm.hc.stat.csv
  done
done


opath=/home/aaji/temp/hc

for image in oligoIII.2
do
  for i in min center max
  do
    algo=hc.${i}
    for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
    do
      echo "[${k}]"
      echo -n "${algo},${k}," >> pais.hc.stat.csv
      python ../evalpartition.py 2031548 < ${opath}/${i}/c${k}/${image}.1.part >>pais.hc.stat.csv
    done
  done
done


