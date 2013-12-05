#! /bin/bash

# osm
geomdata=/data2/ablimit/Data/spatialdata/osmout/osm.planet.tar.gz

opath=/scratch/data/partition/osm

for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
  for algo in rp rt fg
  do
    echo "[${k}] [${algo}]"
    python remaptogeom.py ${geomdata} < ${opath}/${algo}/c${k}/osm.part > ${opath}/${algo}/c${k}/osm.geom.dat
  done

  algo=st
  echo "[${k}] [${algo}]"
  python remaptogeom.py ${geomdata} < ${opath}/${algo}/x/c${k}/osm.part > ${opath}/${algo}/x/c${k}/osm.geom.dat
done

