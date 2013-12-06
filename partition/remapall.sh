#! /bin/bash

# osm
geomdata=/dev/shm/osm.planet.dat

opath=/scratch/data/partition/osm

# > remap.log

for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
  for algo in rp rt fg
  do
    echo "[${k}] [${algo}]"
    if [ ! -e ${opath}/${algo}/c${k}/osm.geom.dat.gz ] ;
    then 
      echo "[${k}] [${algo}]" >> remap.log
      python remaptogeom.py ${opath}/${algo}/c${k}/osm.part < ${geomdata} | gzip > ${opath}/${algo}/c${k}/osm.geom.dat.gz
    fi
  done

  algo=st
  echo "[${k}] [${algo}]"
  if [ ! -e ${opath}/${algo}/x/c${k}/osm.geom.dat.gz ] ;
  then
    echo "[${k}] [${algo}]"   >> remap.log
    python remaptogeom.py ${opath}/${algo}/x/c${k}/osm.part < ${geomdata} | gzip > ${opath}/${algo}/x/c${k}/osm.geom.dat.gz
  fi
done

