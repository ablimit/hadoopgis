#! /bin/bash

# osm
geompath=/data2/ablimit/Data/spatialdata/pais/geom

for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
  for i in 1 2
  do
    echo "[${image}] [${i}]" 
    fgrep "|${image}|${i}|" /dev/shm/pais.geom.dat  > ${geompath}/${image}.${i}.dat
  done

done

