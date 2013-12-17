#! /bin/bash

# osm
geompath=/data2/ablimit/Data/spatialdata/pais/geom

opath=/scratch/data/partition/pais

logg=pais.remap.log
# > remap.log
for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
  for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
  do

  for algo in rp rt fg
  do
    echo "[${k}] [${algo}]"
    if [ ! -e ${opath}/${algo}/c${k}/${image}.geom.1.dat ] ;
    then 
      echo "[${k}] [${algo}]" >> ${logg}
      python remaptogeom.py ${opath}/${algo}/c${k}/${image}.part < ${geompath}/${image}.1.dat > ${opath}/${algo}/c${k}/${image}.geom.1.dat
    fi
  done

  algo=st
  echo "[${k}] [${algo}]"
  if [ ! -e ${opath}/${algo}/x/c${k}/${image}.geom.1.dat ] ;
  then
    echo "[${k}] [${algo}]"   >> ${logg}
    python remaptogeom.py ${opath}/${algo}/x/c${k}/${image}.part < ${geompath}/${image}.1.dat > ${opath}/${algo}/x/c${k}/${image}.geom.1.dat
  fi
done

