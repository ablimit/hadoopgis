#! /bin/bash

for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
do
    echo "batch ${k} "

  for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
  do
    echo "image: ${image}"
    echo "-----------------------------"
    cat /data/ablimit/Data/spatialdata/bakup/data/partition/pais/bsp/c${k}/${image}.geom.1.dat >> /data/ablimit/Data/spatialdata/bakup/data/partition/pais/bsp/c${k}/pais.geom.1.dat 
    cat /data/ablimit/Data/spatialdata/bakup/data/partition/pais/bsp/c${k}/${image}.geom.2.dat >> /data/ablimit/Data/spatialdata/bakup/data/partition/pais/bsp/c${k}/pais.geom.2.dat

  done
done

