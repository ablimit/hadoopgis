#! /bin/bash

for algo in fg rp rt
do
  for c in 20 100 200 400 1000 2000 4000 10000 20000 100000
  do
    echo "${algo} -- ${c} -- 1"

    cat /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/${algo}/c${c}/*.geom.1.dat > /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/${algo}/c${c}/pais.geom.1.tsv 

    echo "${algo} -- ${c} -- 2"
    cat /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/${algo}/c${c}/*.geom.2.dat > /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/${algo}/c${c}/pais.geom.2.tsv 

  done
done

for c in 20 100 200 400 1000 2000 4000 10000 20000 100000
do
  echo "1 -- ${c}"
  cat /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/st/x/c${c}/*.geom.1.dat > /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/s3/st/c${c}/pais.geom.1.tsv 

  echo "2 -- ${c}"
  cat /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/st/x/c${c}/*.geom.2.dat > /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/s3/st/c${c}/pais.geom.2.tsv 

done

