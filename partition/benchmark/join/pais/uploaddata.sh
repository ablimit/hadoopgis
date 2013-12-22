#! /bin/bash

hmkdir="hadoop fs -mkdir "
hput="hadoop fs -copyFromLocal "

for algo in fg rp rt
do
  for c in 20 100 200 400 1000 2000 4000 10000 20000 100000
  do
    ${hmkdir} -p /user/aaji/partition/pais/${algo}/c${c}
    
    echo "${algo} -- ${c} -- 1"
    
    ${hput} /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/${algo}/c${c}/pais.geom.1.tsv /user/aaji/data/partition/pais/${algo}/c${c}/
    
    rc=$?
    if [ $rc -eq 0 ];then
    echo "${algo} -- ${c} -- 1 -- okay"   >> datapush.log
    else
    echo "${algo} -- ${c} -- 1 -- failed" >> datapush.log
      exit $rc ;
    fi

    echo "${algo} -- ${c} -- 2"
    
    ${hput} /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/${algo}/c${c}/pais.geom.2.tsv /user/aaji/data/partition/pais/${algo}/c${c}/
    rc=$?
    if [ $rc -eq 0 ];then
    echo "${algo} -- ${c} -- 2 -- okay"   >> datapush.log
    else
    echo "${algo} -- ${c} -- 2 -- failed" >> datapush.log
      exit $rc ;
    fi
  done
done

