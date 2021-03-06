#! /bin/bash

hmkdir="hadoop fs -mkdir "
hput="hadoop fs -copyFromLocal "

for algo in fg rp rt
do
  for c in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062

  do 
    ${hmkdir} -p /user/aaji/partition/osm/${algo}/c${c}
    
    echo "${algo} -- ${c} -- 1"
    
    ${hput} /data2/ablimit/Data/spatialdata/bakup/data/partition/osm/${algo}/c${c}/osm.geom.dat /user/aaji/data/partition/osm/${algo}/c${c}/osm.geom.1.tsv
    
    rc=$?
    if [ $rc -eq 0 ];then
    echo "${algo} -- ${c} -- 1 -- okay"   >> datapush.log
    else
    echo "${algo} -- ${c} -- 1 -- failed" >> datapush.log
      exit $rc ;
    fi

    echo "${algo} -- ${c} -- 2"
    
    ${hput} /data2/ablimit/Data/spatialdata/bakup/data/partition/osm/${algo}/c${c}/osm.geom.2.dat /user/aaji/data/partition/osm/${algo}/c${c}/osm.geom.2.tsv
    rc=$?
    if [ $rc -eq 0 ];then
    echo "${algo} -- ${c} -- 2 -- okay"   >> datapush.log
    else
    echo "${algo} -- ${c} -- 2 -- failed" >> datapush.log
      exit $rc ;
    fi
  done
done

