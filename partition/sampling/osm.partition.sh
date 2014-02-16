#! /bin/bash

# ipath=/home/aaji/proj/data/osm/osm.mbb.norm.filter.dat
dir=/data2/ablimit/Data/spatialdata/bakup/data/partition/sampdata/osm
opath=/data/ablimit/Data/spatialdata/bakup/data/partition/samp/osm

for algo in bsp slc bos #str #hc 
do
  if [ ! -e parr/region.${algo} ] ;
  then
    # touch parr/region.${algo} ;
    # for f in 01 05 10 15 20 25 
    for f in 0001 0005 001 0015 0020 0025
    do
      for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
      do
        # echo "0.${f}% --- ${k} ---- ${algo}"

        prog=algo/${algo}

        ipath=${dir}/osm.sample.${f}.dat

        dest=${opath}/${algo}/f${f}/c${k}

        mkdir -p ${dest}

        b=$(echo "(${k} * 0.${f}+0.5)/1" | bc ) # | cut -d"." -f1)

        if [ "${b}" -lt "1" ] ;then 
          continue ;
        fi
        
        echo "${k} x 0.${f} = ${b} "
        ${prog} -i ${ipath} -b ${b} > ${dest}/regionmbb.txt

        rc=$?
        if [ ! $rc -eq 0 ];then
          echo "ERROR: partition generation failed. "
          # exit $rc ;
        fi
      done
    done
  fi
done

