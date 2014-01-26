#! /bin/bash

# ipath=/home/aaji/proj/data/osm/osm.mbb.norm.filter.dat
dir=/home/aaji/proj/data/sampling/osm
opath=/home/aaji/proj/data/partition/samp/osm

for f in 01 05 10 15 20 25 
do
  for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
  do
    for algo in fg bsp hc slc bos str
    do
      echo "0.${f}% --- ${k} ---- ${algo}"
      
      prog=algo/${algo}
      
      ipath=${dir}/osm.sample.${f}.dat

      dest=${opath}/${algo}/f${f}/c${k}
      
      mkdir -p ${dest}

      b=$(echo "(${k} * 0.${f}+0.5)/1" | bc ) # | cut -d"." -f1)
      
      #echo "${k} x 0.${f} = ${b} "
      
      ${prog} -i ${ipath} -b ${b} > ${dest}/regionmbb.txt

      rc=$?
      if [ ! $rc -eq 0 ];then
        echo "ERROR: partition generation failed. "
        exit $rc ;
      fi
    done
  done
done

